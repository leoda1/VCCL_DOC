# SM Free & Overlap

SM Free & Overlap is VCCL's innovative feature that ensures P2P communication operations do not occupy GPU Streaming Multiprocessor (SM) resources, achieving true overlap between communication and computation, significantly improving GPU utilization and training efficiency.

## Feature Overview

### What is SM Free?
In traditional NCCL implementations, communication operations typically require GPU SM resources to execute kernel functions, which competes with computational tasks for SM resources, leading to:
- **Resource Competition**: Communication and computation compete for SM resources
- **Pseudo Overlap**: Appears parallel but actually executes serially
- **Reduced Efficiency**: Decreased GPU utilization

VCCL's SM Free mode achieves true overlap through the following technologies:
- **Dedicated Communication Path**: Communication operations use Memcpy path instead of kernels
- **Zero SM Occupation**: Communication consumes no SM resources
- **True Parallelism**: P2P communication and computation can execute completely in parallel through higher bandwidth kernel-free zerocopy mode

## How It Works

### Technical Architecture
```
Traditional Mode:
GPU SM ←→ [Compute Kernel] ←Competition→ [Communication Kernel]

SM Free Mode:
GPU SM ←→ [Compute Kernel]
GPU DMA ←→ [Communication Hardware] (Independent Path)
```

## Training Performance Testing

### Overview
Regression testing for training performance improvement using VCCL zerocopy.

### Preparation
#### Required Modifications for Bond0 Clusters
For clusters with bond0 configuration, VCCL needs to be modified to enable load balancing across two ports. Apply the following patch to VCCL:

**Key Changes**:
1. **Makefile**: Added `-lmlx5` library linking
2. **net_ib.cc**: Added MLX5 QP modification functions for LAG port configuration

```cpp
diff --git a/src/Makefile b/src/Makefile
index 65da630..e420eea 100644
--- a/src/Makefile
+++ b/src/Makefile
@@ -52,7 +52,7 @@ PKGTARGET  := $(PKGCONFIGFILE)
 LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)
 BINOBJ     := $(BINSRCFILES:%.cc=$(OBJDIR)/%.o)
 DEPFILES   := $(LIBOBJ:%.o=%.d) $(BINOBJ:%.o=%.d)
-LDFLAGS    += -L${CUDA_LIB} -l$(CUDARTLIB) -lpthread -lrt -ldl
+LDFLAGS    += -L${CUDA_LIB} -l$(CUDARTLIB) -lpthread -lrt -ldl -lmlx5
 INCPLUGIN  := include/plugin
 
 DEVMANIFEST := $(BUILDDIR)/obj/device/manifest
diff --git a/src/transport/net_ib.cc b/src/transport/net_ib.cc
index 0540e1e..9892742 100644
--- a/src/transport/net_ib.cc
+++ b/src/transport/net_ib.cc
@@ -27,7 +27,11 @@
 #include "ibvwrap.h"
 #define NET_IB_CC
 #include "timer_log.h"
+extern "C"
+{
+  int mlx5dv_modify_qp_udp_sport(struct ibv_qp *qp, uint32_t udp_sport);
+  int mlx5dv_modify_qp_lag_port(struct ibv_qp *qp, uint8_t port_num);
+}
 #define MAXNAMESIZE 64
 static char ncclIbIfName[MAX_IF_NAME_SIZE+1];
 static union ncclSocketAddress ncclIbIfAddr;
@@ -1661,7 +1665,7 @@ ib_connect:
     }
   }
   comm->base.nRemDevs = remMeta.ndevs;
+  static int channel_loop = 0;
   for (int q = 0; q < comm->base.nqps; q++) {
     struct ncclIbQpInfo* remQpInfo   = remMeta.qpInfo + q;
     struct ncclIbDevInfo* remDevInfo = remMeta.devs + remQpInfo->devIndex;
@@ -1686,7 +1690,20 @@ ib_connect:
     remDevInfo->mtu = std::min(remDevInfo->mtu, ibDev->portAttr.active_mtu);
     NCCLCHECKGOTO(ncclIbRtrQp(qp, &commDev->base.gidInfo, remQpInfo->qpn, remDevInfo, false, remMeta.tc, remMeta.sl), ret, fail);
     NCCLCHECKGOTO(ncclIbRtsQp(qp), ret, fail);
+    mlx5dv_modify_qp_lag_port(qp, channel_loop % 2 + 1);
+    channel_loop++;
    memcpy(&comm->base.qps[q].gidInfo, &commDev->base.gidInfo, sizeof(struct ncclIbGidInfo));
    comm->base.qps[q].dest_qp_num = remQpInfo->qpn;
    memcpy(&comm->base.qps[q].info, remDevInfo, sizeof(struct ncclIbDevInfo));
```

**Patch Explanation**: 
We have added support for batched isend/irecv operations in Megatron training to ensure that pipeline parallel (PP) communication does not leave any rank stuck in a pending communication state.

In the original implementation, each communication round consists of a single send and a single receive, but these operations may occur across different groupEnd boundaries. Without batched P2P communication, this behavior is fatal for the SM-free VCCL backend: if a send in comm1 is issued asynchronously by the CPU before its corresponding recv is posted, the receive will never be matched or completed, leading to a deadlock.

By batching isend/irecv, we guarantee that sends and receives are aligned across communication groups, eliminating the risk of unmatched operations and ensuring safe and efficient pipeline parallel communication under VCCL.

**Megatron Code Setup**:
```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.13.0
# Apply the patch after downloading, this patch is located in `VCCL/asset/Megatron-change.patch`
git apply ./VCCL/asset/Megatron-change.patch
```

### Testing Configuration

#### Training Arguments
Add `--batch-p2p-communication` option in the model configuration file to enable P2P batch functionality:

```bash
TRAINING_ARGS="
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers-per-virtual-pipeline-stage 5 \
    --use-distributed-optimizer \
    # Optional parameter for VCCL zerocopy dense model scenarios, if enable account-for-loss-in-pipeline-split, need to NLAYES-1.
    # --account-for-loss-in-pipeline-split \
    --batch-p2p-communication \
    --recompute-granularity selective \
"
```

#### Output Arguments
```bash
OUTPUT_ARGS="
    --log-interval 1 \
    --eval-iters 4 \
    --eval-interval 10000 \
    --save-interval $SAVE_INTERVAL \
    --log-throughput \
    --timing-log-level 0 \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --seed 1024 \
"
```

#### Parallelism Configuration
Set training parameters appropriately to ensure PP communication groups span across machines:
- For server counts that are multiples of 4, set TP=2 and PP=4
- For PP=6, recommend: 6, 12, 18, 24 machines
- For PP=4, recommend: 4, 8, 12, 16 machines

#### Environment Variables for SM Free Zerocopy
Enable the following environment variables in MPI scripts for improved training TFLOPS:

```bash
-x NCCL_PASS_SM=1 \
-x NCCL_PXN_DISABLE=1 \
# NCCL_PSM_FORCE_ZEROCOPY requires NCCL_PASS_SM to be enabled
# If only using kernel-free mode, NCCL_PSM_FORCE_ZEROCOPY is not required
-x NCCL_PSM_FORCE_ZEROCOPY=1 \
-x NCCL_ENABLE_FAULT_TOLERANCE=0 \
```

### Performance Benefits
- **True Communication-Computation Overlap**: Communication operations don't compete with computation for SM resources
- **Higher Throughput**: Zerocopy mode provides higher bandwidth for P2P communication
- **Improved Training Efficiency**: Better GPU utilization leads to faster training convergence

### co-design
Registered buffers were introduced for two reasons. First, to achieve zero-copy, optimize latency, and save resources. Second, using zero-copy avoids multiple send/recv operations during the send/recv phase.

Net Transport
Origin
![image.png](https://liuda-1370225914.cos.ap-beijing.myqcloud.com/obsidian/picgo/20250917195406670.png)
SM-Free
![image.png](https://liuda-1370225914.cos.ap-beijing.myqcloud.com/obsidian/picgo/20250917195427130.png)
SM-Free with Register buffer(Zerocopy)
![image.png](https://liuda-1370225914.cos.ap-beijing.myqcloud.com/obsidian/picgo/20250917195444004.png)

Net Transport with PXN
Origin
![image.png](https://liuda-1370225914.cos.ap-beijing.myqcloud.com/obsidian/picgo/20250917195501739.png)
SM-Free with Register buff(Zerocopy)
![image.png](https://liuda-1370225914.cos.ap-beijing.myqcloud.com/obsidian/picgo/20250917195518936.png)

P2PTransport
SM-Free
![image.png](https://liuda-1370225914.cos.ap-beijing.myqcloud.com/obsidian/picgo/20250917195539744.png)

SM-Free with register buff(Zerocopy)
![image.png](https://liuda-1370225914.cos.ap-beijing.myqcloud.com/obsidian/picgo/20250917195556993.png)
