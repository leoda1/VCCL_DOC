# VCCL v2 Installation and Setup Guide

# Overview

VCCL v2 introduces VCCL Manager, a runtime component that enables Megatron\-LM to invoke VCCL AlltoAllv directly through the nccl4py interface\.

To simplify integration, we provide a Megatron patch implementing all core VCCL Manager functionalities\. Applying this patch allows Megatron\-LM to seamlessly leverage VCCL AlltoAllv for accelerated MoE training\.

In addition to the patch, users must install and configure:

- VCCL v2

- nccl4py

- Transformer Engine \(TE\) 2\.8\.0

> **Note:** Currently, only Transformer Engine 2\.8\.0 is officially supported\.
> 
> 

---

# Prerequisites

Before configuring VCCL Manager, install and build VCCL according to the official documentation:

VCCL Installation Guide:

[https://vccl\-doc\.readthedocs\.io/en/latest/usage/installation/](https://vccl-doc.readthedocs.io/en/latest/usage/installation/)

---

## Megatron\-LM Integration

Clone the supported Megatron\-LM version and apply the VCCL Manager patch\.

```Plain Text
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

git checkout 55103f9a560b3fdefc235df53da80740e37df85e

git apply path_to_VCCL/asset/megatron_vccl_v2.patch

cd vccl_alloc
nvcc -Xcompiler -fPIC -shared -std=c++17 -o libvccl_arena_allocator.so vccl_arena_allocator.cu
```

---

## nccl4py Setup

Ensure VCCL has been successfully compiled before building nccl4py\.

```Plain Text
cd path_to_VCCL/nccl4py

export CUDA_HOME=/usr/local/cuda

python setup.py build_ext --inplace
```

---

## Transformer Engine Configuration

VCCL Manager currently relies on a customized implementation of `_GroupedLinear` in Transformer Engine\.

Replace the default implementation with the VCCL\-compatible version:

```Plain Text
cd /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/module

cp path_to_Megatron-LM/te/grouped_linear.py grouped_linear.py

rm /usr/local/lib/python3.12/dist-packages/transformer_engine/pytorch/__pycache__/*
```

If Transformer Engine is installed in a different location, adjust the path accordingly\.

---

# Training Configuration

To enable AlltoAllv acceleration for MoE training, add the following Megatron options\.

### Enable MoE Dual Pipeline

```Plain Text
--overlap-moe-expert-parallel-comm
```

### Enable VCCL AlltoAllv Dispatcher

```Plain Text
--moe-token-dispatcher-type alltoallv
```

### Enable Grouped GEMM

```Plain Text
--moe-grouped-gemm
```

> **Note:** The current activation and gradient interception mechanism depends on Transformer Engine's `_GroupedLinear` implementation\. Support for additional execution paths will be added in future releases\.
> 
> 

---

# Environment Variables

Configure the following environment variables before launching training\.

### Load VCCL v2 Runtime

```Plain Text
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path_to_VCCL/build/lib
```

### Enable nccl4py

```Plain Text
export PYTHONPATH="${PYTHONPATH}:path_to_VCCL/nccl4py"
```

### Enable MoE Dual Pipeline Scheduling

```Plain Text
export CUDA_DEVICE_MAX_CONNECTIONS=32
```

> **Important:** `CUDA_DEVICE_MAX_CONNECTIONS=32` is required for Megatron\-LM to correctly schedule the MoE dual pipeline execution\.
> 
> 

---

# Example Launch Scripts

mpi\.sh

```Bash
#! /bin/bash

NET_DEVICE="bond0"
MLP_GPU=8
MLP_MPI_HOSTFILE=$2
MLP_WORKER_0_PORT=29500
MLP_WORKER_NUM=$3
source $1

mkdir -p logs/${EXP_NAME}
mpirun -np $((MLP_WORKER_NUM * MLP_GPU)) \
        --hostfile ${MLP_MPI_HOSTFILE} \
        --allow-run-as-root   \
        --output-filename logs/${TIMESTAMP} \
        --mca oob_tcp_if_include ${NET_DEVICE} \
        -x NCCL_IB_TC=106 \
        -x NCCL_IB_GID_INDEX=3 \
        -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/inspire/hdd/VCCL_nccl4py/build/lib \
        -x PYTHONPATH="${PYTHONPATH}:/inspire/hdd/VCCL_nccl4py/nccl4py" \
        -x NCCL_DEBUG=WARN \
        -x NCCL_IB_HCA==mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1 \
        -x PATH \
        -x MASTER_ADDR=$(cat $MLP_MPI_HOSTFILE | head -n 1 | sed -s 's/slots=8//g') \
        -x MASTER_PORT=${MLP_WORKER_0_PORT} \
        -x GLOO_SOCKET_IFNAME=${NET_DEVICE} \
        -x NCCL_SOCKET_IFNAME=${NET_DEVICE} \
        -x UCX_NET_DEVICES=${NET_DEVICE} \
        -x CUDA_DEVICE_MAX_CONNECTIONS=32 \
        -x NCCL_PXN_DISABLE=0 \
        -x NCCL_CUMEM_ENABLE=1 \
        -x NCCL_NVLS_ENABLE=0 \
        script/nsys.sh python ${script_path} ${gpt_options}
```

mixtral\_8x7b\.sh

```Bash
#!/bin/bash

EXP_NAME="moe-group"
script_path="pretrain_gpt.py"

MODEL_ARGS="
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
"
MOE_ARGS="
    --num-experts 8
    --moe-router-topk 2
    --moe-router-force-load-balancing
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoallv
    --moe-grouped-gemm
    --overlap-moe-expert-parallel-comm
"

#    --delay-wgrad-compute
DATA_ARGS="
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ./moe/tokenizer.model \
    --data-path ./moe/wudao_mistralbpe_content_document \
    --split 949,50,1 \
"

TRAINING_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --lr 1e-4 \
    --train-iters 20000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 0.1 \
    --lr-warmup-iters 0 \
    --clip-grad 1.0 \
    --bf16 \
"
MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1\
    --context-parallel-size 1 \
    --use-distributed-optimizer \
    --sequence-parallel \
    --num-layers-per-virtual-pipeline-stage 4 \
    --use-flash-attn \
    --disable-bias-linear \
"

LOGGING_ARGS="
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 10000 \
    --save-interval 10000 \
    --no-load-optim \
    --no-load-rng \
    --log-throughput \
    --timing-log-level 0 \
"
gpt_options="
    ${MODEL_ARGS} \
    ${MODEL_PARALLEL_ARGS} \
    ${TRAIN_ARGS} \
    ${MOE_ARGS} \
    ${DATA_ARGS} \
    ${TRAINING_ARGS} \
    ${LOGGING_ARGS} \
"
```



