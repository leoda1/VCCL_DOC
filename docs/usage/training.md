# 端到端训练

本页面详细介绍如何在实际的深度学习训练场景中部署和使用 VCCL，包括与主流训练框架的集成、配置优化和生产环境的最佳实践。

## 训练框架集成

### Megatron-LM 集成

#### 步骤 1: 准备 VCCL 环境
```bash
# 确保 VCCL 已正确编译和安装
export LD_LIBRARY_PATH=/workspace/infrawaves/share/liuda/vc226/vccl_2.26.6-1/build/lib:$LD_LIBRARY_PATH

# 验证 VCCL 版本
strings $LD_LIBRARY_PATH/libnccl.so | grep VCCL
```

#### 步骤 2: 获取并配置 Megatron-LM
```bash
# 克隆 Megatron-LM 仓库
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# 切换到指定版本
git checkout core_v0.13.0

# 应用 VCCL 优化补丁
# 下载补丁文件后执行
git apply change.patch
```

#### 步骤 3: 配置训练参数
```bash
# 设置基本训练参数
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=512
TP_SIZE=2              # Tensor 并行度
PP_SIZE=4              # Pipeline 并行度

# VCCL 优化配置
TRAINING_ARGS=\"
    --micro-batch-size $MICRO_BATCH_SIZE \\
    --global-batch-size $GLOBAL_BATCH_SIZE \\
    --tensor-model-parallel-size $TP_SIZE \\
    --pipeline-model-parallel-size $PP_SIZE \\
    --num-layers-per-virtual-pipeline-stage 5 \\
    --use-distributed-optimizer \\
    --batch-p2p-communication \\
    --recompute-granularity selective \\
\"
```

#### 步骤 4: MPI 启动脚本配置
```bash
#!/bin/bash
# train_gpt_vccl.sh

# VCCL 环境配置
export LD_LIBRARY_PATH=/workspace/infrawaves/share/liuda/vc226/vccl_2.26.6-1/build/lib:$LD_LIBRARY_PATH

# VCCL 特性启用
export NCCL_PXN_DISABLE=1      # 禁用 PXN（根据需要）
export NCCL_PASS_SM=1          # 启用 SM Free 模式
export NCCL_CUMEM_ENABLE=1     # 启用 CUDA 内存池
export NCCL_ENABLE_FAULT_TOLERANCE=1  # 启用容错

# 网络配置
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1

# 调试和监控
export NCCL_DEBUG=WARN
export NCCL_TELEMETRY_ENABLE=1
export NCCL_TELEMETRY_LOG_PATH=/tmp/training_telemetry

# 训练启动
mpirun -np 128 \\
    --hostfile hostfile \\
    --allow-run-as-root \\
    --map-by ppr:8:node \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_PXN_DISABLE=1 \\
    -x NCCL_PASS_SM=1 \\
    -x NCCL_CUMEM_ENABLE=1 \\
    -x NCCL_ENABLE_FAULT_TOLERANCE=1 \\
    -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1 \\
    python pretrain_gpt.py $TRAINING_ARGS
```

### PyTorch 分布式训练

#### 基本配置
```python
# train_distributed.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def setup_vccl_environment():
    \"\"\"配置 VCCL 环境变量\"\"\"
    os.environ['NCCL_PASS_SM'] = '1'
    os.environ['NCCL_CUMEM_ENABLE'] = '1'
    os.environ['NCCL_ENABLE_FAULT_TOLERANCE'] = '1'
    os.environ['NCCL_IB_GID_INDEX'] = '3'
    
    # 可选：启用遥测
    os.environ['NCCL_TELEMETRY_ENABLE'] = '1'
    os.environ['NCCL_TELEMETRY_LOG_PATH'] = '/tmp/pytorch_telemetry'

def init_distributed(rank, world_size):
    \"\"\"初始化分布式环境\"\"\"
    setup_vccl_environment()
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前设备
    torch.cuda.set_device(rank % torch.cuda.device_count())

def train_model(rank, world_size, model, dataloader):
    \"\"\"分布式训练主函数\"\"\"
    init_distributed(rank, world_size)
    
    # 包装模型为 DDP
    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(rank % torch.cuda.device_count()),
        device_ids=[rank % torch.cuda.device_count()]
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # VCCL 会自动处理梯度同步
            optimizer.step()

if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    mp.spawn(train_model, args=(world_size, model, dataloader), nprocs=world_size)
```

#### 启动脚本
```bash
#!/bin/bash
# pytorch_distributed_train.sh

export LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH

# 多节点训练配置
export MASTER_ADDR=node01
export MASTER_PORT=29500
export WORLD_SIZE=32
export RANK=0

# VCCL 配置
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_ENABLE_FAULT_TOLERANCE=1

# 启动训练
python -m torch.distributed.launch \\
    --nproc_per_node=8 \\
    --nnodes=4 \\
    --node_rank=$SLURM_NODEID \\
    --master_addr=$MASTER_ADDR \\
    --master_port=$MASTER_PORT \\
    train_distributed.py \\
    --batch-size 32 \\
    --epochs 100
```

### TensorFlow 集成

#### 基本配置
```python
# tensorflow_vccl_train.py
import tensorflow as tf
import os

def setup_vccl_for_tensorflow():
    \"\"\"为 TensorFlow 配置 VCCL\"\"\"
    os.environ['NCCL_PASS_SM'] = '1'
    os.environ['NCCL_CUMEM_ENABLE'] = '1'
    os.environ['NCCL_ENABLE_FAULT_TOLERANCE'] = '1'
    
    # TensorFlow GPU 配置
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

def create_distributed_strategy():
    \"\"\"创建分布式策略\"\"\"
    setup_vccl_for_tensorflow()
    
    # 使用 MultiWorkerMirroredStrategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy(
        communication_options=tf.distribute.experimental.CommunicationOptions(
            implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
        )
    )
    return strategy

# 分布式训练
strategy = create_distributed_strategy()

with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# 训练
model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset
)
```

## 大规模训练配置

### 多服务器 PP 通信配置
```bash
# 确保 PP 通信组跨机器分布
# 示例：4 台服务器，每台 8 个 GPU

# 服务器配置
SERVERS=4
GPUS_PER_SERVER=8
TOTAL_GPUS=$((SERVERS * GPUS_PER_SERVER))

# 并行配置（确保跨机器通信）
TP_SIZE=2              # Tensor 并行
PP_SIZE=4              # Pipeline 并行（跨服务器）
DP_SIZE=$((TOTAL_GPUS / (TP_SIZE * PP_SIZE)))  # 数据并行

echo \"Total GPUs: $TOTAL_GPUS\"
echo \"TP=$TP_SIZE, PP=$PP_SIZE, DP=$DP_SIZE\"
```

### Slurm 集群配置
```bash
#!/bin/bash
#SBATCH --job-name=vccl_training
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# VCCL 环境配置
export LD_LIBRARY_PATH=/opt/vccl/lib:$LD_LIBRARY_PATH
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1

# Slurm 环境变量
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# 启动训练
srun python -u train_large_model.py \\
    --distributed \\
    --backend nccl \\
    --model-size 175B \\
    --batch-size 4 \\
    --sequence-length 2048
```

### Kubernetes 部署
```yaml
# vccl-training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: vccl-training
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: training
        image: nvcr.io/nvidia/pytorch:23.08-py3
        resources:
          limits:
            nvidia.com/gpu: 8
        env:
        - name: LD_LIBRARY_PATH
          value: \"/opt/vccl/lib:$(LD_LIBRARY_PATH)\"
        - name: NCCL_PASS_SM
          value: \"1\"
        - name: NCCL_CUMEM_ENABLE
          value: \"1\"
        - name: NCCL_ENABLE_FAULT_TOLERANCE
          value: \"1\"
        volumeMounts:
        - name: vccl-lib
          mountPath: /opt/vccl
        - name: shared-storage
          mountPath: /workspace
        command:
        - python
        - /workspace/train_distributed.py
      volumes:
      - name: vccl-lib
        hostPath:
          path: /opt/vccl
      - name: shared-storage
        nfs:
          server: nfs-server
          path: /shared/workspace
```

## 性能优化配置

### 内存优化
```bash
# GPU 内存优化
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_POOL_SIZE=8GB
export NCCL_CUMEM_PREALLOC=1

# 系统内存优化
export NCCL_BUFFSIZE=8388608      # 8MB 通信缓冲区
export NCCL_P2P_LEVEL=PIX         # P2P 传输级别
```

### 网络优化
```bash
# InfiniBand 优化
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
export NCCL_IB_AR_THRESHOLD=8192

# 拓扑优化
export NCCL_TOPO_VERBOSE=1
export NCCL_NET_GDR_LEVEL=PHB
```

### 计算通信重叠优化
```bash
# 启用 SM Free 和重叠优化
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1

# 梯度压缩（可选）
export NCCL_COMPRESSION=1
export NCCL_COMPRESSION_THRESHOLD=65536
```

## 监控和调试

### 性能监控脚本
```python
#!/usr/bin/env python3
# monitor_training.py

import time
import json
import psutil
import nvidia_ml_py3 as nvml
from pathlib import Path

def monitor_training_performance():
    nvml.nvmlInit()
    device_count = nvml.nvmlDeviceGetCount()
    
    while True:
        # GPU 使用率监控
        gpu_stats = []
        for i in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            util = nvml.nvmlDeviceGetUtilizationRates(handle)
            memory = nvml.nvmlDeviceGetMemoryInfo(handle)
            
            gpu_stats.append({
                'gpu_id': i,
                'utilization': util.gpu,
                'memory_used': memory.used // 1024**2,  # MB
                'memory_total': memory.total // 1024**2
            })
        
        # CPU 和内存监控
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # 检查 VCCL 遥测数据
        telemetry_files = list(Path('/tmp/training_telemetry').glob('rank_*.json'))
        if telemetry_files:
            with open(telemetry_files[0]) as f:
                latest_telemetry = f.readlines()[-1]
                telemetry_data = json.loads(latest_telemetry)
        else:
            telemetry_data = {}
        
        # 输出监控信息
        print(f\"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\")
        print(f\"CPU: {cpu_percent}%, Memory: {memory.percent}%\")
        for gpu in gpu_stats:
            print(f\"GPU {gpu['gpu_id']}: {gpu['utilization']}%, \"
                  f\"Memory: {gpu['memory_used']}/{gpu['memory_total']} MB\")
        
        if telemetry_data:
            print(f\"VCCL Bandwidth: {telemetry_data.get('bandwidth_gbps', 'N/A')} GB/s\")
        
        print(\"-\" * 50)
        time.sleep(10)

if __name__ == '__main__':
    monitor_training_performance()
```

### 故障自动恢复
```python
# fault_recovery.py
import subprocess
import time
import signal
import sys

class TrainingFaultRecovery:
    def __init__(self, training_script, checkpoint_dir):
        self.training_script = training_script
        self.checkpoint_dir = checkpoint_dir
        self.max_retries = 3
        self.current_retries = 0
        
    def start_training(self):
        while self.current_retries < self.max_retries:
            try:
                # 启动训练进程
                process = subprocess.Popen([
                    'python', self.training_script,
                    '--resume-from-checkpoint', self.checkpoint_dir
                ])
                
                # 等待进程完成
                return_code = process.wait()
                
                if return_code == 0:
                    print(\"Training completed successfully\")
                    break
                else:
                    print(f\"Training failed with return code {return_code}\")
                    self.current_retries += 1
                    self.handle_failure()
                    
            except Exception as e:
                print(f\"Exception during training: {e}\")
                self.current_retries += 1
                self.handle_failure()
    
    def handle_failure(self):
        if self.current_retries < self.max_retries:
            print(f\"Retrying training... (attempt {self.current_retries + 1})\")
            time.sleep(60)  # 等待 1 分钟后重试
        else:
            print(\"Max retries exceeded. Training failed.\")
            sys.exit(1)

# 使用示例
recovery = TrainingFaultRecovery('train_gpt.py', '/shared/checkpoints')
recovery.start_training()
```

## 生产环境最佳实践

### 检查点管理
```python
# checkpoint_manager.py
import torch
import os
from pathlib import Path

class VCCLCheckpointManager:
    def __init__(self, checkpoint_dir, keep_last_n=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.keep_last_n = keep_last_n
    
    def save_checkpoint(self, model, optimizer, epoch, step, loss):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'vccl_config': self.get_vccl_config()
        }
        
        checkpoint_path = self.checkpoint_dir / f\"checkpoint_epoch_{epoch}_step_{step}.pt\"
        torch.save(checkpoint, checkpoint_path)
        
        # 清理旧检查点
        self.cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def get_vccl_config(self):
        \"\"\"保存当前 VCCL 配置\"\"\"
        return {
            'NCCL_PASS_SM': os.environ.get('NCCL_PASS_SM'),
            'NCCL_CUMEM_ENABLE': os.environ.get('NCCL_CUMEM_ENABLE'),
            'NCCL_ENABLE_FAULT_TOLERANCE': os.environ.get('NCCL_ENABLE_FAULT_TOLERANCE'),
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH')
        }
    
    def cleanup_old_checkpoints(self):
        \"\"\"清理旧的检查点文件\"\"\"
        checkpoints = sorted(self.checkpoint_dir.glob(\"checkpoint_*.pt\"))
        if len(checkpoints) > self.keep_last_n:
            for old_checkpoint in checkpoints[:-self.keep_last_n]:
                old_checkpoint.unlink()
```

### 配置管理
```bash
# config_management.sh
#!/bin/bash

# VCCL 配置模板
create_vccl_config() {
    local environment=$1  # dev, staging, prod
    
    case $environment in
        \"dev\")
            export NCCL_DEBUG=INFO
            export NCCL_TELEMETRY_ENABLE=1
            export NCCL_PASS_SM=0  # 调试时可能需要禁用
            ;;
        \"staging\")
            export NCCL_DEBUG=WARN
            export NCCL_TELEMETRY_ENABLE=1
            export NCCL_PASS_SM=1
            ;;
        \"prod\")
            export NCCL_DEBUG=ERROR
            export NCCL_TELEMETRY_ENABLE=0  # 生产环境可选择关闭
            export NCCL_PASS_SM=1
            ;;
    esac
    
    # 通用配置
    export NCCL_CUMEM_ENABLE=1
    export NCCL_ENABLE_FAULT_TOLERANCE=1
    export NCCL_IB_GID_INDEX=3
}

# 验证配置
validate_vccl_config() {
    echo \"Validating VCCL configuration...\"
    
    # 检查库文件
    if ! ldd $(which python) | grep -q libnccl; then
        echo \"Warning: NCCL library not found\"
    fi
    
    # 检查环境变量
    echo \"LD_LIBRARY_PATH: $LD_LIBRARY_PATH\"
    echo \"NCCL_PASS_SM: $NCCL_PASS_SM\"
    echo \"NCCL_CUMEM_ENABLE: $NCCL_CUMEM_ENABLE\"
    
    # 检查 GPU 和网络
    nvidia-smi --query-gpu=name --format=csv,noheader
    ibstat | grep -E \"(State|Rate)\"
}
```

### 自动化部署脚本
```bash
#!/bin/bash
# deploy_vccl_training.sh

set -e

# 配置参数
CLUSTER_CONFIG=\"$1\"
MODEL_CONFIG=\"$2\"
ENVIRONMENT=\"${3:-staging}\"

# 验证参数
if [[ -z \"$CLUSTER_CONFIG\" || -z \"$MODEL_CONFIG\" ]]; then
    echo \"Usage: $0 <cluster_config> <model_config> [environment]\"
    exit 1
fi

# 加载配置
source \"$CLUSTER_CONFIG\"
source \"$MODEL_CONFIG\"
source \"config_management.sh\"

# 设置环境
create_vccl_config \"$ENVIRONMENT\"
validate_vccl_config

# 准备训练环境
echo \"Preparing training environment...\"
mkdir -p \"$CHECKPOINT_DIR\"
mkdir -p \"$LOG_DIR\"

# 生成 hostfile
generate_hostfile() {
    for ((i=0; i<$NODE_COUNT; i++)); do
        echo \"${NODE_PREFIX}$(printf %02d $i) slots=$GPUS_PER_NODE\"
    done > hostfile
}

generate_hostfile

# 启动训练
echo \"Starting VCCL training...\"
mpirun -np $((NODE_COUNT * GPUS_PER_NODE)) \\
    --hostfile hostfile \\
    --allow-run-as-root \\
    --map-by ppr:$GPUS_PER_NODE:node \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_PASS_SM \\
    -x NCCL_CUMEM_ENABLE \\
    -x NCCL_ENABLE_FAULT_TOLERANCE \\
    python train_model.py \\
        --config \"$MODEL_CONFIG\" \\
        --checkpoint-dir \"$CHECKPOINT_DIR\" \\
        --log-dir \"$LOG_DIR\"

echo \"Training deployment completed.\"
```

---

!!! tip \"性能调优建议\"
    在生产环境中，建议先在小规模集群上验证配置，然后逐步扩展到大规模集群。

!!! warning \"检查点管理\"
    长时间训练务必启用定期检查点保存，并配置故障自动恢复机制。

!!! info \"监控重要性\"
    持续监控 GPU 利用率、网络带宽和 VCCL 遥测数据，及时发现和解决性能瓶颈。
