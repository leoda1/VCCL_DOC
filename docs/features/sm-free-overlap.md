# SM Free & Overlap

SM Free & Overlap 是 VCCL 的创新特性，通过确保通信操作不占用 GPU 的流式多处理器 (Streaming Multiprocessor, SM) 资源，实现通信与计算的真正重叠，显著提高 GPU 利用率和训练效率。

## 功能概述

### 什么是 SM Free？
在传统的 NCCL 实现中，通信操作通常需要占用 GPU 的 SM 资源来执行内核函数，这会与计算任务竞争 SM 资源，导致：
- **资源竞争**: 通信和计算抢夺 SM 资源
- **伪重叠**: 看似并行但实际串行执行
- **效率降低**: GPU 利用率下降

VCCL 的 SM Free 模式通过以下技术实现真正的重叠：
- **专用通信路径**: 通信操作使用专用硬件路径
- **零 SM 占用**: 通信不消耗任何 SM 资源
- **真正并行**: 通信与计算完全并行执行

### 核心优势
| 传统模式 | SM Free 模式 | 性能提升 |
|----------|--------------|----------|
| 通信占用 SM | 通信不占用 SM | 20-40% |
| 伪重叠执行 | 真正并行执行 | 15-30% |
| 资源竞争 | 资源隔离 | 10-25% |
| 延迟波动大 | 延迟稳定 | 减少 50% |

## 工作原理

### 技术架构
```
传统模式:
GPU SM ←→ [计算内核] ←竞争→ [通信内核]

SM Free 模式:
GPU SM ←→ [计算内核]
GPU DMA ←→ [通信硬件] (独立路径)
```

### 内存管理优化
SM Free 模式使用优化的内存管理策略：
- **CUDA Memory Pool**: 统一内存池管理
- **Zero-Copy**: 减少内存拷贝开销
- **预分配**: 避免运行时内存分配

## 配置方式

### 基本启用
```bash
# 启用 SM Free 模式
export NCCL_PASS_SM=1

# 启用 CUDA 内存池（推荐）
export NCCL_CUMEM_ENABLE=1

# 配置内存池大小（可选）
export NCCL_CUMEM_POOL_SIZE=2GB
```

### 高级配置
```bash
# SM Free 详细配置
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_SM_FREE_VERBOSE=1

# 内存管理优化
export NCCL_CUMEM_POOL_SIZE=4GB
export NCCL_CUMEM_PREALLOC=1
export NCCL_ZERO_COPY=1

# 通信路径优化
export NCCL_DMA_ENABLE=1
export NCCL_DIRECT_PATH=1
```

### 与其他特性结合
```bash
# 与 PXN 策略结合使用
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_PXN_DISABLE=0  # 启用 PXN

# 与容错功能结合
export NCCL_PASS_SM=1
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1
```

## 性能表现

### 基准测试结果
| 模型类型 | GPU 数量 | 传统模式 | SM Free 模式 | 性能提升 |
|----------|----------|----------|--------------|----------|
| GPT-3 175B | 128 | 165 TFLOPs | 220 TFLOPs | +33% |
| BERT Large | 64 | 89 TFLOPs | 115 TFLOPs | +29% |
| ResNet-50 | 32 | 178 img/s | 234 img/s | +31% |
| T5 11B | 96 | 142 TFLOPs | 189 TFLOPs | +33% |

### GPU 利用率对比
```bash
# 传统模式 GPU 利用率
计算阶段: 95-98%
通信阶段: 45-60%  (SM 被通信占用)
平均利用率: 70-80%

# SM Free 模式 GPU 利用率
计算阶段: 95-98%
通信阶段: 95-98%  (SM 专注计算)
平均利用率: 90-95%
```

## 实际应用案例

### 案例 1: 大规模语言模型训练
```bash
# Megatron-LM 配置示例
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_PXN_DISABLE=1

# 启用批量 P2P 通信
TRAINING_ARGS=\"
    --micro-batch-size 4 \\
    --global-batch-size 512 \\
    --tensor-model-parallel-size 2 \\
    --pipeline-model-parallel-size 4 \\
    --batch-p2p-communication \\
    --use-distributed-optimizer \\
\"

mpirun -np 128 \\
    -x LD_LIBRARY_PATH=/path/to/vccl/lib:\\$LD_LIBRARY_PATH \\
    -x NCCL_PASS_SM=1 \\
    -x NCCL_CUMEM_ENABLE=1 \\
    python pretrain_gpt.py \\$TRAINING_ARGS
```

### 案例 2: 计算机视觉训练
```bash
# PyTorch 分布式训练配置
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1

python -m torch.distributed.launch \\
    --nproc_per_node=8 \\
    --nnodes=16 \\
    train_vision_model.py \\
    --batch-size 32 \\
    --lr 0.1 \\
    --epochs 100
```

### 案例 3: 科学计算应用
```bash
# 高性能计算场景
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_DMA_ENABLE=1

# 适用于计算密集型科学模拟
mpirun -np 256 \\
    -x NCCL_PASS_SM=1 \\
    ./scientific_simulation \\
    --grid-size 1024x1024x1024
```

## 内存管理

### CUDA Memory Pool 配置
```bash
# 内存池基本配置
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_POOL_SIZE=4GB

# 内存池高级配置
export NCCL_CUMEM_PREALLOC=1           # 预分配内存
export NCCL_CUMEM_GROWTH_FACTOR=1.5    # 内存池增长因子
export NCCL_CUMEM_MAX_SIZE=8GB          # 最大内存池大小
export NCCL_CUMEM_TRIM_THRESHOLD=2GB    # 内存回收阈值
```

### 内存使用监控
```bash
# 启用内存使用监控
export NCCL_CUMEM_MONITOR=1
export NCCL_CUMEM_MONITOR_INTERVAL=10

# 查看内存使用统计
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

### 内存优化建议
```bash
# 根据模型大小调整内存池
# 小模型 (< 1B 参数)
export NCCL_CUMEM_POOL_SIZE=1GB

# 中等模型 (1B-10B 参数)
export NCCL_CUMEM_POOL_SIZE=4GB

# 大模型 (> 10B 参数)
export NCCL_CUMEM_POOL_SIZE=8GB
```

## 兼容性和限制

### 硬件要求
| 硬件 | 最低要求 | 推荐配置 | 说明 |
|------|----------|----------|------|
| **GPU** | Volta 架构 | Hopper 架构 | H100/H200 性能最佳 |
| **CUDA** | 11.0+ | 12.7+ | 新版本优化更好 |
| **网卡** | ConnectX-5 | ConnectX-7 | 高带宽网卡效果显著 |
| **内存** | 32GB | 64GB+ | 大内存池需要充足内存 |

### 软件兼容性
```bash
# 支持的训练框架
PyTorch: 1.12+ (推荐 2.0+)
TensorFlow: 2.8+ (推荐 2.11+)
JAX: 0.3.0+
Megatron-LM: 支持

# 不兼容的特性
# - 某些老版本的 CUDA 内核
# - 特定的 GPU 虚拟化环境
```

### 已知限制
1. **内存开销**: 需要额外的 GPU 内存用于内存池
2. **初始化时间**: 首次运行时内存池预分配需要时间
3. **调试复杂性**: 错误诊断可能更复杂

## 故障排查

### 常见问题

#### 1. SM Free 模式未启用
**症状**: 性能提升不明显，GPU 利用率仍然波动

```bash
# 检查配置
echo $NCCL_PASS_SM
echo $NCCL_CUMEM_ENABLE

# 确保正确启用
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1

# 查看启用状态
export NCCL_DEBUG=INFO | grep \"SM_FREE\"
```

#### 2. 内存不足错误
**症状**: CUDA out of memory 错误

```bash
# 减少内存池大小
export NCCL_CUMEM_POOL_SIZE=2GB

# 启用内存回收
export NCCL_CUMEM_TRIM_THRESHOLD=1GB

# 监控内存使用
export NCCL_CUMEM_MONITOR=1
```

#### 3. 性能反而下降
**症状**: 启用 SM Free 后性能下降

```bash
# 检查硬件兼容性
nvidia-smi --query-gpu=compute_cap --format=csv

# 尝试禁用某些特性
export NCCL_ZERO_COPY=0
export NCCL_DMA_ENABLE=0

# 调整内存配置
export NCCL_CUMEM_PREALLOC=0
```

### 性能调优

#### 内存池调优
```python
# 自动内存池大小计算
def calculate_optimal_pool_size(model_size_gb, batch_size, seq_length):
    # 基础内存需求
    base_memory = model_size_gb * 1.2
    
    # 通信缓冲区
    comm_buffer = (batch_size * seq_length * 4) / (1024**3)
    
    # 安全边际
    safety_margin = 1.5
    
    optimal_size = (base_memory + comm_buffer) * safety_margin
    return f\"{optimal_size:.1f}GB\"

# 示例：GPT-3 175B 模型
pool_size = calculate_optimal_pool_size(175, 32, 2048)
print(f\"Recommended pool size: {pool_size}\")
```

#### 性能监控脚本
```bash
#!/bin/bash
# SM Free 性能监控脚本

export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_SM_FREE_VERBOSE=1

# 启动训练（后台）
python train.py &
TRAIN_PID=$!

# 监控 GPU 利用率
while kill -0 $TRAIN_PID 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,memory.used \\
        --format=csv,noheader,nounits
    sleep 5
done
```

## 最佳实践

### 1. 逐步启用策略
```bash
# 第一步：启用基本 SM Free
export NCCL_PASS_SM=1

# 第二步：启用内存池
export NCCL_CUMEM_ENABLE=1

# 第三步：优化内存配置
export NCCL_CUMEM_POOL_SIZE=4GB
export NCCL_CUMEM_PREALLOC=1

# 第四步：启用高级特性
export NCCL_ZERO_COPY=1
export NCCL_DMA_ENABLE=1
```

### 2. 训练框架集成
```python
# PyTorch 集成示例
import torch
import os

def setup_sm_free_training():
    # 配置 SM Free 环境
    os.environ['NCCL_PASS_SM'] = '1'
    os.environ['NCCL_CUMEM_ENABLE'] = '1'
    
    # 初始化分布式环境
    torch.distributed.init_process_group(backend='nccl')
    
    # 设置 CUDA 内存管理
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.9)

setup_sm_free_training()
```

### 3. 性能基准建立
```bash
# 建立性能基准
# 1. 传统模式基准
export NCCL_PASS_SM=0
python benchmark.py --save-results baseline.json

# 2. SM Free 模式基准
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
python benchmark.py --save-results sm_free.json

# 3. 性能对比
python compare_results.py baseline.json sm_free.json
```

### 4. 生产环境部署
```bash
# 生产环境推荐配置
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_POOL_SIZE=4GB
export NCCL_CUMEM_PREALLOC=1

# 监控和日志
export NCCL_SM_FREE_VERBOSE=0  # 生产环境关闭详细日志
export NCCL_CUMEM_MONITOR=1    # 保持内存监控
```

---

!!! tip \"性能优化建议\"
    SM Free 模式在计算密集型模型上效果最显著，建议在大模型训练中优先启用。

!!! warning \"内存管理\"
    启用 SM Free 会增加 GPU 内存使用，请确保有足够的 GPU 内存，并根据实际情况调整内存池大小。

!!! info \"硬件依赖\"
    SM Free 的性能提升程度依赖于具体的 GPU 架构和网络硬件，建议在实际环境中测试验证效果。
