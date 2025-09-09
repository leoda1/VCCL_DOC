# PCIe 异构与智能 RNIC 选择 (SRS)

智能 RNIC 选择 (Smart RNIC Selection, SRS) 是 VCCL 针对现代异构集群设计的关键特性，能够在复杂的 PCIe 拓扑环境中自动选择最优的网络接口。

## 功能概述

### 什么是 PCIe 异构？
现代 AI 集群通常包含：
- **不同代数的 PCIe**: PCIe 3.0、4.0、5.0 混合
- **不同带宽的插槽**: x8、x16 插槽混合  
- **多种网络卡**: 不同型号的 InfiniBand/Ethernet 卡
- **NUMA 拓扑**: 复杂的 CPU-GPU-网卡亲和性

### SRS 工作原理
智能 RNIC 选择通过以下机制实现最优选择：

1. **硬件拓扑扫描**: 自动发现 PCIe 拓扑和设备信息
2. **性能基准测试**: 实时测量各网卡的实际性能
3. **亲和性分析**: 分析 GPU-网卡-CPU 的亲和性关系
4. **动态选择**: 根据当前负载选择最优 RNIC

## 支持的异构场景

### PCIe 代数混合
| PCIe 版本 | 理论带宽 | 实际性能 | SRS 权重 |
|-----------|----------|----------|----------|
| PCIe 5.0 x16 | 63 GB/s | ~50 GB/s | 1.0 |
| PCIe 4.0 x16 | 31.5 GB/s | ~25 GB/s | 0.8 |
| PCIe 3.0 x16 | 15.75 GB/s | ~12 GB/s | 0.6 |
| PCIe 4.0 x8 | 15.75 GB/s | ~10 GB/s | 0.4 |

### 网卡型号混合
```bash
# 支持的网卡类型示例
mlx5_0: ConnectX-5 (100 Gb/s)
mlx5_1: ConnectX-6 (200 Gb/s)  
mlx5_2: ConnectX-7 (400 Gb/s)
mlx5_3: ConnectX-5 (100 Gb/s)
```

## 配置方式

### 默认行为
SRS 功能默认启用，自动进行最优选择：

```bash
# 默认启用，无需额外配置
# VCCL 会自动发现和选择最优 RNIC
```

### 手动指定 RNIC
某些场景下可能需要手动指定：

```bash
# 指定特定的网卡
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1

# 指定网卡端口
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:2

# 禁用某些网卡
export NCCL_IB_DISABLE=mlx5_2,mlx5_3
```

### SRS 高级配置

```bash
# 启用 SRS 详细日志
export NCCL_SRS_VERBOSE=1

# 设置性能测试持续时间（秒）
export NCCL_SRS_BENCHMARK_TIME=5

# 设置选择算法
export NCCL_SRS_ALGORITHM=performance  # 或 bandwidth, latency

# 启用亲和性优化
export NCCL_SRS_AFFINITY=1
```

## 性能优化策略

### 选择算法

#### 1. 性能优先 (performance)
- **目标**: 最大化整体通信性能
- **适用**: 大规模 AllReduce 操作
- **权重**: 带宽 (40%) + 延迟 (30%) + 亲和性 (30%)

#### 2. 带宽优先 (bandwidth)  
- **目标**: 最大化数据传输带宽
- **适用**: 大数据量传输场景
- **权重**: 带宽 (70%) + 亲和性 (30%)

#### 3. 延迟优先 (latency)
- **目标**: 最小化通信延迟
- **适用**: 小消息频繁通信
- **权重**: 延迟 (70%) + 亲和性 (30%)

### 亲和性优化

```bash
# GPU-网卡亲和性映射示例
GPU 0,1 -> mlx5_0 (same PCIe root complex)
GPU 2,3 -> mlx5_1 (same PCIe root complex)  
GPU 4,5 -> mlx5_2 (same PCIe root complex)
GPU 6,7 -> mlx5_3 (same PCIe root complex)
```

## 实际应用案例

### 案例 1: 异构 DGX 集群
```bash
# 场景：新老 DGX 节点混合的集群
export NCCL_SRS_VERBOSE=1
export NCCL_SRS_ALGORITHM=performance
export NCCL_DEBUG=INFO

mpirun -np 64 \
    --hostfile mixed_dgx_hosts \
    -x LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH \
    -x NCCL_SRS_VERBOSE=1 \
    -x NCCL_DEBUG=INFO \
    python distributed_training.py

# SRS 会自动适配不同节点的网卡配置
```

### 案例 2: 云环境部署
```bash
# 场景：不同实例类型的云环境
export NCCL_SRS_ALGORITHM=bandwidth
export NCCL_SRS_AFFINITY=1

# 自动适配不同云实例的网络配置
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=16 \
    train_model.py
```

## 性能基准测试

### 自动基准测试
SRS 在初始化时会自动进行性能测试：

```bash
# 查看基准测试结果
export NCCL_SRS_VERBOSE=1
export NCCL_DEBUG=INFO

# 测试输出示例：
# SRS: mlx5_0 bandwidth=195.2 GB/s latency=1.2us affinity=1.0
# SRS: mlx5_1 bandwidth=187.5 GB/s latency=1.4us affinity=0.8  
# SRS: Selected mlx5_0 for GPU 0 (score=0.94)
```

### 手动基准测试
```bash
# 运行独立的网卡性能测试
export NCCL_SRS_BENCHMARK_ONLY=1
mpirun -np 2 ./nccl_test

# 生成详细的性能报告
export NCCL_SRS_REPORT_FILE=/tmp/srs_report.json
```

## 故障排查

### 常见问题

#### 1. RNIC 检测失败
**症状**: 无法检测到网卡或检测信息不准确

```bash
# 诊断命令
export NCCL_SRS_VERBOSE=1
export NCCL_DEBUG=WARN

# 手动检查网卡
ibstat
lspci | grep -i infiniband
```

#### 2. 性能选择不当
**症状**: SRS 选择的网卡性能不如预期

```bash
# 强制重新基准测试
export NCCL_SRS_FORCE_BENCHMARK=1

# 调整选择算法
export NCCL_SRS_ALGORITHM=bandwidth

# 手动指定网卡
export NCCL_IB_HCA=mlx5_1:1
```

#### 3. 亲和性配置错误
**症状**: GPU-网卡亲和性不合理

```bash
# 查看当前亲和性
nvidia-smi topo -m
lstopo

# 禁用亲和性优化
export NCCL_SRS_AFFINITY=0
```

### 调试工具

```bash
# SRS 状态查看
export NCCL_SRS_STATUS_FILE=/tmp/srs_status.txt

# 网卡性能监控
export NCCL_SRS_MONITOR=1
export NCCL_SRS_MONITOR_INTERVAL=10  # 10秒间隔

# 生成 SRS 报告
export NCCL_SRS_GENERATE_REPORT=1
```

## 最佳实践

### 1. 硬件配置建议
- **统一网卡型号**: 尽量使用相同型号的网卡
- **PCIe 插槽**: 确保网卡插在高速 PCIe 插槽上
- **散热设计**: 注意网卡散热，避免温度过高影响性能

### 2. 软件配置优化
```bash
# 生产环境推荐配置
export NCCL_SRS_ALGORITHM=performance
export NCCL_SRS_AFFINITY=1
export NCCL_SRS_BENCHMARK_TIME=3

# 启用监控（调试时）
export NCCL_SRS_VERBOSE=1
export NCCL_SRS_MONITOR=1
```

### 3. 性能调优流程
1. **基线测试**: 使用默认配置测试性能
2. **算法调优**: 尝试不同的选择算法
3. **手动调优**: 根据实际情况手动指定最优网卡
4. **监控验证**: 使用遥测功能验证优化效果

### 4. 异构环境部署
- **渐进升级**: 逐步替换老旧硬件
- **性能基准**: 建立不同硬件配置的性能基准
- **配置管理**: 使用配置管理工具统一管理不同节点

---

!!! tip "性能优化建议"
    在异构环境中，建议定期运行基准测试，因为硬件性能可能因温度、负载等因素变化。

!!! warning "兼容性注意"
    某些老型号网卡可能不支持所有 SRS 功能，建议查看兼容性列表。

!!! info "硬件亲和性"
    正确的 GPU-网卡亲和性配置可以显著提升性能，建议使用 `nvidia-smi topo -m` 查看拓扑关系。
