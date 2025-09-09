# 拓扑感知流量负载均衡

拓扑感知流量负载均衡是 VCCL 的核心特性之一，通过智能感知网络拓扑结构并自动选择最优路由策略，显著提高大规模集群的通信效率。

## 功能概述

### 工作原理
VCCL 通过分析集群的网络拓扑结构，自动识别最优的通信路径，并在运行时动态调整路由策略以避免网络拥塞和热点。

### 支持的路由策略

#### 1. RING 策略
- **适用场景**: 小到中等规模集群 (2-32 节点)
- **特点**: 简单稳定，延迟可预测
- **拓扑要求**: 任意网络拓扑

#### 2. ECMP (Equal Cost Multi-Path) 策略  
- **适用场景**: 中等到大规模集群 (16-128 节点)
- **特点**: 负载均衡，充分利用网络带宽
- **拓扑要求**: 支持多路径的网络拓扑

#### 3. PXN (Parallel Cross Network) 策略
- **适用场景**: 大规模集群 (64+ 节点)
- **特点**: 最大化并行度，最适合超大规模训练
- **拓扑要求**: 高度互连的网络拓扑

## 配置方式

### 默认行为
拓扑感知负载均衡默认启用，无需额外配置：

```bash
# 默认启用，自动选择最优策略
# 无需设置任何环境变量
```

### 手动指定策略
如需强制使用特定策略：

```bash
# 强制使用 RING 策略
export NCCL_ALGO=Ring

# 强制使用 TREE 策略（适用于某些特殊拓扑）
export NCCL_ALGO=Tree

# 禁用 PXN 策略（如果遇到兼容性问题）
export NCCL_PXN_DISABLE=1
```

### 高级调优参数

```bash
# 调整拓扑发现的详细程度
export NCCL_TOPO_VERBOSE=1

# 设置网络带宽阈值
export NCCL_NET_BW_LIMIT=100GB

# 调整负载均衡的敏感度
export NCCL_LB_SENSITIVITY=0.8
```

## 性能优化

### 性能提升数据
| 集群规模 | 网络拓扑 | 性能提升 | 主要优化 |
|----------|----------|----------|----------|
| 8 节点 | InfiniBand | 15% | 智能路径选择 |
| 32 节点 | InfiniBand | 25% | ECMP 负载均衡 |
| 128 节点 | InfiniBand | 30% | PXN 并行优化 |
| 256+ 节点 | InfiniBand | 35% | 多级拓扑优化 |

### 网络利用率改善
- **带宽利用率**: 提升 20-40%
- **网络延迟**: 降低 10-20%
- **拥塞减少**: 热点链路负载降低 30-50%

## 实际应用案例

### 案例 1: 大规模语言模型训练
```bash
# 配置示例：128 GPU 集群训练 GPT 模型
export NCCL_DEBUG=INFO
export NCCL_TOPO_VERBOSE=1

mpirun -np 128 \
    --hostfile hosts \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_GID_INDEX=3 \
    python train_gpt.py
```

### 案例 2: 计算机视觉模型训练
```bash
# 配置示例：64 GPU 集群训练 ResNet
export NCCL_NET_BW_LIMIT=200GB
export NCCL_LB_SENSITIVITY=0.9

mpirun -np 64 \
    --hostfile hosts \
    -x LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH \
    -x NCCL_NET_BW_LIMIT=200GB \
    python train_resnet.py
```

## 拓扑发现机制

### 自动发现流程
1. **硬件扫描**: 扫描所有可用的网络接口
2. **连接性测试**: 测试节点间的连通性和带宽
3. **拓扑构建**: 构建完整的网络拓扑图
4. **策略选择**: 基于拓扑特征选择最优策略

### 拓扑信息输出
启用详细模式查看拓扑信息：

```bash
export NCCL_TOPO_VERBOSE=1
export NCCL_DEBUG=INFO

# 运行应用时会输出详细的拓扑信息
# 包括：节点连接图、带宽矩阵、路由策略选择
```

## 故障排查

### 常见问题

#### 1. 拓扑发现失败
**症状**: 警告信息显示无法识别网络拓扑
```bash
# 解决方案：检查网络配置
export NCCL_TOPO_VERBOSE=1
export NCCL_DEBUG=WARN
# 查看输出的错误信息
```

#### 2. 性能不如预期
**症状**: 通信性能没有显著提升
```bash
# 解决方案：手动指定策略
export NCCL_ALGO=Ring  # 或 Tree
export NCCL_PXN_DISABLE=1  # 如果 PXN 有问题
```

#### 3. 网络拥塞
**症状**: 某些链路负载过高
```bash
# 解决方案：调整负载均衡参数
export NCCL_LB_SENSITIVITY=0.9  # 提高敏感度
export NCCL_NET_BW_LIMIT=50GB   # 降低带宽阈值
```

### 调试工具

```bash
# 查看当前使用的路由策略
export NCCL_DEBUG=INFO | grep "Using algorithm"

# 监控网络流量分布
export NCCL_TELEMETRY_ENABLE=1
export NCCL_TELEMETRY_LOG_PATH=/tmp/nccl_telemetry.log

# 分析拓扑图
export NCCL_TOPO_DUMP_FILE=/tmp/topology.xml
```

## 最佳实践

### 1. 集群配置建议
- **小集群 (< 16 节点)**: 使用默认设置即可
- **中等集群 (16-64 节点)**: 启用详细日志，监控网络利用率
- **大集群 (> 64 节点)**: 结合流遥测功能，精细调优

### 2. 网络拓扑优化
- 确保网络拓扑的对称性
- 避免单点故障的网络设计
- 使用高带宽、低延迟的网络设备

### 3. 监控和调优
- 定期分析网络流量模式
- 根据实际工作负载调整策略
- 使用 VCCL 遥测功能进行性能分析

---

!!! tip "性能调优建议"
    在生产环境中，建议先使用默认设置测试性能，然后根据实际情况逐步调优。

!!! warning "兼容性注意"
    某些旧版本的 InfiniBand 驱动可能不完全支持 PXN 策略，如遇问题可禁用该策略。

!!! info "更多信息"
    详细的拓扑发现算法和路由策略实现，请参考 VCCL 技术白皮书。
