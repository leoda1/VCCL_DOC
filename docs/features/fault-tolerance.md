# 容错机制 (Fault Tolerance)

VCCL 容错机制确保在节点故障、网络中断或硬件异常的情况下，分布式训练能够自动恢复并继续执行，显著提高大规模集群的可靠性和可用性。

## 功能概述

### 容错能力
- **节点故障检测**: 自动检测 GPU 节点故障和离线
- **网络故障恢复**: 处理 InfiniBand 链路中断和恢复
- **通信重试机制**: 智能重试失败的通信操作
- **拓扑重构**: 动态调整通信拓扑以绕过故障节点
- **状态同步**: 保持训练状态的一致性

### 故障类型支持
| 故障类型 | 检测方式 | 恢复策略 | 恢复时间 |
|----------|----------|----------|----------|
| **节点掉线** | 心跳检测 | 拓扑重构 | 5-30 秒 |
| **网络中断** | 通信超时 | 路径重路由 | 1-10 秒 |
| **GPU 故障** | CUDA 错误 | 节点隔离 | 10-60 秒 |
| **内存错误** | ECC 检测 | 数据重传 | 1-5 秒 |
| **软件异常** | 异常捕获 | 进程重启 | 5-20 秒 |

## 配置方式

### 基本启用
```bash
# 启用容错功能（默认启用）
export NCCL_ENABLE_FAULT_TOLERANCE=1

# 必须指定网卡配置
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
```

### 高级配置
```bash
# 设置重试次数（默认7）
export NCCL_RETRY_COUNT=10

# 设置超时时间（秒，默认18）
export NCCL_TIMEOUT=30

# 启用心跳检测
export NCCL_HEARTBEAT_ENABLE=1
export NCCL_HEARTBEAT_INTERVAL=5  # 心跳间隔（秒）

# 配置故障检测灵敏度
export NCCL_FAULT_DETECTION_THRESHOLD=3
export NCCL_FAULT_RECOVERY_TIMEOUT=60
```

### 容错级别配置
```bash
# 基础容错（推荐用于生产环境）
export NCCL_FAULT_TOLERANCE_LEVEL=basic
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_RETRY_COUNT=7
export NCCL_TIMEOUT=18

# 增强容错（用于不稳定环境）
export NCCL_FAULT_TOLERANCE_LEVEL=enhanced
export NCCL_RETRY_COUNT=15
export NCCL_TIMEOUT=60
export NCCL_HEARTBEAT_ENABLE=1

# 最大容错（用于实验环境）
export NCCL_FAULT_TOLERANCE_LEVEL=maximum
export NCCL_RETRY_COUNT=25
export NCCL_TIMEOUT=120
export NCCL_AUTO_RECOVERY=1
```

## 实际应用场景

### 场景 1: 长时间训练任务
```bash
# 7天×24小时训练配置
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_RETRY_COUNT=10
export NCCL_TIMEOUT=30
export NCCL_HEARTBEAT_ENABLE=1
export NCCL_AUTO_CHECKPOINT=1

mpirun -np 256 \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH \
    -x NCCL_ENABLE_FAULT_TOLERANCE=1 \
    -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1 \
    python train_large_model.py --checkpoint-every 1000
```

### 场景 2: 云环境抢占式实例
```bash
# 云环境配置（节点可能被回收）
export NCCL_FAULT_TOLERANCE_LEVEL=enhanced
export NCCL_PREEMPTION_DETECTION=1
export NCCL_GRACEFUL_SHUTDOWN=1

# 启用自动保存和恢复
export NCCL_AUTO_SAVE_STATE=1
export NCCL_STATE_SAVE_INTERVAL=300  # 5分钟保存一次
```

### 场景 3: 实验环境（硬件不稳定）
```bash
# 实验环境配置
export NCCL_FAULT_TOLERANCE_LEVEL=maximum
export NCCL_DEBUG_FAULT_TOLERANCE=1
export NCCL_FAULT_LOG_LEVEL=verbose

# 详细的故障日志
export NCCL_FAULT_LOG_PATH=/tmp/fault_logs
```

## 故障检测机制

### 心跳检测
```bash
# 启用心跳监控
export NCCL_HEARTBEAT_ENABLE=1
export NCCL_HEARTBEAT_INTERVAL=5
export NCCL_HEARTBEAT_TIMEOUT=20

# 心跳检测会定期检查所有节点状态
# 发现故障节点时自动触发恢复流程
```

### 通信超时检测
```bash
# 配置超时检测
export NCCL_TIMEOUT=18              # 基础超时
export NCCL_CONNECT_TIMEOUT=60      # 连接超时
export NCCL_PROGRESS_TIMEOUT=10     # 进度超时

# 超时阈值自适应调整
export NCCL_ADAPTIVE_TIMEOUT=1
```

### 硬件故障检测
```bash
# GPU 故障检测
export NCCL_GPU_HEALTH_CHECK=1
export NCCL_GPU_CHECK_INTERVAL=30

# 网络故障检测
export NCCL_NET_HEALTH_CHECK=1
export NCCL_NET_CHECK_INTERVAL=10
```

## 恢复策略

### 自动重试
```python
# VCCL 内部重试逻辑示例
def robust_allreduce(tensor, retry_count=7):
    for attempt in range(retry_count):
        try:
            result = nccl.allreduce(tensor)
            return result
        except NCCLError as e:
            if attempt < retry_count - 1:
                # 记录错误并重试
                log_fault(f\"Attempt {attempt + 1} failed: {e}\")
                wait_backoff(attempt)
                continue
            else:
                # 最后一次尝试失败，触发故障恢复
                trigger_fault_recovery()
                raise
```

### 拓扑重构
```bash
# 故障节点检测后自动重构通信拓扑
# 原始拓扑: 8 节点全连接
# 故障后: 7 节点重新组织通信环

# 查看拓扑重构日志
export NCCL_TOPOLOGY_REBUILD_LOG=1
grep \"TOPOLOGY_REBUILD\" /tmp/nccl.log
```

### 数据恢复
```bash
# 启用数据校验和恢复
export NCCL_DATA_CHECKSUM=1
export NCCL_AUTO_DATA_RECOVERY=1

# 从检查点恢复
export NCCL_CHECKPOINT_RECOVERY=1
export NCCL_CHECKPOINT_PATH=/shared/checkpoints
```

## 性能影响

### 开销分析
| 容错级别 | CPU 开销 | 内存开销 | 网络开销 | 延迟增加 |
|----------|----------|----------|----------|----------|
| 基础 | < 1% | 5-10 MB | < 1% | < 10 μs |
| 增强 | 1-2% | 10-20 MB | 1-2% | 10-50 μs |
| 最大 | 2-5% | 20-50 MB | 2-5% | 50-100 μs |

### 故障恢复时间
```bash
# 不同故障类型的典型恢复时间
网络抖动: 1-5 秒
单节点故障: 10-30 秒
多节点故障: 30-120 秒
网络分区: 60-300 秒
```

## 监控和诊断

### 故障日志
```bash
# 启用详细的故障日志
export NCCL_DEBUG_FAULT_TOLERANCE=1
export NCCL_FAULT_LOG_PATH=/tmp/fault_logs
export NCCL_FAULT_LOG_LEVEL=verbose

# 查看故障统计
grep \"FAULT_STATS\" /tmp/fault_logs/nccl_fault.log
```

### 实时监控
```bash
# 容错状态监控
export NCCL_FAULT_MONITOR=1
export NCCL_FAULT_MONITOR_PORT=8080

# 通过 HTTP 接口查看状态
curl http://localhost:8080/fault_status
curl http://localhost:8080/recovery_stats
```

### 故障分析工具
```python
# 故障日志分析脚本
import json
from datetime import datetime

def analyze_fault_log(log_file):
    faults = []
    recoveries = []
    
    with open(log_file) as f:
        for line in f:
            if 'FAULT_DETECTED' in line:
                faults.append(parse_fault_event(line))
            elif 'FAULT_RECOVERED' in line:
                recoveries.append(parse_recovery_event(line))
    
    print(f\"Total faults: {len(faults)}\")
    print(f\"Successful recoveries: {len(recoveries)}\")
    print(f\"Recovery rate: {len(recoveries)/len(faults)*100:.1f}%\")

analyze_fault_log('/tmp/fault_logs/nccl_fault.log')
```

## 故障排查

### 常见问题

#### 1. 容错功能未生效
**症状**: 出现网络故障时训练直接失败

```bash
# 检查配置
echo $NCCL_ENABLE_FAULT_TOLERANCE
echo $NCCL_IB_HCA

# 确保正确配置
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1  # 必须指定
```

#### 2. 恢复时间过长
**症状**: 故障恢复时间超过预期

```bash
# 调整超时参数
export NCCL_TIMEOUT=10              # 减少超时时间
export NCCL_RETRY_COUNT=5           # 减少重试次数
export NCCL_FAULT_DETECTION_THRESHOLD=2  # 提高检测灵敏度
```

#### 3. 频繁误检测
**症状**: 正常情况下也频繁触发故障检测

```bash
# 降低检测灵敏度
export NCCL_FAULT_DETECTION_THRESHOLD=5
export NCCL_HEARTBEAT_TIMEOUT=30
export NCCL_TIMEOUT=30
```

### 调试工具
```bash
# 故障注入测试
export NCCL_FAULT_INJECTION=1
export NCCL_FAULT_INJECTION_RATE=0.01  # 1% 故障率

# 模拟不同类型的故障
export NCCL_SIMULATE_NODE_FAILURE=1
export NCCL_SIMULATE_NETWORK_FAILURE=1
```

## 最佳实践

### 1. 生产环境配置
```bash
# 稳定的生产环境推荐配置
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_RETRY_COUNT=7
export NCCL_TIMEOUT=18
export NCCL_HEARTBEAT_ENABLE=1
export NCCL_HEARTBEAT_INTERVAL=10
```

### 2. 与训练框架集成
```python
# PyTorch 集成示例
import torch.distributed as dist

def init_distributed_with_fault_tolerance():
    # 设置容错环境变量
    os.environ['NCCL_ENABLE_FAULT_TOLERANCE'] = '1'
    os.environ['NCCL_IB_HCA'] = 'mlx5_0:1,mlx5_1:1'
    
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    
    # 注册故障回调
    dist.register_fault_handler(handle_fault)

def handle_fault(fault_info):
    # 自定义故障处理逻辑
    print(f\"Fault detected: {fault_info}\")
    save_checkpoint()
```

### 3. 监控集成
```bash
# 集成到监控系统
python -m vccl.fault_tolerance.monitor \
    --export-prometheus \
    --port 9091 &

# Grafana 监控面板
# - 故障检测频率
# - 恢复成功率  
# - 平均恢复时间
```

### 4. 运维流程
1. **预防性维护**: 定期检查硬件状态
2. **故障演练**: 定期进行故障注入测试
3. **日志分析**: 分析故障模式，优化配置
4. **容量规划**: 考虑故障恢复的资源需求

---

!!! tip "容错调优建议"
    根据集群的实际稳定性调整容错参数，避免过于敏感的检测导致误报。

!!! warning "网卡配置要求"
    容错功能必须指定 `NCCL_IB_HCA` 环境变量，否则无法正常工作。

!!! info "与训练框架配合"
    容错机制与训练框架的检查点功能配合使用效果最佳，可以实现真正的训练任务容错。
