# 流可视化遥测 (Flow Telemetry)

VCCL 流可视化遥测功能提供实时的通信流跟踪和诊断能力，帮助用户深度理解分布式训练的通信模式，识别性能瓶颈并进行精准优化。

## 功能概述

### 核心能力
- **实时监控**: 捕获所有 NCCL 通信操作的详细信息
- **流量分析**: 分析通信模式、带宽利用率和延迟分布
- **热点识别**: 自动识别网络热点和性能瓶颈
- **历史追踪**: 可配置的时间窗口进行历史数据分析

### 监控维度
| 维度 | 描述 | 用途 |
|------|------|------|
| **带宽利用率** | 实时网络带宽使用情况 | 识别网络饱和度 |
| **通信延迟** | 端到端通信延迟统计 | 定位延迟瓶颈 |
| **消息大小** | 通信消息的大小分布 | 优化消息聚合 |
| **通信拓扑** | 节点间通信模式 | 优化节点分配 |
| **错误统计** | 通信错误和重试次数 | 识别网络问题 |

## 配置方式

### 基本启用
```bash
# 启用遥测功能
export NCCL_TELEMETRY_ENABLE=1

# 设置数据窗口大小（默认50）
export TELEMETRY_WINDOWSIZE=100

# 设置日志输出路径
export NCCL_TELEMETRY_LOG_PATH=/tmp/vccl_telemetry
```

### 高级配置
```bash
# 设置采样率（0.0-1.0，默认1.0）
export NCCL_TELEMETRY_SAMPLE_RATE=0.8

# 设置刷新间隔（秒，默认5）
export NCCL_TELEMETRY_FLUSH_INTERVAL=10

# 启用详细模式
export NCCL_TELEMETRY_VERBOSE=1

# 设置输出格式（json/csv/binary）
export NCCL_TELEMETRY_FORMAT=json

# 启用实时输出
export NCCL_TELEMETRY_REALTIME=1
```

### 选择性监控
```bash
# 只监控特定操作类型
export NCCL_TELEMETRY_OPS=allreduce,broadcast

# 只监控特定 GPU
export NCCL_TELEMETRY_RANKS=0,1,2,3

# 设置消息大小过滤器（字节）
export NCCL_TELEMETRY_MIN_SIZE=1024
export NCCL_TELEMETRY_MAX_SIZE=1073741824  # 1GB
```

## 数据格式和字段

### JSON 格式示例
```json
{
  \"timestamp\": 1693958400.123,
  \"rank\": 0,
  \"operation\": \"allreduce\",
  \"algorithm\": \"ring\",
  \"message_size\": 4194304,
  \"duration_us\": 1250.5,
  \"bandwidth_gbps\": 26.8,
  \"src_rank\": 0,
  \"dst_rank\": 1,
  \"channel_id\": 0,
  \"stream_id\": 7,
  \"error_count\": 0,
  \"retry_count\": 0
}
```

### 关键字段说明
| 字段 | 类型 | 描述 |
|------|------|------|
| `timestamp` | float | Unix 时间戳（微秒精度） |
| `rank` | int | 当前进程的 rank ID |
| `operation` | string | 操作类型（allreduce/broadcast/reduce等） |
| `algorithm` | string | 使用的算法（ring/tree/ecmp等） |
| `message_size` | int | 消息大小（字节） |
| `duration_us` | float | 操作持续时间（微秒） |
| `bandwidth_gbps` | float | 实际带宽（GB/s） |
| `channel_id` | int | 通信通道 ID |
| `error_count` | int | 错误次数 |

## 使用示例

### 基本监控
```bash
# 启动带遥测的训练
export NCCL_TELEMETRY_ENABLE=1
export TELEMETRY_WINDOWSIZE=50
export NCCL_TELEMETRY_LOG_PATH=/tmp/training_telemetry

mpirun -np 8 \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH \
    -x NCCL_TELEMETRY_ENABLE=1 \
    -x NCCL_TELEMETRY_LOG_PATH=/tmp/training_telemetry \
    python train_model.py
```

### 实时监控脚本
```bash
#!/bin/bash
# 实时监控脚本

export NCCL_TELEMETRY_ENABLE=1
export NCCL_TELEMETRY_REALTIME=1
export NCCL_TELEMETRY_FORMAT=json
export NCCL_TELEMETRY_LOG_PATH=/tmp/realtime_telemetry

# 启动训练（后台）
mpirun -np 16 \
    -x NCCL_TELEMETRY_ENABLE=1 \
    -x NCCL_TELEMETRY_REALTIME=1 \
    python train_model.py &

# 实时分析遥测数据
tail -f /tmp/realtime_telemetry/rank_*.json | \
    python analyze_telemetry.py --realtime
```

### 性能分析示例
```python
# analyze_telemetry.py
import json
import numpy as np
from pathlib import Path

def analyze_bandwidth(telemetry_dir):
    bandwidth_data = []
    
    for log_file in Path(telemetry_dir).glob(\"rank_*.json\"):
        with open(log_file) as f:
            for line in f:
                data = json.loads(line)
                if data['operation'] == 'allreduce':
                    bandwidth_data.append(data['bandwidth_gbps'])
    
    print(f\"Average Bandwidth: {np.mean(bandwidth_data):.2f} GB/s\")
    print(f\"P95 Bandwidth: {np.percentile(bandwidth_data, 95):.2f} GB/s\")
    print(f\"P99 Bandwidth: {np.percentile(bandwidth_data, 99):.2f} GB/s\")

analyze_bandwidth(\"/tmp/training_telemetry\")
```

## 性能影响和开销

### 资源开销
| 配置 | 内存开销 | CPU 开销 | 磁盘 I/O |
|------|----------|---------|----------|
| 基本监控 | 2-5 MB | < 1% | 10-50 MB/小时 |
| 详细监控 | 5-10 MB | 1-3% | 100-500 MB/小时 |
| 实时模式 | 3-8 MB | 2-5% | 持续写入 |

### 性能影响
- **延迟增加**: < 5 微秒每操作
- **带宽影响**: < 1% 带宽开销
- **训练速度**: 通常 < 2% 影响

### 优化建议
```bash
# 生产环境优化配置
export NCCL_TELEMETRY_SAMPLE_RATE=0.1     # 降低采样率
export NCCL_TELEMETRY_FLUSH_INTERVAL=30   # 增加刷新间隔
export NCCL_TELEMETRY_MIN_SIZE=65536      # 只监控大消息
```

## 数据分析和可视化

### 性能分析工具
```bash
# 使用内置分析工具
python -m vccl.telemetry.analyzer \
    --input /tmp/training_telemetry \
    --output /tmp/analysis_report.html \
    --format html

# 生成性能报告
python -m vccl.telemetry.report \
    --telemetry-dir /tmp/training_telemetry \
    --report-type bandwidth,latency,topology
```

### 可视化示例
```python
# 带宽时序图
import matplotlib.pyplot as plt
import pandas as pd

def plot_bandwidth_timeline(telemetry_file):
    data = []
    with open(telemetry_file) as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['bandwidth_gbps'])
    plt.title('Bandwidth Over Time')
    plt.xlabel('Time')
    plt.ylabel('Bandwidth (GB/s)')
    plt.grid(True)
    plt.show()

plot_bandwidth_timeline('/tmp/training_telemetry/rank_0.json')
```

## 故障排查和诊断

### 常见性能问题识别

#### 1. 网络带宽瓶颈
```bash
# 查找带宽利用率异常
grep \"bandwidth_gbps\" /tmp/telemetry/*.json | \
    awk -F'\"' '{print $8}' | sort -n | tail -10
```

#### 2. 通信延迟异常
```bash
# 分析延迟分布
python -c \"
import json
latencies = []
with open('/tmp/telemetry/rank_0.json') as f:
    for line in f:
        data = json.loads(line)
        latencies.append(data['duration_us'])

import numpy as np
print(f'P99 latency: {np.percentile(latencies, 99):.2f} μs')
print(f'Max latency: {max(latencies):.2f} μs')
\"
```

#### 3. 通信模式分析
```bash
# 分析通信拓扑热点
python -m vccl.telemetry.topology_analyzer \
    --input /tmp/telemetry \
    --output /tmp/topology_heatmap.png
```

### 自动异常检测
```bash
# 启用异常检测
export NCCL_TELEMETRY_ANOMALY_DETECTION=1
export NCCL_TELEMETRY_ANOMALY_THRESHOLD=2.0  # 标准差阈值

# 异常会自动记录到单独的日志文件
tail -f /tmp/telemetry/anomalies.log
```

## 最佳实践

### 1. 生产环境使用
```bash
# 推荐的生产配置
export NCCL_TELEMETRY_ENABLE=1
export TELEMETRY_WINDOWSIZE=100
export NCCL_TELEMETRY_SAMPLE_RATE=0.1
export NCCL_TELEMETRY_MIN_SIZE=1048576  # 只监控 >= 1MB 的消息
export NCCL_TELEMETRY_LOG_PATH=/shared/telemetry
```

### 2. 调试和优化
```bash
# 调试时的详细配置
export NCCL_TELEMETRY_ENABLE=1
export NCCL_TELEMETRY_VERBOSE=1
export NCCL_TELEMETRY_REALTIME=1
export NCCL_TELEMETRY_SAMPLE_RATE=1.0
```

### 3. 数据管理
```bash
# 自动清理老数据
find /tmp/telemetry -name \"*.json\" -mtime +7 -delete

# 压缩历史数据
tar -czf telemetry_$(date +%Y%m%d).tar.gz /tmp/telemetry/*.json
```

### 4. 集成监控系统
```bash
# 集成到 Prometheus/Grafana
python -m vccl.telemetry.prometheus_exporter \
    --telemetry-dir /tmp/telemetry \
    --port 9090 &
```

---

!!! tip "性能监控建议"
    在长时间训练中，建议定期分析遥测数据以识别性能退化趋势。

!!! warning "存储空间"
    详细的遥测数据可能产生大量日志文件，请确保有足够的存储空间并定期清理。

!!! info "隐私注意"
    遥测数据可能包含敏感的网络拓扑信息，请妥善保管日志文件。
