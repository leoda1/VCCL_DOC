# 环境变量参考

本页面提供 VCCL 所有环境变量的详细参考，包括功能说明、默认值、取值范围和使用示例。

## 核心功能环境变量

### 容错机制
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_ENABLE_FAULT_TOLERANCE` | `1` | `0, 1` | 启用/禁用容错功能 |
| `NCCL_RETRY_COUNT` | `7` | `1-50` | 通信失败重试次数 |
| `NCCL_TIMEOUT` | `18` | `5-300` | 通信超时时间（秒） |
| `NCCL_IB_HCA` | 无 | 字符串 | **必须设置** - 指定网卡配置 |
| `NCCL_HEARTBEAT_ENABLE` | `0` | `0, 1` | 启用心跳检测 |
| `NCCL_HEARTBEAT_INTERVAL` | `10` | `1-60` | 心跳间隔（秒） |

```bash
# 容错配置示例
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_RETRY_COUNT=10
export NCCL_TIMEOUT=30
```

### SM Free & Overlap
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_PASS_SM` | `0` | `0, 1` | 启用 SM Free 模式 |
| `NCCL_CUMEM_ENABLE` | `0` | `0, 1` | 启用 CUDA 内存池 |
| `NCCL_CUMEM_POOL_SIZE` | `2GB` | 字符串 | 内存池大小 |
| `NCCL_CUMEM_PREALLOC` | `0` | `0, 1` | 内存池预分配 |
| `NCCL_ZERO_COPY` | `0` | `0, 1` | 启用零拷贝优化 |
| `NCCL_DMA_ENABLE` | `0` | `0, 1` | 启用 DMA 传输 |

```bash
# SM Free 配置示例
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_POOL_SIZE=4GB
export NCCL_CUMEM_PREALLOC=1
```

### 流可视化遥测
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_TELEMETRY_ENABLE` | `0` | `0, 1` | 启用遥测功能 |
| `TELEMETRY_WINDOWSIZE` | `50` | `10-1000` | 数据窗口大小 |
| `NCCL_TELEMETRY_LOG_PATH` | 无 | 路径字符串 | 遥测日志输出路径 |
| `NCCL_TELEMETRY_SAMPLE_RATE` | `1.0` | `0.0-1.0` | 采样率 |
| `NCCL_TELEMETRY_FORMAT` | `json` | `json, csv, binary` | 输出格式 |
| `NCCL_TELEMETRY_REALTIME` | `0` | `0, 1` | 实时输出模式 |

```bash
# 遥测配置示例
export NCCL_TELEMETRY_ENABLE=1
export TELEMETRY_WINDOWSIZE=100
export NCCL_TELEMETRY_LOG_PATH=/tmp/vccl_telemetry
export NCCL_TELEMETRY_SAMPLE_RATE=0.8
```

### 拓扑感知负载均衡
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_ALGO` | `Auto` | `Ring, Tree, Auto` | 强制指定算法 |
| `NCCL_PXN_DISABLE` | `0` | `0, 1` | 禁用 PXN 策略 |
| `NCCL_TOPO_VERBOSE` | `0` | `0, 1` | 拓扑发现详细日志 |
| `NCCL_LB_SENSITIVITY` | `0.5` | `0.0-1.0` | 负载均衡敏感度 |
| `NCCL_NET_BW_LIMIT` | 无 | 带宽字符串 | 网络带宽限制 |

```bash
# 拓扑感知配置示例
export NCCL_TOPO_VERBOSE=1
export NCCL_LB_SENSITIVITY=0.8
export NCCL_NET_BW_LIMIT=100GB
```

### 智能 RNIC 选择
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_SRS_VERBOSE` | `0` | `0, 1` | SRS 详细日志 |
| `NCCL_SRS_ALGORITHM` | `performance` | `performance, bandwidth, latency` | 选择算法 |
| `NCCL_SRS_AFFINITY` | `1` | `0, 1` | 启用亲和性优化 |
| `NCCL_SRS_BENCHMARK_TIME` | `3` | `1-30` | 基准测试时间（秒） |
| `NCCL_SRS_MONITOR` | `0` | `0, 1` | 启用运行时监控 |

```bash
# SRS 配置示例
export NCCL_SRS_VERBOSE=1
export NCCL_SRS_ALGORITHM=performance
export NCCL_SRS_AFFINITY=1
```

## 网络配置环境变量

### InfiniBand 配置
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_IB_GID_INDEX` | `0` | `0-15` | IB GID 索引 |
| `NCCL_IB_HCA` | 自动检测 | 设备列表 | 指定 IB 设备 |
| `NCCL_IB_DISABLE` | 无 | 设备列表 | 禁用特定 IB 设备 |
| `NCCL_IB_TIMEOUT` | `18` | `1-100` | IB 传输超时 |
| `NCCL_IB_RETRY_CNT` | `7` | `1-20` | IB 重试次数 |
| `NCCL_IB_AR_THRESHOLD` | `8192` | 字节数 | 自适应路由阈值 |

```bash
# InfiniBand 配置示例
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
```

### 网络传输优化
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_NET_GDR_LEVEL` | `PHB` | `SYS, PHB, PIX` | GPU Direct RDMA 级别 |
| `NCCL_NET_GDR_READ` | `1` | `0, 1` | 启用 GDR 读取 |
| `NCCL_P2P_LEVEL` | `PIX` | `SYS, PHB, PIX` | P2P 传输级别 |
| `NCCL_BUFFSIZE` | `4194304` | 字节数 | 通信缓冲区大小 |
| `NCCL_NTHREADS` | `自动` | `1-64` | 网络线程数 |

```bash
# 网络传输优化示例
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_P2P_LEVEL=PIX
export NCCL_BUFFSIZE=8388608  # 8MB
```

## 调试和日志环境变量

### 调试级别
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_DEBUG` | `WARN` | `ERROR, WARN, INFO, TRACE` | 调试日志级别 |
| `NCCL_DEBUG_SUBSYS` | `ALL` | 子系统列表 | 指定调试子系统 |
| `NCCL_DEBUG_FILE` | 无 | 文件路径 | 调试日志文件 |
| `NCCL_LOG_LEVEL` | `WARN` | `ERROR, WARN, INFO, DEBUG` | 应用日志级别 |

```bash
# 调试配置示例
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
export NCCL_DEBUG_FILE=/tmp/nccl_debug.log
```

### 子系统调试选项
| 子系统 | 说明 |
|--------|------|
| `INIT` | 初始化过程 |
| `NET` | 网络通信 |
| `GRAPH` | 通信图构建 |
| `REG` | 内存注册 |
| `COLL` | 集合通信操作 |
| `P2P` | 点对点通信 |
| `SHM` | 共享内存 |
| `PROXY` | 代理线程 |

```bash
# 特定子系统调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=NET,COLL
```

## 性能调优环境变量

### 算法选择
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_ALGO` | `Auto` | `Ring, Tree, CollNet` | 算法选择 |
| `NCCL_PROTO` | `Simple` | `Simple, LL, LL128` | 协议选择 |
| `NCCL_MIN_NCHANNELS` | `1` | `1-32` | 最小通道数 |
| `NCCL_MAX_NCHANNELS` | `32` | `1-32` | 最大通道数 |
| `NCCL_NCHANNELS_PER_NET_PEER` | `1` | `1-8` | 每个网络对等点的通道数 |

```bash
# 算法调优示例
export NCCL_ALGO=Ring
export NCCL_PROTO=LL128
export NCCL_MAX_NCHANNELS=16
```

### 内存管理
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_BUFFSIZE` | `4194304` | 字节数 | 缓冲区大小 |
| `NCCL_LL_BUFFSIZE` | `1048576` | 字节数 | LL 缓冲区大小 |
| `NCCL_LL128_BUFFSIZE` | `134217728` | 字节数 | LL128 缓冲区大小 |
| `NCCL_GRAPH_MIXING_SUPPORT` | `1` | `0, 1` | 图混合支持 |

```bash
# 内存管理调优
export NCCL_BUFFSIZE=8388608
export NCCL_LL_BUFFSIZE=2097152
export NCCL_LL128_BUFFSIZE=268435456
```

## 高级功能环境变量

### 异常检测和恢复
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_FAULT_DETECTION_THRESHOLD` | `3` | `1-10` | 故障检测阈值 |
| `NCCL_FAULT_RECOVERY_TIMEOUT` | `60` | `10-300` | 故障恢复超时 |
| `NCCL_AUTO_RECOVERY` | `0` | `0, 1` | 自动恢复模式 |
| `NCCL_GRACEFUL_SHUTDOWN` | `0` | `0, 1` | 优雅关闭 |

```bash
# 异常恢复配置
export NCCL_FAULT_DETECTION_THRESHOLD=5
export NCCL_FAULT_RECOVERY_TIMEOUT=120
export NCCL_AUTO_RECOVERY=1
```

### 实验性功能
| 变量名 | 默认值 | 取值范围 | 说明 |
|--------|--------|----------|------|
| `NCCL_COMPRESSION` | `0` | `0, 1` | 启用数据压缩 |
| `NCCL_COMPRESSION_THRESHOLD` | `65536` | 字节数 | 压缩阈值 |
| `NCCL_ADAPTIVE_ROUTING` | `0` | `0, 1` | 自适应路由 |
| `NCCL_DYNAMIC_CHUNK_SIZE` | `0` | `0, 1` | 动态块大小 |

```bash
# 实验性功能
export NCCL_COMPRESSION=1
export NCCL_COMPRESSION_THRESHOLD=32768
export NCCL_ADAPTIVE_ROUTING=1
```

## 环境配置模板

### 开发环境
```bash
#!/bin/bash
# dev_environment.sh

# 基础配置
export LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH

# 调试配置
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 功能配置
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1
export NCCL_PASS_SM=0  # 开发时可能需要禁用
export NCCL_TELEMETRY_ENABLE=1
export NCCL_TELEMETRY_LOG_PATH=/tmp/dev_telemetry

# 网络配置
export NCCL_IB_GID_INDEX=3
export NCCL_TIMEOUT=30
```

### 生产环境
```bash
#!/bin/bash
# prod_environment.sh

# 基础配置
export LD_LIBRARY_PATH=/opt/vccl/lib:$LD_LIBRARY_PATH

# 生产日志配置
export NCCL_DEBUG=WARN
export NCCL_DEBUG_FILE=/var/log/nccl.log

# 核心功能
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1

# 性能优化
export NCCL_BUFFSIZE=8388608
export NCCL_MAX_NCHANNELS=16
export NCCL_P2P_LEVEL=PIX

# 可选遥测（生产环境可关闭以减少开销）
export NCCL_TELEMETRY_ENABLE=0
```

### 大规模集群环境
```bash
#!/bin/bash
# large_cluster_environment.sh

# 基础配置
export LD_LIBRARY_PATH=/shared/vccl/lib:$LD_LIBRARY_PATH

# 日志配置
export NCCL_DEBUG=ERROR  # 大规模集群减少日志
export NCCL_DEBUG_FILE=/shared/logs/nccl_$(hostname).log

# 容错配置
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1
export NCCL_RETRY_COUNT=15
export NCCL_TIMEOUT=60
export NCCL_HEARTBEAT_ENABLE=1

# 性能优化
export NCCL_PASS_SM=1
export NCCL_CUMEM_ENABLE=1
export NCCL_CUMEM_POOL_SIZE=8GB

# 拓扑优化
export NCCL_TOPO_VERBOSE=0  # 大规模时关闭详细输出
export NCCL_LB_SENSITIVITY=0.9

# 遥测配置（采样模式）
export NCCL_TELEMETRY_ENABLE=1
export NCCL_TELEMETRY_SAMPLE_RATE=0.1
export NCCL_TELEMETRY_LOG_PATH=/shared/telemetry
```

## 环境变量验证脚本

```bash
#!/bin/bash
# validate_vccl_env.sh

echo \"=== VCCL Environment Validation ===\"

# 检查必需的环境变量
check_required_vars() {
    echo \"Checking required environment variables...\"
    
    if [[ -z \"$LD_LIBRARY_PATH\" ]]; then
        echo \"❌ LD_LIBRARY_PATH not set\"
        return 1
    fi
    
    if [[ \"$NCCL_ENABLE_FAULT_TOLERANCE\" == \"1\" && -z \"$NCCL_IB_HCA\" ]]; then
        echo \"❌ NCCL_IB_HCA required when fault tolerance is enabled\"
        return 1
    fi
    
    echo \"✅ Required variables OK\"
}

# 检查库文件
check_libraries() {
    echo \"Checking VCCL libraries...\"
    
    if ! echo $LD_LIBRARY_PATH | grep -q vccl; then
        echo \"⚠️  VCCL library path not in LD_LIBRARY_PATH\"
    fi
    
    if ! ldconfig -p | grep -q libnccl; then
        echo \"❌ NCCL library not found in system\"
        return 1
    fi
    
    echo \"✅ Libraries OK\"
}

# 检查 GPU 和网络
check_hardware() {
    echo \"Checking hardware...\"
    
    if ! nvidia-smi &>/dev/null; then
        echo \"❌ NVIDIA GPU not available\"
        return 1
    fi
    
    if ! ibstat &>/dev/null; then
        echo \"⚠️  InfiniBand not available\"
    fi
    
    echo \"✅ Hardware OK\"
}

# 打印配置摘要
print_config_summary() {
    echo \"\"
    echo \"=== Current VCCL Configuration ===\"
    echo \"LD_LIBRARY_PATH: $LD_LIBRARY_PATH\"
    echo \"NCCL_DEBUG: ${NCCL_DEBUG:-WARN}\"
    echo \"NCCL_ENABLE_FAULT_TOLERANCE: ${NCCL_ENABLE_FAULT_TOLERANCE:-0}\"
    echo \"NCCL_PASS_SM: ${NCCL_PASS_SM:-0}\"
    echo \"NCCL_CUMEM_ENABLE: ${NCCL_CUMEM_ENABLE:-0}\"
    echo \"NCCL_TELEMETRY_ENABLE: ${NCCL_TELEMETRY_ENABLE:-0}\"
    echo \"NCCL_IB_HCA: ${NCCL_IB_HCA:-auto}\"
    echo \"NCCL_IB_GID_INDEX: ${NCCL_IB_GID_INDEX:-0}\"
}

# 运行验证
check_required_vars
check_libraries  
check_hardware
print_config_summary

echo \"\"
echo \"Validation completed.\"
```

---

!!! tip \"配置建议\"
    建议为不同环境（开发、测试、生产）创建标准化的配置模板，确保环境一致性。

!!! warning \"必需变量\"
    当启用容错功能时，必须设置 `NCCL_IB_HCA` 环境变量，否则容错功能无法正常工作。

!!! info \"性能调优\"
    环境变量的最优配置取决于具体的硬件和工作负载，建议通过实际测试确定最佳参数组合。
