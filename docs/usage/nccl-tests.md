# NCCL 测试

本页面介绍如何使用 NCCL 官方测试套件验证 VCCL 的功能和性能，包括基准测试、性能评估和故障排查的详细指南。

## 测试环境准备

### 获取 NCCL 测试套件
```bash
# 克隆官方测试仓库
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# 设置 VCCL 库路径
export LD_LIBRARY_PATH=/path/to/vccl/build/lib:$LD_LIBRARY_PATH

# 编译测试套件
make
```

### 编译选项
```bash
# 基本编译
make

# 指定 NCCL 路径编译
make NCCL_HOME=/opt/vccl

# 启用 MPI 支持
make MPI=1

# 编译 CUDA 版本
make CUDA_HOME=/usr/local/cuda
```

## 基本测试配置

### 必要的环境变量
```bash
# VCCL 特定配置
export LD_LIBRARY_PATH=/workspace/infrawaves/share/liuda/vc226/vccl_2.26.6-1/build/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=REG

# 网络配置
export NCCL_IB_GID_INDEX=3

# 容错配置 (VCCL 默认启用)
export NCCL_ENABLE_FAULT_TOLERANCE=1

# SM Free 模式 (可选)
export NCCL_PASS_SM=1
```

### 基础网络测试
```bash
# 检查 InfiniBand 状态
ibstat
ibdev2netdev

# 测试网络连通性
ping <remote_host>
ib_write_bw <remote_host>
```

## 单机测试

### AllReduce 测试
```bash
# 基本 AllReduce 测试
mpirun -np 8 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_DEBUG=INFO \\
    ./build/allreduce_perf -b 1KB -e 1GB -f 2 -g 1

# 参数说明：
# -b: 起始消息大小
# -e: 结束消息大小  
# -f: 大小增长因子
# -g: 每个进程的 GPU 数量
```

### SendRecv 测试 (SM Free 模式)
```bash
# 启用 SM Free 模式的 SendRecv 测试
mpirun -np 2 \\
    --host 10.1.3.201:2 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH=/workspace/infrawaves/share/liuda/vc226/vccl_2.26.6-1/build/lib:$LD_LIBRARY_PATH \\
    -x NCCL_DEBUG=INFO \\
    -x MASTER_PORT=29500 \\
    -x NCCL_DEBUG_SUBSYS=REG \\
    -x NCCL_IB_GID_INDEX=3 \\
    -x NCCL_PASS_SM=1 \\
    ./build/sendrecv_perf -b 1KB -e 2GB -f 2 -g 1 -R 1

# 参数说明：
# -R: 运行次数
```

### Broadcast 测试
```bash
# Broadcast 性能测试
mpirun -np 4 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_DEBUG=WARN \\
    ./build/broadcast_perf -b 1MB -e 128MB -f 2 -g 2
```

### Reduce 测试
```bash
# Reduce 操作测试
mpirun -np 4 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_DEBUG=INFO \\
    ./build/reduce_perf -b 4KB -e 64MB -f 2 -g 1
```

## 多机测试

### 主机文件配置
```bash
# 创建主机文件 hostfile
cat > hostfile << EOF
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
EOF
```

### 跨节点 AllReduce 测试
```bash
# 多节点 AllReduce 测试
mpirun -np 32 \\
    --hostfile hostfile \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_DEBUG=INFO \\
    -x NCCL_IB_GID_INDEX=3 \\
    -x NCCL_ENABLE_FAULT_TOLERANCE=1 \\
    ./build/allreduce_perf -b 8KB -e 512MB -f 2 -g 1
```

### 大规模集群测试
```bash
# 128 GPU 大规模测试
mpirun -np 128 \\
    --hostfile large_cluster_hosts \\
    --allow-run-as-root \\
    --map-by ppr:8:node \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_DEBUG=WARN \\
    -x NCCL_IB_GID_INDEX=3 \\
    -x NCCL_ENABLE_FAULT_TOLERANCE=1 \\
    -x NCCL_PASS_SM=1 \\
    ./build/allreduce_perf -b 64KB -e 1GB -f 2 -g 1
```

## VCCL 特性测试

### 容错功能测试
```bash
# 启用容错功能的测试
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_RETRY_COUNT=10
export NCCL_TIMEOUT=30

mpirun -np 16 \\
    --hostfile hostfile \\
    --allow-run-as-root \\
    -x NCCL_ENABLE_FAULT_TOLERANCE=1 \\
    -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1 \\
    -x NCCL_DEBUG=INFO \\
    ./build/allreduce_perf -b 1MB -e 100MB -f 2 -g 1
```

### 流遥测功能测试
```bash
# 启用遥测功能测试
export NCCL_TELEMETRY_ENABLE=1
export TELEMETRY_WINDOWSIZE=50
export NCCL_TELEMETRY_LOG_PATH=/tmp/nccl_telemetry

mpirun -np 8 \\
    --allow-run-as-root \\
    -x NCCL_TELEMETRY_ENABLE=1 \\
    -x TELEMETRY_WINDOWSIZE=50 \\
    -x NCCL_TELEMETRY_LOG_PATH=/tmp/nccl_telemetry \\
    ./build/allreduce_perf -b 4KB -e 64MB -f 2 -g 1

# 查看遥测数据
ls -la /tmp/nccl_telemetry/
head -10 /tmp/nccl_telemetry/rank_0.log
```

### 拓扑感知测试
```bash
# 启用拓扑感知详细信息
export NCCL_TOPO_VERBOSE=1
export NCCL_DEBUG=INFO

mpirun -np 16 \\
    --hostfile hostfile \\
    -x NCCL_TOPO_VERBOSE=1 \\
    -x NCCL_DEBUG=INFO \\
    ./build/allreduce_perf -b 1MB -e 32MB -f 2 -g 1 2>&1 | \\
    grep -E \"(TOPO|RING|TREE|ECMP)\"
```

### PXN 策略测试
```bash
# 测试 PXN 策略（默认启用）
mpirun -np 32 \\
    --hostfile hostfile \\
    -x NCCL_DEBUG=INFO \\
    -x NCCL_ALGO=Ring \\  # 或强制其他算法
    ./build/allreduce_perf -b 8MB -e 128MB -f 2 -g 1

# 禁用 PXN 进行对比测试
mpirun -np 32 \\
    --hostfile hostfile \\
    -x NCCL_PXN_DISABLE=1 \\
    -x NCCL_DEBUG=INFO \\
    ./build/allreduce_perf -b 8MB -e 128MB -f 2 -g 1
```

## 性能基准测试

### 标准基准测试套件
```bash
#!/bin/bash
# 完整的 VCCL 性能基准测试脚本

# 设置基本环境
export LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3

# 测试结果目录
RESULT_DIR=\"/tmp/vccl_benchmark_$(date +%Y%m%d_%H%M%S)\"
mkdir -p $RESULT_DIR

# 1. AllReduce 基准测试
echo \"Running AllReduce benchmark...\"
mpirun -np 8 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_DEBUG=WARN \\
    ./build/allreduce_perf -b 1KB -e 1GB -f 2 -g 1 \\
    > $RESULT_DIR/allreduce_result.txt

# 2. Broadcast 基准测试
echo \"Running Broadcast benchmark...\"
mpirun -np 8 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    ./build/broadcast_perf -b 1KB -e 1GB -f 2 -g 1 \\
    > $RESULT_DIR/broadcast_result.txt

# 3. SendRecv 基准测试
echo \"Running SendRecv benchmark...\"
mpirun -np 2 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    -x NCCL_PASS_SM=1 \\
    ./build/sendrecv_perf -b 1KB -e 1GB -f 2 -g 1 \\
    > $RESULT_DIR/sendrecv_result.txt

echo \"Benchmark completed. Results saved to $RESULT_DIR\"
```

### 性能对比测试
```bash
# 原生 NCCL vs VCCL 性能对比脚本
#!/bin/bash

# 测试原生 NCCL
export LD_LIBRARY_PATH=/usr/local/nccl/lib:$LD_LIBRARY_PATH
mpirun -np 8 ./build/allreduce_perf -b 1MB -e 100MB -f 2 -g 1 \\
    > nccl_original_result.txt

# 测试 VCCL
export LD_LIBRARY_PATH=/path/to/vccl/lib:$LD_LIBRARY_PATH
mpirun -np 8 ./build/allreduce_perf -b 1MB -e 100MB -f 2 -g 1 \\
    > vccl_result.txt

# 分析结果
python analyze_performance.py nccl_original_result.txt vccl_result.txt
```

### 延迟和带宽测试
```bash
# 延迟测试 (小消息)
mpirun -np 2 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    ./build/sendrecv_perf -b 4B -e 4KB -f 2 -g 1 -R 1000

# 带宽测试 (大消息)  
mpirun -np 2 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH \\
    ./build/sendrecv_perf -b 1MB -e 1GB -f 2 -g 1 -R 100
```

## 结果分析

### 性能指标解读
```bash
# 典型的测试输出解释
#       size         count      type   redop     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)       
           8             2     float     sum    28.93    0.00    0.00  2e-07
          16             4     float     sum    29.21    0.00    0.00  2e-07
          32             8     float     sum    29.84    0.00    0.00  2e-07

# 字段说明:
# size: 消息大小
# count: 元素数量
# time: 平均时间 (微秒)
# algbw: 算法带宽 (GB/s)
# busbw: 总线带宽 (GB/s)
# error: 数值误差
```

### 性能分析脚本
```python
#!/usr/bin/env python3
# analyze_nccl_results.py

import re
import sys
import matplotlib.pyplot as plt

def parse_results(filename):
    sizes = []
    times = []
    algbw = []
    
    with open(filename, 'r') as f:
        for line in f:
            if re.match(r'^\\s*\\d+', line):
                parts = line.split()
                sizes.append(int(parts[0]))
                times.append(float(parts[4]))
                algbw.append(float(parts[5]))
    
    return sizes, times, algbw

def plot_performance(sizes, algbw, title):
    plt.figure(figsize=(10, 6))
    plt.semilogx(sizes, algbw, 'o-')
    plt.xlabel('Message Size (Bytes)')
    plt.ylabel('Algorithm Bandwidth (GB/s)')
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(\"Usage: python analyze_nccl_results.py <result_file>\")
        sys.exit(1)
    
    sizes, times, algbw = parse_results(sys.argv[1])
    plot_performance(sizes, algbw, 'VCCL Performance Analysis')
    
    # 打印关键指标
    peak_bw = max(algbw)
    peak_size = sizes[algbw.index(peak_bw)]
    print(f\"Peak bandwidth: {peak_bw:.2f} GB/s at {peak_size} bytes\")
```

## 故障排查

### 常见错误和解决方案

#### 1. 库文件找不到
```bash
# 错误: libnccl.so.2: cannot open shared object file
# 解决方案:
export LD_LIBRARY_PATH=/path/to/vccl/build/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
ldd ./build/allreduce_perf
```

#### 2. GPU 通信失败
```bash
# 错误: NCCL WARN Call to connect returned Connection refused
# 解决方案:
# 检查防火墙设置
sudo ufw status
sudo firewall-cmd --list-all

# 检查 InfiniBand 状态
ibstat
ibdev2netdev

# 重启 InfiniBand 服务
sudo systemctl restart openibd
```

#### 3. 内存不足
```bash
# 错误: CUDA out of memory
# 解决方案:
nvidia-smi  # 查看显存使用情况

# 减少测试规模
mpirun -np 4 ./build/allreduce_perf -b 1KB -e 10MB -f 2 -g 1

# 或启用内存优化
export NCCL_CUMEM_ENABLE=1
```

#### 4. 网络配置问题
```bash
# 错误: NCCL WARN NET/IB: No device found
# 解决方案:
# 检查 IB 设备
ls /dev/infiniband/
ibv_devices

# 检查 GID 索引
show_gids

# 设置正确的 GID 索引
export NCCL_IB_GID_INDEX=3
```

### 调试技巧

#### 启用详细日志
```bash
# 最详细的调试信息
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

# 特定子系统调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
```

#### 网络诊断
```bash
# IB 网络诊断
ibdiagnet
ibnetdiscover

# 性能测试
ib_write_bw <remote_host>
ib_send_lat <remote_host>
```

#### 进程跟踪
```bash
# 使用 strace 跟踪系统调用
mpirun -np 2 strace -f -o trace.log ./build/allreduce_perf

# 使用 gdb 调试
mpirun -np 1 gdb --args ./build/allreduce_perf -b 1KB -e 1MB
```

## 自动化测试脚本

### 完整测试套件
```bash
#!/bin/bash
# vccl_test_suite.sh - VCCL 完整测试套件

set -e

# 配置
VCCL_LIB_PATH=\"/path/to/vccl/build/lib\"
NCCL_TESTS_DIR=\"./nccl-tests\"
RESULT_DIR=\"/tmp/vccl_test_$(date +%Y%m%d_%H%M%S)\"

# 环境设置
export LD_LIBRARY_PATH=\"$VCCL_LIB_PATH:$LD_LIBRARY_PATH\"
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3

mkdir -p \"$RESULT_DIR\"

echo \"VCCL Test Suite Starting...\"
echo \"Results will be saved to: $RESULT_DIR\"

# 测试函数
run_test() {
    local test_name=$1
    local np=$2
    local cmd=$3
    
    echo \"Running $test_name...\"
    if timeout 300 mpirun -np $np --allow-run-as-root $cmd > \"$RESULT_DIR/${test_name}.log\" 2>&1; then
        echo \"✓ $test_name PASSED\"
    else
        echo \"✗ $test_name FAILED\"
        tail -20 \"$RESULT_DIR/${test_name}.log\"
    fi
}

# 执行测试
cd \"$NCCL_TESTS_DIR\"

run_test \"allreduce_small\" 4 \"./build/allreduce_perf -b 1KB -e 1MB -f 2 -g 1\"
run_test \"allreduce_large\" 4 \"./build/allreduce_perf -b 1MB -e 100MB -f 2 -g 1\"
run_test \"broadcast\" 4 \"./build/broadcast_perf -b 1KB -e 10MB -f 2 -g 1\"
run_test \"sendrecv_sm_free\" 2 \"-x NCCL_PASS_SM=1 ./build/sendrecv_perf -b 1KB -e 10MB -f 2 -g 1\"

echo \"Test suite completed. Check $RESULT_DIR for detailed results.\"
```

---

!!! tip \"测试建议\"
    建议从小规模测试开始，逐步增加节点数量和消息大小，以便快速定位问题。

!!! warning \"资源使用\"
    大规模测试可能消耗大量 GPU 内存和网络带宽，建议在非生产时间进行。

!!! info \"性能基准\"
    建立性能基准数据库，定期运行测试以监控 VCCL 性能变化趋势。
