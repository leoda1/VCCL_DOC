# VCCL Tests

This page describes how to use the NCCL official test suite to verify VCCL functionality and performance, including detailed guidelines for benchmarking, performance evaluation, and troubleshooting.

## Test Environment Setup

### Get NCCL Test Suite
```bash
# Clone official test repository
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests

# Set VCCL library path
export LD_LIBRARY_PATH=/path/to/vccl/build/lib:$LD_LIBRARY_PATH

# Build test suite
make
```

### Build Options
```bash
# Basic build
make

# Build with specified NCCL path
make NCCL_HOME=/opt/vccl

# Enable MPI support
make MPI=1

# Build CUDA version
make CUDA_HOME=/usr/local/cuda
```

## Basic Test Configuration

### Required Environment Variables
```bash
# VCCL specific configuration
export LD_LIBRARY_PATH=/workspace/VCCL/build/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO

# Network configuration
export NCCL_IB_GID_INDEX=3

# Fault tolerance configuration (VCCL default disabled)
export NCCL_ENABLE_FAULT_TOLERANCE=1

# SM Free mode (optional)
export NCCL_PASS_SM=1
```

### Basic Network Testing
```bash
# Check InfiniBand status
ibstat
ibdev2netdev

# Test network connectivity
ping <remote_host>
ib_write_bw <remote_host>
```

## Single-Node Testing

### AllReduce Test
```bash
# Basic AllReduce test
mpirun -np 8 \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/workspace/VCCL/build/lib:$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=INFO \
    ./build/allreduce_perf -b 1KB -e 1GB -f 2 -g 1

# Parameter description:
# -b: Starting message size
# -e: Ending message size
# -f: Size growth factor
# -g: Number of GPUs per process
```

### SendRecv Test (SM Free Mode)
```bash
# SendRecv test with SM Free mode enabled
mpirun -np 2 \
    --host 10.1.3.201:2 \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/workspace/VCCL/build/lib:$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=INFO \
    -x MASTER_PORT=29500 \
    -x NCCL_DEBUG_SUBSYS=REG \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_PASS_SM=1 \
    # SM Free mode also supports higher bandwidth Zerocopy mode, enable zerocopy with -R 1
    # -x NCCL_PSM_FORCE_ZEROCOPY=1 \
    ./build/sendrecv_perf -b 1KB -e 2GB -f 2 -g 1 -R 0

# Parameter description:
# -R: Default 0, if 1 will use ncclCommRegister
```

### Broadcast Test
```bash
# Broadcast performance test
mpirun -np 4 \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/workspace/VCCL/build/lib:$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=WARN \
    ./build/broadcast_perf -b 1MB -e 128MB -f 2 -g 2
```

### Reduce Test
```bash
# Reduce operation test
mpirun -np 4 \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/workspace/VCCL/build/lib:$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=INFO \
    ./build/reduce_perf -b 4KB -e 64MB -f 2 -g 1
```

## Multi-Node Testing

### Host File Configuration
```bash
# Create host file hostfile
cat > hostfile-4nodes << EOF
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
EOF
```

### Cross-Node AllReduce Test
```bash
# Multi-node AllReduce test
mpirun -np 32 \
    --hostfile hostfile-4nodes \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/workspace/VCCL/build/lib:$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=INFO \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_ENABLE_FAULT_TOLERANCE=1 \
    ./build/allreduce_perf -b 8KB -e 512MB -f 2 -g 1
```

### Large-Scale Cluster Test
```bash
# 128 GPU large-scale test
mpirun -np 128 \
    --hostfile large_cluster_hosts \
    --allow-run-as-root \
    -x LD_LIBRARY_PATH=/workspace/VCCL/build/lib:$LD_LIBRARY_PATH \
    -x NCCL_DEBUG=WARN \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_ENABLE_FAULT_TOLERANCE=1 \
    -x NCCL_PASS_SM=1 \
    ./build/allreduce_perf -b 64KB -e 1GB -f 2 -g 1
```

## VCCL Feature Testing

### Fault Tolerance Feature Test
```bash
# Test with fault tolerance enabled
export NCCL_ENABLE_FAULT_TOLERANCE=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_RETRY_COUNT=10
export NCCL_TIMEOUT=30

mpirun -np 16 \
    --hostfile hostfile \
    --allow-run-as-root \
    -x NCCL_ENABLE_FAULT_TOLERANCE=1 \
    -x NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1 \
    -x NCCL_DEBUG=INFO \
    ./build/allreduce_perf -b 1MB -e 100MB -f 2 -g 1
```

### Flow Telemetry Feature Test
```bash
# Test with telemetry enabled
export NCCL_TELEMETRY_ENABLE=1
export TELEMETRY_WINDOWSIZE=50
export NCCL_TELEMETRY_LOG_PATH=/tmp/nccl_telemetry

mpirun -np 8 \
    --allow-run-as-root \
    -x NCCL_TELEMETRY_ENABLE=1 \
    -x TELEMETRY_WINDOWSIZE=50 \
    -x NCCL_TELEMETRY_LOG_PATH=/tmp/nccl_telemetry \
    ./build/allreduce_perf -b 4KB -e 64MB -f 2 -g 1

# View telemetry data
ls -la /tmp/nccl_telemetry/
head -10 /tmp/nccl_telemetry/rank_0.log
```

## Result Analysis

### Performance Metrics Interpretation
```bash
# Typical test output explanation
#       size         count      type   redop     time   algbw   busbw  error
#        (B)    (elements)                       (us)  (GB/s)  (GB/s)       
           8             2     float     sum    28.93    0.00    0.00  2e-07
          16             4     float     sum    29.21    0.00    0.00  2e-07
          32             8     float     sum    29.84    0.00    0.00  2e-07

# Field description:
# size: Message size
# count: Element count
# time: Average time (microseconds)
# algbw: Algorithm bandwidth (GB/s)
# busbw: Bus bandwidth (GB/s)
# error: Numerical error
```

### Performance Analysis Script
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
            if re.match(r'^\s*\d+', line):
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
        print("Usage: python analyze_nccl_results.py <result_file>")
        sys.exit(1)
    
    sizes, times, algbw = parse_results(sys.argv[1])
    plot_performance(sizes, algbw, 'VCCL Performance Analysis')
    
    # Print key metrics
    peak_bw = max(algbw)
    peak_size = sizes[algbw.index(peak_bw)]
    print(f"Peak bandwidth: {peak_bw:.2f} GB/s at {peak_size} bytes")
```

## Troubleshooting

### Common Errors and Solutions

#### 1. Library File Not Found
```bash
# Error: libnccl.so.2: cannot open shared object file
# Solution:
export LD_LIBRARY_PATH=/workspace/VCCL/build/lib:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
ldd ./build/allreduce_perf
```

#### 2. GPU Communication Failure
```bash
# Error: NCCL WARN Call to connect returned Connection refused
# Solution:
# Check firewall settings
sudo ufw status
sudo firewall-cmd --list-all

# Check InfiniBand status
ibstat
ibdev2netdev

# Restart InfiniBand service
sudo systemctl restart openibd
```

#### 3. Out of Memory
```bash
# Error: CUDA out of memory
# Solution:
nvidia-smi  # Check GPU memory usage

# Reduce test scale
mpirun -np 4 ./build/allreduce_perf -b 1KB -e 10MB -f 2 -g 1

# Or enable memory optimization
export NCCL_CUMEM_ENABLE=1
```

### Debugging Tips

#### Enable Verbose Logging
```bash
# Most detailed debug information
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

# Specific subsystem debugging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH
```

---

!!! tip "Testing Recommendations"
    It's recommended to start with small-scale tests and gradually increase the number of nodes and message sizes to quickly identify issues.

!!! warning "Resource Usage"
    Large-scale tests may consume significant GPU memory and network bandwidth. It's recommended to run them during non-production hours.

!!! info "Performance Baseline"
    Establish a performance baseline database and run tests regularly to monitor VCCL performance trends.
