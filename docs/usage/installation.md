# 下载与编译

本页面详细介绍如何下载、编译和安装 VCCL，包括系统要求、依赖项安装和编译过程的完整指南。

## 系统要求

### 硬件要求
| 组件 | 最低要求 | 推荐配置 | 说明 |
|------|----------|----------|------|
| **GPU** | NVIDIA Volta 架构 | Hopper 架构 (H100/H200) | 支持 CUDA 计算能力 7.0+ |
| **CPU** | Intel x86_64 | AMD EPYC 或 Intel Xeon | 多核处理器，支持 AVX2 |
| **内存** | 32 GB | 64 GB+ | 大模型训练需要更多内存 |
| **网络** | InfiniBand FDR | InfiniBand HDR/NDR | 高带宽低延迟网络 |
| **存储** | 100 GB 可用空间 | NVMe SSD | 快速存储提升编译速度 |

### 软件要求
| 软件 | 版本要求 | 安装方式 |
|------|----------|----------|
| **CUDA** | 12.7 - 12.9 | [NVIDIA官网](https://developer.nvidia.com/cuda-downloads) |
| **GCC** | 7.5 - 11.x | `apt install gcc g++` 或 `yum install gcc gcc-c++` |
| **CMake** | 3.18+ | `apt install cmake` 或编译安装 |
| **Python** | 3.8+ | 系统自带或 Anaconda |
| **Git** | 2.0+ | `apt install git` |

## 下载 VCCL

### 方式一：下载 Tarball (推荐)
```bash
# 下载预编译包
wget https://releases.example.com/vccl/vccl_2.26.6-1.tar.gz

# 解压缩
tar -xzvf vccl_2.26.6-1.tar.gz

# 进入目录
cd vccl_2.26.6-1
```

### 方式二：Git 克隆
```bash
# 克隆代码仓库
git clone git@183.207.7.174:moon/vccl_2.26.6-1.git

# 进入目录
cd vccl_2.26.6-1

# 查看版本信息
git log --oneline -5
```

### 方式三：从 GitHub 镜像下载
```bash
# 如果有 GitHub 镜像仓库
git clone https://github.com/your-org/vccl.git
cd vccl
```

## 环境准备

### CUDA 环境检查
```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 信息
nvidia-smi

# 验证 CUDA 安装
cat /usr/local/cuda/version.txt
```

### 依赖项安装

#### Ubuntu/Debian 系统
```bash
# 更新包管理器
sudo apt update

# 安装基本编译工具
sudo apt install -y build-essential cmake git

# 安装 CUDA (如果未安装)
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_545.23.06_linux.run
sudo sh cuda_12.9.0_545.23.06_linux.run

# 安装 InfiniBand 驱动和工具
sudo apt install -y ibverbs-utils rdma-core libibverbs-dev
```

#### CentOS/RHEL 系统
```bash
# 安装开发工具
sudo yum groupinstall -y \"Development Tools\"
sudo yum install -y cmake git

# 安装 CUDA
sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo yum install -y cuda

# 安装 InfiniBand 驱动
sudo yum install -y rdma-core libibverbs-devel
```

### 环境变量设置
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 应用环境变量
source ~/.bashrc
```

## 编译 VCCL

### 查看编译说明
```bash
# 进入 VCCL 目录
cd vccl_2.26.6-1

# 查看详细编译说明
cat README.md

# 查看 Makefile 选项
make help
```

### 基本编译

#### Hopper 架构 (H100/H200)
```bash
# 针对 Hopper 架构的编译命令
make -j80 src.build NVCC_GENCODE=\"-gencode=arch=compute_90,code=sm_90\"
```

#### Ampere 架构 (A100)
```bash
# 针对 Ampere 架构的编译
make -j$(nproc) src.build NVCC_GENCODE=\"-gencode=arch=compute_80,code=sm_80\"
```

#### 多架构支持
```bash
# 编译支持多种架构的版本
make -j$(nproc) src.build NVCC_GENCODE=\"\\
    -gencode=arch=compute_70,code=sm_70 \\
    -gencode=arch=compute_80,code=sm_80 \\
    -gencode=arch=compute_90,code=sm_90\"
```

### 编译选项说明

#### 常用编译参数
| 参数 | 说明 | 示例 |
|------|------|------|
| `-j<N>` | 并行编译线程数 | `-j80` (80线程) |
| `NVCC_GENCODE` | GPU 架构指定 | `compute_90,code=sm_90` |
| `DEBUG` | 调试版本 | `DEBUG=1` |
| `TRACE` | 启用跟踪 | `TRACE=1` |
| `VERBOSE` | 详细输出 | `VERBOSE=1` |

#### 高级编译选项
```bash
# 调试版本编译
make -j$(nproc) src.build DEBUG=1 TRACE=1

# 启用所有优化
make -j$(nproc) src.build MAXOPTIMIZE=1

# 静态链接版本
make -j$(nproc) src.build BUILDSTATIC=1

# 指定安装路径
make -j$(nproc) src.build PREFIX=/opt/vccl
```

### 编译过程监控
```bash
# 查看编译进度
make -j$(nproc) src.build VERBOSE=1 2>&1 | tee build.log

# 监控编译资源使用
htop

# 监控磁盘使用
df -h
du -sh build/
```

## 安装

### 默认安装
```bash
# 安装到默认位置 (/usr/local)
sudo make install

# 或者指定安装路径
make install PREFIX=/opt/vccl
```

### 检查安装结果
```bash
# 检查安装的文件
ls -la /usr/local/lib/libnccl*
ls -la /usr/local/include/nccl*

# 或者检查自定义路径
ls -la /opt/vccl/lib/
ls -la /opt/vccl/include/
```

### 环境变量配置
```bash
# 添加 VCCL 库路径到环境变量
export LD_LIBRARY_PATH=/opt/vccl/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/opt/vccl/lib:$LIBRARY_PATH
export C_INCLUDE_PATH=/opt/vccl/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/opt/vccl/include:$CPLUS_INCLUDE_PATH

# 永久保存到配置文件
echo 'export LD_LIBRARY_PATH=/opt/vccl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 验证安装

### 基本功能测试
```bash
# 编译并运行简单测试
cd vccl_2.26.6-1
make -j$(nproc) test

# 运行基本的 NCCL 测试
export LD_LIBRARY_PATH=/opt/vccl/lib:$LD_LIBRARY_PATH
mpirun -np 2 --allow-run-as-root ./build/test/single_process_test
```

### 版本信息验证
```bash
# 检查 VCCL 版本
strings /opt/vccl/lib/libnccl.so | grep VCCL

# 运行版本检查程序
./build/version_check

# 查看编译信息
./build/build_info
```

### 网络连接测试
```bash
# 测试 InfiniBand 连接
ibstat
ibdev2netdev

# 简单的双节点测试
mpirun -np 2 \\
    --host node1:1,node2:1 \\
    --allow-run-as-root \\
    -x LD_LIBRARY_PATH=/opt/vccl/lib:$LD_LIBRARY_PATH \\
    ./build/test/allreduce_test
```

## 常见问题和解决方案

### 编译问题

#### 问题 1: CUDA 版本不兼容
```bash
# 错误信息
error: #error -- unsupported GNU version! gcc versions later than 11 are not supported!

# 解决方案：安装兼容的 GCC 版本
sudo apt install gcc-9 g++-9
export CC=gcc-9
export CXX=g++-9
make clean && make -j$(nproc) src.build
```

#### 问题 2: 内存不足
```bash
# 错误信息
virtual memory exhausted: Cannot allocate memory

# 解决方案：减少并行编译线程
make -j8 src.build  # 减少到 8 个线程

# 或者增加交换空间
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 问题 3: 依赖项缺失
```bash
# 错误信息
fatal error: infiniband/verbs.h: No such file or directory

# 解决方案：安装 InfiniBand 开发包
sudo apt install libibverbs-dev librdmacm-dev
# 或者在 CentOS
sudo yum install rdma-core-devel
```

### 运行时问题

#### 问题 1: 库文件找不到
```bash
# 错误信息
error while loading shared libraries: libnccl.so.2: cannot open shared object file

# 解决方案：设置正确的库路径
export LD_LIBRARY_PATH=/opt/vccl/lib:$LD_LIBRARY_PATH
ldconfig /opt/vccl/lib
```

#### 问题 2: GPU 访问权限
```bash
# 错误信息
CUDA error: no CUDA-capable device is detected

# 解决方案：检查 GPU 驱动和权限
nvidia-smi
sudo chmod 666 /dev/nvidia*
```

## 性能优化编译

### 针对特定硬件的优化
```bash
# 针对特定 CPU 架构优化
export CFLAGS=\"-march=native -mtune=native\"
export CXXFLAGS=\"-march=native -mtune=native\"

# 针对特定 GPU 优化
make -j$(nproc) src.build \\
    NVCC_GENCODE=\"-gencode=arch=compute_90,code=sm_90\" \\
    MAXOPTIMIZE=1
```

### 编译缓存优化
```bash
# 安装 ccache 加速重复编译
sudo apt install ccache
export PATH=/usr/lib/ccache:$PATH

# 设置缓存大小
ccache -M 10G

# 查看缓存统计
ccache -s
```

### 并行编译优化
```bash
# 根据系统资源调整并行度
NCPUS=$(nproc)
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')

# 保守估计：每个编译任务需要 2GB 内存
PARALLEL_JOBS=$((MEMORY_GB / 2))
if [ $PARALLEL_JOBS -gt $NCPUS ]; then
    PARALLEL_JOBS=$NCPUS
fi

echo \"Using $PARALLEL_JOBS parallel jobs\"
make -j$PARALLEL_JOBS src.build
```

## 容器化部署

### Docker 构建
```dockerfile
# Dockerfile 示例
FROM nvidia/cuda:12.9-devel-ubuntu20.04

# 安装依赖
RUN apt-get update && apt-get install -y \\
    build-essential cmake git \\
    libibverbs-dev librdmacm-dev \\
    && rm -rf /var/lib/apt/lists/*

# 复制源码并编译
COPY vccl_2.26.6-1 /opt/vccl-src
WORKDIR /opt/vccl-src
RUN make -j$(nproc) src.build NVCC_GENCODE=\"-gencode=arch=compute_90,code=sm_90\"
RUN make install PREFIX=/opt/vccl

# 设置环境变量
ENV LD_LIBRARY_PATH=/opt/vccl/lib:$LD_LIBRARY_PATH
```

### 构建和使用
```bash
# 构建镜像
docker build -t vccl:latest .

# 运行容器
docker run --gpus all --ipc=host --network=host \\
    -v /dev/infiniband:/dev/infiniband \\
    vccl:latest
```

---

!!! tip \"编译优化建议\"
    在多核服务器上编译时，建议使用 `-j$(nproc)` 充分利用 CPU 资源，但要注意内存使用情况。

!!! warning \"版本兼容性\"
    确保 CUDA、GCC 和 VCCL 版本的兼容性，建议使用推荐的版本组合以避免兼容性问题。

!!! info \"性能调优\"
    针对目标 GPU 架构指定正确的 `NVCC_GENCODE` 参数可以获得最佳性能。
