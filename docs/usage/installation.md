# Download and Build

## Software Requirements
| Software | Version Requirement | Installation Method |
|----------|-------------------|-------------------|
| **CUDA** | 12.7 - 12.9 | [NVIDIA Official](https://developer.nvidia.com/cuda-downloads) |
| **Python** | 3.8+ | System default or Anaconda |
| **Git** | 2.0+ | `apt install git` |

## Environment Setup

### CUDA Environment Check
```bash
# Check CUDA version
nvcc --version

# Check GPU information
nvidia-smi

# Verify CUDA installation
cat /usr/local/cuda/version.txt
```

## Download VCCL

```bash
# Clone the repository
git clone https://github.com/sii-research/VCCL.git

# Enter directory
cd VCCL

# README contains detailed build instructions
cat README.md
```

## Build VCCL Examples

### Hopper Architecture (H100/H200)
```bash
# Build command for Hopper architecture
make -j80 src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"
```

### Ampere Architecture (A100)
```bash
# Build for Ampere architecture
make -j80 src.build NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

### Common Build Parameters
| Parameter | Description | Example |
|-----------|-------------|---------|
| `-j<N>` | Parallel compilation threads | `-j80` (80 threads) |
| `NVCC_GENCODE` | GPU architecture specification | `compute_90,code=sm_90` |
| `DEBUG` | Debug version | `DEBUG=1` |
| `TRACE` | Enable tracing | `TRACE=1` |
| `VERBOSE` | Verbose output | `VERBOSE=1` |

---

!!! tip "Build Optimization Tips"
    When building on multi-core servers, it's recommended to use `-j$(nproc)` to fully utilize CPU resources, but be mindful of memory usage.

!!! warning "Version Compatibility"
    Ensure compatibility between CUDA, GCC, and VCCL versions. It's recommended to use the suggested version combinations to avoid compatibility issues.

!!! info "Performance Tuning"
    Specifying the correct `NVCC_GENCODE` parameter for your target GPU architecture will yield the best performance.
