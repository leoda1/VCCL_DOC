# Version Information
****

### Current Version

- **Based on**: nccl_2.26.6-1 development
- **Supported CUDA Version**: 12.9 (supports CUDA 12.6 and above)

### New Features
- ✅ **Topology-Aware Traffic Load Balancing**: Supports RING, ECMP, and PXN strategies
- ✅ **Flow Telemetry Visualization**: Real-time communication flow tracking and diagnostics
- ✅ **Fault Tolerance Mechanism**: Reliable communication guarantee during node or link failures
- ✅ **SM Free & Overlap**: True overlap of communication and computation without occupying GPU SM

## Getting Version Information

### Compile-time Version Check
```bash
# Check version information at compile time
grep -r "VCCL_VERSION" build/include/
```

### Git Version Information
```bash
# View detailed version information in source directory
git log --oneline -1
git describe --tags
```

## Release Notes

### v0.3.1 Major Improvements
1. **Performance Optimization**: Compared to the original NCCL, communication efficiency on large-scale clusters has been consistently improved by 1.5% to 3.5%. Test report: https://infrawaves.feishu.cn/wiki/WYR9wTxYRixqKGkldbFcpAe4nPb
2. **Enhanced Stability**: Added fault tolerance mechanism for automatic recovery during network failures
3. **Improved Usability**: Most features are enabled by default, reducing configuration complexity
4. **Observability**: Added flow telemetry functionality for easier performance tuning and fault diagnosis

### Known Issues
- PXN strategy may require manual tuning in certain specific network topologies
- Fault tolerance functionality requires pre-specified NIC configuration
- SM Free mode may require additional tuning for certain workloads

### Future Version Planning
- **v0.3.2**: Enhanced adaptive routing algorithms
- In the discussion
---