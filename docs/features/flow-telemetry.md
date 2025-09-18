# Flow Telemetry

VCCL Flow Telemetry provides microsecond-level GPU-to-GPU point-to-point traffic measurement, helping users gain deep insights into distributed training communication patterns, identify performance bottlenecks, and perform precise optimizations.

## Feature Overview

- **Real-time monitoring**: provides microsecond-level GPU-to-GPU point-to-point traffic measurement
- **Congestion awareness**: inference of network congestion conditions
- **Developer assistance**: aids in R&D tuning and optimization


## Config

### Basic usage
```bash
# Enable telemetry
export NCCL_TELEMETRY_ENABLE=1

# Set data window size (default: 50)
export TELEMETRY_WINDOWSIZE=100

# Set log output path
export NCCL_TELEMETRY_LOG_PATH=/tmp/vccl_telemetry
```

