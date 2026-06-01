# VCCL Documentation

Welcome to VCCL Documentation! This contains the complete usage guide, features, and technical references for VCCL.

## 🚀 Quick Start

Get started with VCCL quickly, recommended in the following order:

- **[Download and Build](usage/installation.md)**
- **[Run VCCL Tests](usage/nccl-tests.md)**

## 📚 Documentation Index

- **[Version Information](version.md)**: Detailed version information and changelog
- **[Core Features & Usage](features/overview.md)**: Deep dive into VCCL's core features
    - [SM-Free AlltoAllv & Overlap Training](features/vccl_v2_setup_guide.md) *(requires VCCL tag `v2.0.0`)*
    - [SM-Free P2P & Overlap Training](features/sm-free-overlap.md) *(requires VCCL tag `v2.0.0`)*
    - [Fault Tolerance](features/fault-tolerance.md) *(requires VCCL tag `v0.3.1`)*
    - [Flow Telemetry](features/flow-telemetry.md) *(requires VCCL tag `v0.3.1`)*
- **[Environment Variables Reference](environment-variables.md)**: Detailed description of all configurable environment variables

---

!!! warning "Version Notice"
    Different features require different VCCL versions:

    - **SM-Free AlltoAllv** and **SM-Free P2P** (overlap training): please checkout tag **`v2.0.0`**
      ```bash
      git checkout v2.0.0
      ```
    - **Fault Tolerance** and **Flow Telemetry**: please checkout tag **`v0.3.1`**
      ```bash
      git checkout v0.3.1
      ```

!!! tip "Getting Started"
    If you're using VCCL for the first time, we recommend starting with [Download and Compile](usage/installation.md) to complete the download, compilation, and basic configuration steps.

!!! info "Get Help"
    If you have questions or need technical support, please refer to the relevant documentation sections or contact the development team:
    
    - [jiayanmin@infrawaves.com](mailto:jiayanmin@infrawaves.com)
    - [liuda@infrawaves.com](mailto:liuda@infrawaves.com)
    - [zhangyan@infrawaves.com](mailto:zhangyan@infrawaves.com)
    - [zhangmingjun@infrawaves.com](mailto:zhangmingjun@infrawaves.com)
    - [chenqing@infrawaves.com](mailto:chenqing@infrawaves.com)
