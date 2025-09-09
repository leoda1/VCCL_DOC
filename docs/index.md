# VCCL 文档

欢迎使用 VCCL (Virtualized Collective Communication Library) 文档！

## 什么是 VCCL？

VCCL 是基于 NCCL 2.26.6-1 开发的虚拟化集合通信库，专为现代 AI 训练集群设计。它提供了拓扑感知的流量负载均衡、智能RNIC选择、流可视化、容错机制等先进特性，以实现更高效、更可靠的大规模分布式训练。

## 核心特性

### 🎯 拓扑感知负载均衡
支持 RING、ECMP 和 PXN 策略的优化路由，默认启用。

### 🔧 PCIe 异构与智能 RNIC 选择
适应异构 PCIe 拓扑并根据运行时条件选择最优 RNIC，默认启用。

### 📊 流可视化遥测
提供实时通信流跟踪和诊断功能。

### 🛡️ 容错机制
在节点或链路故障时确保可靠的通信，默认启用。

### ⚡ SM Free & Overlap
实现通信与计算的真正重叠，通信不占用 GPU SM 资源。

## 快速开始

### 系统要求

- **CUDA**: 12.9 (支持 CUDA 12.7 及以上版本)
- **架构**: 支持 Hopper 架构 (H100, H200)
- **网络**: InfiniBand 网络环境

### 版本信息

- **软件名称**: 20250827-vccl2.26_v0.3.1
- **基于**: nccl_2.26.6-1
- **版本**: v0.2
- **发布日期**: 2025-08-27

## 导航指南

### 📚 文档结构

- **[版本信息](version.md)**: 详细的版本信息和更新日志
- **[核心功能](features/overview.md)**: 深入了解 VCCL 的各项核心特性
- **[使用指南](usage/installation.md)**: 从下载到部署的完整使用教程
- **[环境变量参考](environment-variables.md)**: 所有可配置环境变量的详细说明

### 🚀 快速链接

- [立即下载和编译](usage/installation.md)
- [运行 NCCL 测试](usage/nccl-tests.md)
- [端到端训练配置](usage/training.md)

---

!!! tip "开始使用"
    如果您是第一次使用 VCCL，建议从 [使用指南](usage/installation.md) 开始，按照步骤完成下载、编译和基本配置。

!!! info "获取帮助"
    如有问题或需要技术支持，请参考相应的文档章节或联系开发团队。
