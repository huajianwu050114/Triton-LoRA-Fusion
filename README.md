# Triton-LoRA-Fusion: High-Performance Operator Fusion for Multi-LoRA Serving

![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0.dev-red.svg)
![Triton](https://img.shields.io/badge/Triton-Nightly-blue.svg)
![Hardware](https://img.shields.io/badge/Hardware-RTX_5090_(Blackwell)-black.svg)

## 📖 Project Overview

在现阶段的大型语言模型（LLM）系统中，多 LoRA 混合部署（Multi-LoRA Serving）面临着严重的性能瓶颈。特别是 LoRA 计算涉及的低秩矩阵（如 $r=8$）运算，极易导致 GPU 的 SM（流多处理器）利用率低下，并引发大量中间变量的显存（HBM）读写，形成典型的 **Memory Wall（访存墙）**。

本项目针对 Llama3-8B 在 `gate_up_proj` 层的计算场景（$M=64, K=4096, N=28672$），基于 OpenAI Triton 实现了一个工业级的端到端算子融合方案。通过将 **主干矩阵乘（Base GEMM）** 与 **升维矩阵乘（LoRA Expand）** 进行深度融合，本项目彻底消除了中间张量写回 HBM 的开销，显著提升了推理吞吐量。

---

## ✨ Core Technical Highlights

### 1. 架构设计抉择：Epilogue Fusion vs. Spatial Partitioning
针对“降维算子与主干算子融合”以及“升维算子与主干算子融合”两条路径，本项目在架构设计上进行了深入的权衡：
* **关于 Spatial Partitioning (空间并行) 的探讨：** 理论上，可以通过 Launch 满载 SM 的 Block 数量，并依赖调度器实现两个算子的共驻（例如前一半 Block 执行算子 1，后一半执行算子 3）。然而，NVIDIA 的底层 `pid` 调度并非严格的 Round-Robin 映射，强行依赖 PID 极易导致复杂的负载不均（Load Imbalance）。若要实现完美的空间并行，必须引入基于 Global Atomic Counter 的 Persistent Threads 范式。
* **我们的解法：Epilogue Fusion (尾部融合)：** 考虑到 LoRA 计算的实际依赖关系，本项目最终采用更高效的 **访存级融合** 策略。Thread Block 在 SRAM（寄存器）中完成 Base GEMM 的局部 Tile 计算后，**不将其写回 HBM**，而是原地加载 LoRA 的局部输出 $Y$ 和权重 $B$，直接在处于 FP32 精度的累加器中进行点积累加。这带来了 $O(1)$ 的额外显存开销，并实现了物理带宽收益的最大化。

### 2. 硬件级优化：Zero-Padding for Blackwell Tensor Cores
在最新的 Blackwell 架构（如 RTX 5090）上，底层的 MMA（矩阵乘加）指令强制要求输入维度 $K \ge 16$。针对 LoRA 常见的极小秩（$r=8$），本项目在 Triton 寄存器层面引入了静态常量填充（Zero-Padding）。通过将 $8 \sim 15$ 的维度 Mask 强制填充为 0.0，成功骗过编译器发射最高效的硬件级 Tensor Core 指令，且保证了数学上的绝对等价。

### 3. 工业级动态路由：Punica SGMV 深度集成
真实场景下的 Batch 往往包含多个不同的 LoRA 请求。本项目不仅是一个 Dense GEMM，更从 SGLang 框架中提取了 Punica 核心的 SGMV（Segmented GEMV）逻辑。融合算子完整支持 `seg_indptr`、`weight_indices` 和 `permutation` 等物理指针寻址，具备直接接入 vLLM/SGLang 生产环境的能力。

---

## 📊 Performance Evaluation

在 NVIDIA GeForce RTX 5090 (170 SMs, Blackwell) 上进行严苛的端到端 Benchmarking：

| 指标 | 原生串行基线 (cuBLAS + 分离计算) | Triton 融合策略 (Fused Kernel) |
| :--- | :--- | :--- |
| **单次执行时延 ($M=64$)** | 170.70 us | **158.95 us** |
| **绝对时间节省** | - | **11.75 us** / layer |
| **数值精度对齐** | 基准 (Reference) | ✅ **PASSED** (fp16 range) |

**💡 宏观系统级收益分析：**
表面上的 11.75 us 节省实际上已经**逼近了物理带宽极限**（FP16 下一读一写 3.67MB 的中间变量 $W$，在当前频率下的纯理论带宽耗时约 5us，加上系统调用开销，11.75us 代表彻底抹平了访存墙）。
在 Llama3-8B（32 层）的实际生成任务中，假设生成 1024 个 Token，此融合算子单次请求即可为端到端（End-to-End）推理省下近 **385 毫秒 (ms)** 的纯访存延迟，极大优化了 TPOT (Time Per Output Token)。

---

## 🛠️ Environment Setup (Strict Requirements)

本项目大量使用了最新的硬件特性，为了保证性能可被完美复现，请严格按照以下环境基线进行配置：

* **Hardware:** NVIDIA GeForce RTX 5090 (Blackwell Architecture, Compute Capability 12.0)
* **OS:** Ubuntu 22.04 / 24.04
* **CUDA Toolkit:** 13.0+ (Strictly required for RTX 5090 support)

### Step-by-step Installation

**1. Create Conda Environment**
```bash
conda create -n triton-lora python=3.10 -y
conda activate triton-lora
```
**2. Install Core Deep Learning Stack** (Nightly Required)

由于 RTX 5090 架构极新，必须使用适配 CUDA 13.0 的 PyTorch Nightly 版本：
```bash
# Install PyTorch built with CUDA 13.0
pip3 install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu130](https://download.pytorch.org/whl/nightly/cu130)

# Install Triton
pip install -U --pre triton
```
**3. Install Auxiliary Tools**
```bash 
pip install -r requirements.txt
```
## Project Structure
```plaintext
.
├── 01_profiling/          # 原始算子 Profiling 与基线建立
│   ├── baseline.py        # PyTorch Nsight Profiler 脚本
│   └── READ_ME.md         # 访存瓶颈与 SM 闲置分析报告
├── 02_base_gemm/          # 高性能主干矩阵乘实现
│   ├── matmul_kernel.py   # 支持极端长宽比 (64x28672) 的 Triton GEMM
│   └── test_base_gemm.py  # 精度与 cuBLAS 对齐验证
├── 03_fused_kernel/       # 核心融合算子代码库
│   ├── kernels/                  # Triton kernel 实现
│   │   ├── fused_lora_gemm.py
│   │   ├── fused_punica_base_gemm.py
│   │   └── spatial_down_base.py
│   ├── tests/                    # 正确性与性能测试脚本
│   │   ├── test_fused_kernel.py
│   │   ├── test_punica_fusion.py
│   │   └── test_spatial_down_base.py
│   └── docs/                     # Step 3 相关说明文档
├── requirements.txt       # 环境依赖
└── README.md              # 项目主文档
```

## Quick Start
**step1: 复现基线分析 (Profiling):**
```bash 
python 01_profiling/baseline.py
```
**step2: 验证Base GEMM性能**
```bash
python 02_base_gemm/test_base_gemm.py
```
**step3：升维融合性能对比**
```bash 
python 03_fused_kernel/tests/test_fused_kernel.py
```
**step4: 验证 Punica SGMV 动态路由功能:**
```bash 
python 03_fused_kernel/tests/test_punica_fusion.py
```
**step5: 验证 Spatial Down+Base 原型:**
```bash
python 03_fused_kernel/tests/test_spatial_down_base.py
```
## Future Work
后续工作包括将 fused_sgmv_base_expand_kernel 直接集成到 SGLang 的 lora/backend/triton_ops 路径下，以实现系统级、端到端的性能指标提升。