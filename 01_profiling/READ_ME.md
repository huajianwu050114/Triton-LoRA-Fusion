# Step 1: Baseline Profiling (RTX 5090)

## 实验环境
- **GPU**: NVIDIA GeForce RTX 5090 (170 SMs, 32GB VRAM)
- **Framework**: PyTorch 2.13.0.dev (CUDA 13.0)
- **模型场景**: Llama3-8B (`gate_up_proj`), Batch Size = 64, Rank = 8

## 基线性能数据
在组件未融合前，三个独立算子串行执行的平均耗时如下：

| 算子阶段 | 计算表达式 | 矩阵维度 (M, K, N) / r | 耗时 (us) |
| :--- | :--- | :--- | :--- |
| **Base GEMM** | $W = X \times C$ | (64, 4096, 28672) | 154.86 |
| **LoRA Down** | $Y = X \times A$ | (64, 4096, 8) | 14.12 |
| **LoRA Up** | $Z = Y \times B$ | (64, 8, 28672) | 29.99 |
| **整体串行** | $O = W + Z$ | - | **173.14** |

## 瓶颈分析 (Motivation)
1. **SM 利用率极低**：由于 LoRA Rank $r=8$ 极小，LoRA 两个算子的网格（Grid）维度极度萎缩，导致 RTX 5090 的 170 个 SM 在此期间严重不饱和，硬件资源被大量闲置。
2. **访存冗余**：Base GEMM 计算完的中间结果 $W$ 产生了一次不必要的显存（HBM）写入与随后的逐元素相加（Element-wise Add）读取，构成了典型的访存瓶颈（Memory-Bound）。