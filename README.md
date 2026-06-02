# Triton-LoRA-Fusion: High-Performance Operator Fusion for Multi-LoRA Serving

![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13.0.dev-red.svg)
![Triton](https://img.shields.io/badge/Triton-Nightly-blue.svg)
![Hardware](https://img.shields.io/badge/Hardware-RTX_5090_(Blackwell)-black.svg)

## Project Overview

本项目针对 Llama3-8B `gate_up_proj` 的 LoRA 推理场景进行 Triton 算子融合优化。目标 shape 为 `M=64, K=4096, N=28672, r=8`，计算链路为：

```text
Y = X @ A          # LoRA Down, [64, 4096] x [4096, 8]
W = X @ C          # Base GEMM, [64, 4096] x [4096, 28672]
O = W + Y @ B      # LoRA Up/Add, [64, 8] x [8, 28672]
```

初版方案实现了 `Base GEMM + LoRA Up` 的 epilogue fusion，能够避免 `W` 的中间读写，并将串行基线从约 `170 us` 降到约 `158 us`。在老师反馈之后，我们进一步验证了 epilogue fusion 的潜在问题：Base GEMM tile 算完后需要让 accumulator 长时间占据 register，再继续计算 `Y @ B`，这可能增加 register pressure 并降低 kernel 整体效率。

因此，最终优化路线转向 **Spatial Base+Down + Triton Up+Add**：

1. 在第一个 Triton kernel 中，将 Base GEMM blocks 和 LoRA Down blocks 按 range/interleaved 方式混合发射，利用 GPU scheduler 的自动 load balance。
2. 在第二个轻量 Triton kernel 中执行 `O = W + Y @ B`，利用紧邻消费 `W` 带来的 cache locality。
3. 通过 spatial range、Base GEMM tile 和 Up+Add tile 调参，最终稳定超过初版 epilogue fusion。

---

## Core Technical Highlights

### 1. Baseline: Base GEMM + LoRA Up Epilogue Fusion

`03_fused_kernel/kernels/fused_lora_gemm.py` 实现了初版 epilogue fusion：每个 Triton program 先计算一个 Base GEMM tile，将 FP32 accumulator 保留在寄存器中，然后加载 `Y` 和 LoRA Up 权重 `B`，直接累加 `Y @ B` 后一次性写回。

该方案的优势是减少 `W` 的 HBM 写回/读回，但缺点是 accumulator 在 LoRA Up 阶段继续占据大量 register。这个问题正是老师反馈后继续优化的切入点。

### 2. Advisor-Driven Spatial Fusion

`03_fused_kernel/kernels/spatial_down_base.py` 实现了 `Base GEMM + LoRA Down` 的 spatial fusion prototype。不同 blocks 执行不同任务：

```text
Base blocks: W = X @ C
Down blocks: Y = X @ A
```

当前支持三类布局：

```text
contiguous : [all base][all down]
interleaved: [base chunk][down chunk] repeated
three_range: [front base][all down][remaining base]
```

最终最优配置为：

```text
layout = interleaved
base_chunk = 160
down_chunk = 40
BLOCK_N_BASE = 64
BLOCK_K_BASE = 64
BLOCK_K_DOWN = 128
BLOCK_M_DOWN = 16
```

这个配置让 `Base+Down` 从串行约 `160.6 us` 降到约 `149.6 us`。

### 3. Two-Stage Triton Up+Add

`03_fused_kernel/kernels/lora_up_add.py` 实现第二阶段：

```text
O = W + Y @ B
```

单独测这个 kernel 并不比 PyTorch `Y @ B + add` 更快，但紧跟 spatial Base+Down 执行时，整体链路更快。原因很可能是 `W` 刚由第一个 Triton kernel 写出，第二个 Triton kernel 立刻消费 `W`，获得了更好的 cache locality。

最终 Up+Add tile 采用：

```text
BLOCK_M = 32
BLOCK_N = 256
BLOCK_R_PAD = 16
```

### 4. Punica-Style Dynamic Routing Prototype

`03_fused_kernel/kernels/fused_punica_base_gemm.py` 保留了 Punica/SGLang 风格的 SGMV routing 支持，包括 `seg_indptr`、`weight_indices`、`permutation`、`scalings` 等参数。该部分通过 standalone correctness test 验证，用于展示多 LoRA 动态路由能力；完整 SGLang 端到端集成属于 bonus，不在当前交付范围内。

---

## Performance Evaluation

测试环境：NVIDIA GeForce RTX 5090, CUDA 13.0, shape `M=64, K=4096, N=28672, r=8`。

### Main Benchmark

命令：

```bash
python 03_fused_kernel/tests/benchmark_all_strategies.py
```

最新单轮结果：

| Strategy | Latency | Speedup vs Serial |
| :--- | ---: | ---: |
| Serial baseline: `X@C + (X@A)@B` | 170.57 us | 1.00x |
| Epilogue fusion: `X@A + fused(Base+Up)` | 158.43 us | 1.08x |
| Spatial two-stage: `fused(Base+Down) + Triton Up+Add` | **154.71 us** | **1.10x** |

### Repeated Benchmark

命令：

```bash
python 03_fused_kernel/tests/repeat_best_strategies.py
```

12 轮重复测试结果：

| Strategy | Mean | Min | Max | Std | Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Serial baseline | 170.63 us | 170.40 us | 171.01 us | 0.15 us | 1.00x |
| Epilogue fusion | 157.99 us | 157.73 us | 158.23 us | 0.13 us | 1.08x |
| Spatial two-stage | **154.66 us** | **154.52 us** | **154.86 us** | **0.10 us** | **1.10x** |
| Spatial Base+Down only | 149.60 us | 149.49 us | 149.78 us | 0.08 us | - |

结论：赵老师指导后的 spatial route 不只是方向可行，经过 Base tile 和 Up+Add tile 调参后，已经稳定超过初版 epilogue fusion，平均快约 `3.3 us`。

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
├── 01_profiling/
│   ├── baseline.py
│   └── READ_ME.md
├── 02_base_gemm/
│   ├── matmul_kernel.py
│   └── test_base_gemm.py
├── 03_fused_kernel/
│   ├── kernels/
│   │   ├── fused_lora_gemm.py           # Base + LoRA Up epilogue fusion
│   │   ├── fused_punica_base_gemm.py    # Punica-style dynamic routing prototype
│   │   ├── lora_up_add.py               # Second-stage O = W + Y @ B kernel
│   │   └── spatial_down_base.py         # Spatial Base + LoRA Down fusion
│   ├── tests/
│   │   ├── benchmark_all_strategies.py  # Main one-shot comparison
│   │   ├── repeat_best_strategies.py    # Main repeated benchmark
│   │   ├── sweep_spatial_base_tiles.py  # Base tile sweep for spatial route
│   │   ├── sweep_spatial_params.py      # Spatial layout/down sweep
│   │   ├── test_fused_kernel.py
│   │   ├── test_lora_up_add.py
│   │   ├── test_punica_fusion.py
│   │   └── test_spatial_down_base.py
│   └── docs/
├── 04_sglang_integration/
│   └── linear_layer_patch.py            # Illustrative patch only, not production integration
├── NOTES.md                             # Collaboration notes and experiment log
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Baseline profiling

```bash
python 01_profiling/baseline.py
```

### 2. Verify Base GEMM

```bash
python 02_base_gemm/test_base_gemm.py
```

### 3. Verify initial epilogue fusion

```bash
python 03_fused_kernel/tests/test_fused_kernel.py
```

### 4. Verify spatial fusion and Up+Add kernels

```bash
python 03_fused_kernel/tests/test_spatial_down_base.py
python 03_fused_kernel/tests/test_lora_up_add.py
```

### 5. Run final benchmark

```bash
python 03_fused_kernel/tests/benchmark_all_strategies.py
python 03_fused_kernel/tests/repeat_best_strategies.py
```

### 6. Reproduce tuning experiments

```bash
python 03_fused_kernel/tests/sweep_spatial_params.py
python 03_fused_kernel/tests/sweep_spatial_base_tiles.py
python 03_fused_kernel/tests/tune_lora_up_add.py
```

### 7. Verify Punica-style routing prototype

```bash
python 03_fused_kernel/tests/test_punica_fusion.py
```

## Current Conclusion

Bonus 之外的主任务已经完成：

- 完成 Llama3-8B `gate_up_proj` 目标 shape 的 baseline profiling。
- 实现接近 cuBLAS 性能的 Triton Base GEMM。
- 实现并验证初版 `Base + LoRA Up` epilogue fusion。
- 根据老师反馈，实现 `Base + LoRA Down` spatial fusion，并通过 range/tile tuning 证明该路线稳定超过 epilogue fusion。
- 保留 Punica-style dynamic routing prototype，展示多 LoRA routing 支持能力。

SGLang 端到端接入属于 bonus，目前仅提供 illustrative patch，不作为当前完成范围。

## Future Work

- 对最佳 spatial 配置做更细粒度 sweep：`base_chunk=152/160/168`，`down_chunk=36/40/44`，`BLOCK_N_BASE=64`，`BLOCK_K_BASE=64`。
- 使用 Nsight Systems / Nsight Compute 采集 occupancy、register pressure、L2 hit rate 等指标，为 spatial route 的解释提供硬件计数器证据。
- 将最佳 kernel 接入 SGLang LoRA backend 做端到端验证。
