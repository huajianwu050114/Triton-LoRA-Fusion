# Step 3: SGMV Epilogue Fusion (Punica + Base GEMM)

## 设计目标
严格按照工业级标准，从 SGLang 框架中提取 Punica 核心的 SGMV（Segmented GEMV）升维算子，并将其与主干矩阵乘（Base GEMM）在 Triton 层面进行深度融合（Epilogue Fusion），彻底消除中间张量的 HBM（显存）读写。

## 核心技术突破

### 1. 兼容动态分段路由 (Segmented Routing)
真实的线上推理包含同 Batch 内多 LoRA 混合的复杂场景。我们在融合算子中保留了 SGLang 的 `seg_indptr`、`weight_indices` 和 `permutation` 等物理指针寻址逻辑。这使得我们的算子不仅是一个 Dense GEMM，更是一个支持**多 LoRA 动态分段调度**的工业级算子。

### 2. 寄存器级累加与 Zero-Padding
* **Epilogue Fusion**: Thread Block 在完成 Base GEMM 的局部 Tile 计算后，结果保留在 SRAM 中，直接依据路由表加载对应的 LoRA B 权重进行升维计算并累加。
* **硬件级欺骗**: 针对 LoRA 极小的秩（$r=8$），利用常量填充（Zero-Padding）满足了 Blackwell 架构 Tensor Core 对于 $K \ge 16$ 的 MMA 指令硬件强制要求。

## 性能与正确性验证
通过模拟 `M=64` 且包含多个不同 LoRA 请求的 Batch，融合算子在 RTX 5090 (CUDA 13.0) 上成功通过了严格的 fp16 精度验证：
* ✅ `torch.testing.assert_close(O_ref, O_tri)`: **PASSED**