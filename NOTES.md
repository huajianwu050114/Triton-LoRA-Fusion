# Triton LoRA Fusion Notes

## Current Task Goal

Respond to the advisor feedback about register pressure in the current Base GEMM + LoRA Up epilogue fusion path, and build a clearer experimental path for Base GEMM + LoRA Down spatial fusion. The immediate goal is not SGLang end-to-end integration; bonus work is out of scope for now.

## Advisor Feedback Summary

The current epilogue fusion keeps the Base GEMM accumulator live while computing `Y @ B`. This can occupy many registers while the block is no longer doing the heavy Base GEMM work, hurting occupancy and kernel efficiency. The advisor suggested exploring spatial parallelism: launch Base GEMM blocks and LoRA Down blocks in carefully arranged ranges so the GPU scheduler can naturally load-balance work and keep SMs busy. Persistent threads or global atomic task queues are not strictly required for the first version.

## Completed Work

- Step 1 profiling exists for Llama3-8B `gate_up_proj` shape: `M=64, K=4096, N=28672, r=8`.
- Step 2 Base GEMM Triton kernel is implemented and benchmarked near cuBLAS performance.
- Step 3 Base GEMM + LoRA Up epilogue fusion is implemented in Triton and passes fp16 correctness checks.
- Punica-style SGMV routing fusion prototype supports `seg_indptr`, `weight_indices`, `permutation`, and per-LoRA scaling in a standalone correctness test.
- After advisor feedback, a Base GEMM + LoRA Down spatial fusion prototype was added. It computes `W = X @ C` and `Y = X @ A` in one kernel using different block ranges.

## Key Files

- `01_profiling/baseline.py`: baseline PyTorch profiling for serial Base/Down/Up matmuls.
- `02_base_gemm/matmul_kernel.py`: Triton Base GEMM implementation.
- `03_fused_kernel/kernels/fused_lora_gemm.py`: Base GEMM + LoRA Up epilogue fusion.
- `03_fused_kernel/kernels/fused_punica_base_gemm.py`: Punica-style Base GEMM + SGMV Expand fusion prototype.
- `03_fused_kernel/kernels/spatial_down_base.py`: Base GEMM + LoRA Down spatial fusion prototype.
- `03_fused_kernel/tests/test_spatial_down_base.py`: correctness and microbenchmark for spatial layouts.
- `03_fused_kernel/tests/benchmark_all_strategies.py`: unified benchmark for serial, epilogue fusion, and spatial fusion strategies.
- `04_sglang_integration/linear_layer_patch.py`: illustrative SGLang patch only, not a production-ready integration.

## Latest Code Changes

- Added `three_range` layout to `spatial_down_base.py`: `[front base blocks][all down blocks][remaining base blocks]`.
- Added `benchmark_all_strategies.py` to compare:
  - serial baseline: `X @ C + (X @ A) @ B`
  - epilogue fusion: `X @ A` plus fused Base+Up
  - spatial fusion: fused Base+Down plus regular Up

## Latest Experiment Results

Ran on RTX 5090 with `M=64, K=4096, N=28672, r=8`.

Unified benchmark (`python 03_fused_kernel/tests/benchmark_all_strategies.py`):

- Serial baseline: `170.63 us`
- Epilogue fusion (`X @ A` + fused Base+Up): `158.30 us`, `1.08x`
- Best spatial result in that run: `spatial_interleaved_136_34_plus_up`, `164.05 us`, `1.04x`

Focused spatial sweep (`python 03_fused_kernel/tests/sweep_spatial_params.py`):

- Serial Base+Down: `160.54 us`
- Best fused Base+Down: `three136_k256_m32`, `153.00 us`, `1.05x`
- Best spatial full path (`fused Base+Down + regular Up`): `three136_k256_m32`, `164.80 us`, `1.03x`

Current interpretation after the first sweep: spatial fusion validates the advisor's load-balance direction for Base+Down, but `fused Base+Down + torch Up/Add` did not beat the epilogue fusion path.

Second optimization round:

- Added `down_mode="full_k"` to test a no-atomic Down path. It is correct, but did not beat the best K-split atomic path. Best full path stayed around `165.6 us`. This suggests atomics are not the main bottleneck.
- Added `lora_up_add.py`, a Triton kernel for `O = W + Y @ B`, and benchmarked it after spatial Base+Down.
- Unified benchmark result: `spatial_interleaved_136_34 + Triton Up+Add` reached `157.32 us`, slightly faster than epilogue fusion at `158.23 us` in that run.
- Focused sweep result: best `TriFull` cases were around `157.78-157.91 us`, comparable to or slightly better than the epilogue path.
- Debug timing showed standalone Triton Up+Add is slower than Torch Up/Add, but the full `spatial -> Triton Up+Add` path is faster. Likely reason: `W` is consumed immediately after being produced by spatial Base+Down, so the second Triton kernel benefits from cache locality.

Third optimization round:

- Tuned `lora_up_add.py` and changed the default tile from `BLOCK_M=32, BLOCK_N=128` to `BLOCK_M=32, BLOCK_N=256`.
- Latest unified benchmark after tuning:
  - Serial baseline: `170.45 us`
  - Epilogue fusion: `158.31 us`, `1.08x`
  - Best spatial two-stage path: `spatial_interleaved_136_34 + Triton Up+Add`, `157.13 us`, `1.08x`
- This is the first result where the advisor-driven spatial route slightly beats the original epilogue route in the same benchmark script. The margin is small, so the next report should include repeated runs or min/mean/std.

Fourth optimization round:

- Added `sweep_spatial_base_tiles.py` to tune the Base GEMM tile inside the spatial kernel.
- Best swept config: `layout=interleaved`, `base_chunk=160`, `down_chunk=40`, `BLOCK_N_BASE=64`, `BLOCK_K_BASE=64`, `BLOCK_K_DOWN=128`, `BLOCK_M_DOWN=16`.
- Tile sweep result: best full path reached `154.52 us`, with Base+Down at `149.75 us`.
- Repeated benchmark over 12 runs confirmed the result is stable:
  - Serial baseline mean: `170.63 us`
  - Epilogue fusion mean: `157.99 us`, std `0.13 us`
  - Spatial two-stage mean: `154.66 us`, std `0.10 us`
  - Spatial Base+Down only mean: `149.60 us`
- Current conclusion: the advisor-driven spatial route now clearly and stably beats the original epilogue fusion route by about `3.3 us` on this shape.

## Next Plan

1. Run `python 03_fused_kernel/tests/benchmark_all_strategies.py` and record the table.
2. Update README to report the stable best spatial result and explain why the route improved after Base tile tuning.
3. Optionally sweep nearby best configs more finely: `base_chunk=152/160/168`, `down_chunk=36/40/44`, `BLOCK_N_BASE=64`, `BLOCK_K_BASE=64`.
4. Consider measuring Nsight/Torch profiler counters for register pressure, occupancy, and cache behavior to support the advisor-facing explanation.
5. Optional research direction: explore a stronger persistent/cooperative version only if the simple range scheduling plateaus.
