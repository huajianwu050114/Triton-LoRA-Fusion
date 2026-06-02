import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.fused_lora_gemm import fused_lora_matmul
from kernels.lora_up_add import lora_up_add
from kernels.spatial_down_base import spatial_down_base_matmul


def benchmark(func, iters=100, warmup=10):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def check_close(name, ref, out, atol=1e-2, rtol=1e-2):
    try:
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)
    except Exception as exc:
        print(f"{name}: FAILED")
        print(exc)
        raise
    print(f"{name}: PASSED")


def main():
    M, K, N, R = 64, 4096, 28672, 8
    device = "cuda"
    dtype = torch.float16
    iters = 100

    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=dtype, device=device) / 10.0
    c = torch.randn((K, N), dtype=dtype, device=device) / 10.0
    a = torch.randn((K, R), dtype=dtype, device=device) / 10.0
    b = torch.randn((R, N), dtype=dtype, device=device) / 10.0

    print(f"Shape: M={M}, K={K}, N={N}, R={R}, dtype={dtype}")
    print("Computing references...")
    y_ref = x @ a
    out_ref = x @ c + y_ref @ b

    y_ep = x @ a
    out_ep = fused_lora_matmul(x, c, y_ep, b)
    check_close("epilogue_fusion_correctness", out_ref, out_ep)

    spatial_cases = [
        ("spatial_contiguous", {"layout": "contiguous"}),
        ("spatial_interleaved_80_20", {"layout": "interleaved", "base_chunk": 80, "down_chunk": 20}),
        ("spatial_interleaved_136_34", {"layout": "interleaved", "base_chunk": 136, "down_chunk": 34}),
        ("spatial_interleaved_160_40_bn64_bk64", {"layout": "interleaved", "base_chunk": 160, "down_chunk": 40, "block_n_base": 64, "block_k_base": 64, "block_k_down": 128}),
        ("spatial_three_range_128", {"layout": "three_range", "base_chunk": 128}),
        ("spatial_three_range_136", {"layout": "three_range", "base_chunk": 136}),
        ("spatial_three_range_160", {"layout": "three_range", "base_chunk": 160}),
    ]

    spatial_outputs = {}
    for name, kwargs in spatial_cases:
        w_sp, y_sp = spatial_down_base_matmul(x, c, a, **kwargs)
        out_sp = w_sp + y_sp.to(dtype) @ b
        check_close(name + "_correctness", out_ref, out_sp, atol=2e-2, rtol=2e-2)
        out_sp_tri = lora_up_add(w_sp, y_sp, b)
        check_close(name + "_triton_up_add_correctness", out_ref, out_sp_tri, atol=2e-2, rtol=2e-2)
        spatial_outputs[name] = kwargs

    print("\nBenchmarking...")
    baseline_time = benchmark(lambda: x @ c + (x @ a) @ b, iters=iters)
    epilogue_time = benchmark(lambda: fused_lora_matmul(x, c, x @ a, b), iters=iters)

    rows = [
        ("baseline_serial", baseline_time, 1.0),
        ("epilogue_down_plus_fused_base_up", epilogue_time, baseline_time / epilogue_time),
    ]

    for name, kwargs in spatial_outputs.items():
        latency = benchmark(
            lambda kwargs=kwargs: (
                lambda wy: wy[0] + wy[1].to(dtype) @ b
            )(spatial_down_base_matmul(x, c, a, **kwargs)),
            iters=iters,
        )
        rows.append((name + "_plus_torch_up", latency, baseline_time / latency))

        triton_up_latency = benchmark(
            lambda kwargs=kwargs: (
                lambda wy: lora_up_add(wy[0], wy[1], b)
            )(spatial_down_base_matmul(x, c, a, **kwargs)),
            iters=iters,
        )
        rows.append((name + "_plus_triton_up_add", triton_up_latency, baseline_time / triton_up_latency))

    print("-" * 86)
    print(f"{'Strategy':<44} {'Latency(us)':>14} {'Speedup vs baseline':>20}")
    print("-" * 86)
    for name, latency, speedup in rows:
        print(f"{name:<44} {latency:>14.2f} {speedup:>20.2f}x")
    print("-" * 86)


if __name__ == "__main__":
    main()
