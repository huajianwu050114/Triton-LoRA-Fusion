import statistics
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.fused_lora_gemm import fused_lora_matmul
from kernels.lora_up_add import lora_up_add
from kernels.spatial_down_base import spatial_down_base_matmul


def benchmark(func, iters=120, warmup=20):
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


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values),
    }


def main():
    M, K, N, R = 64, 4096, 28672, 8
    dtype = torch.float16
    device = "cuda"
    repeats = 12

    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=dtype, device=device) / 10.0
    c = torch.randn((K, N), dtype=dtype, device=device) / 10.0
    a = torch.randn((K, R), dtype=dtype, device=device) / 10.0
    b = torch.randn((R, N), dtype=dtype, device=device) / 10.0

    out_ref = x @ c + (x @ a) @ b
    ep = fused_lora_matmul(x, c, x @ a, b)
    torch.testing.assert_close(out_ref, ep, atol=1e-2, rtol=1e-2)

    spatial_kwargs = {
        "layout": "interleaved",
        "base_chunk": 160,
        "down_chunk": 40,
        "block_n_base": 64,
        "block_k_base": 64,
        "block_k_down": 128,
        "block_m_down": 16,
    }
    wy = spatial_down_base_matmul(x, c, a, **spatial_kwargs)
    sp = lora_up_add(wy[0], wy[1], b)
    torch.testing.assert_close(out_ref, sp, atol=2e-2, rtol=2e-2)

    timings = {
        "baseline_serial": [],
        "epilogue_down_plus_fused_base_up": [],
        "spatial_interleaved_160_40_bn64_bk64_plus_triton_up_add": [],
        "spatial_base_down_only": [],
    }

    funcs = {
        "baseline_serial": lambda: x @ c + (x @ a) @ b,
        "epilogue_down_plus_fused_base_up": lambda: fused_lora_matmul(x, c, x @ a, b),
        "spatial_interleaved_160_40_bn64_bk64_plus_triton_up_add": lambda: (
            lambda wy: lora_up_add(wy[0], wy[1], b)
        )(spatial_down_base_matmul(x, c, a, **spatial_kwargs)),
        "spatial_base_down_only": lambda: spatial_down_base_matmul(x, c, a, **spatial_kwargs),
    }

    for i in range(repeats):
        for name, func in funcs.items():
            timings[name].append(benchmark(func))
        print(f"finished repeat {i + 1}/{repeats}")

    baseline_mean = summarize(timings["baseline_serial"])["mean"]
    print("-" * 104)
    print(f"{'Strategy':<48} {'mean(us)':>10} {'min':>10} {'max':>10} {'std':>10} {'speedup':>10}")
    print("-" * 104)
    for name, values in timings.items():
        stats = summarize(values)
        speedup = baseline_mean / stats["mean"]
        print(
            f"{name:<48} {stats['mean']:>10.2f} {stats['min']:>10.2f} "
            f"{stats['max']:>10.2f} {stats['std']:>10.2f} {speedup:>9.2f}x"
        )
    print("-" * 104)


if __name__ == "__main__":
    main()
