import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.spatial_down_base import spatial_down_base_matmul


def benchmark(func, iters=100):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        func()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def run_spatial_case(name, x, c, a, w_ref, y_ref, **kwargs):
    w_tri, y_tri = spatial_down_base_matmul(x, c, a, **kwargs)
    torch.testing.assert_close(w_ref, w_tri, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_ref, y_tri, atol=2e-2, rtol=2e-2)

    for _ in range(10):
        _ = spatial_down_base_matmul(x, c, a, **kwargs)
    torch.cuda.synchronize()

    latency = benchmark(lambda: spatial_down_base_matmul(x, c, a, **kwargs))
    return name, latency


def main():
    M, K, N, R = 64, 4096, 28672, 8
    device = "cuda"

    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.float16, device=device) / 10.0
    c = torch.randn((K, N), dtype=torch.float16, device=device) / 10.0
    a = torch.randn((K, R), dtype=torch.float16, device=device) / 10.0

    w_ref = x @ c
    y_ref = (x @ a).float()

    for _ in range(10):
        _ = x @ c
        _ = x @ a
    torch.cuda.synchronize()
    serial_time = benchmark(lambda: (x @ c, x @ a))

    cases = [
        ("contiguous", {"layout": "contiguous"}),
        ("interleaved_40_10", {"layout": "interleaved", "base_chunk": 40, "down_chunk": 10}),
        ("interleaved_80_20", {"layout": "interleaved", "base_chunk": 80, "down_chunk": 20}),
        ("interleaved_160_40", {"layout": "interleaved", "base_chunk": 160, "down_chunk": 40}),
    ]

    results = []
    for name, kwargs in cases:
        results.append(run_spatial_case(name, x, c, a, w_ref, y_ref, **kwargs))

    print("Spatial Down+Base correctness PASSED")
    print("-" * 64)
    print(f"{'Case':<24} {'Latency(us)':>12} {'Speedup vs serial':>18}")
    print("-" * 64)
    print(f"{'serial_base_down':<24} {serial_time:>12.2f} {1.0:>18.2f}x")
    for name, latency in results:
        print(f"{name:<24} {latency:>12.2f} {serial_time / latency:>18.2f}x")
    print("-" * 64)


if __name__ == "__main__":
    main()
