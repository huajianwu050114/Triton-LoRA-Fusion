import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.lora_up_add import lora_up_add
from kernels.spatial_down_base import spatial_down_base_matmul


def benchmark(func, iters=160, warmup=20):
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


def main():
    M, K, N, R = 64, 4096, 28672, 8
    dtype = torch.float16
    device = "cuda"
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=dtype, device=device) / 10.0
    c = torch.randn((K, N), dtype=dtype, device=device) / 10.0
    a = torch.randn((K, R), dtype=dtype, device=device) / 10.0
    b = torch.randn((R, N), dtype=dtype, device=device) / 10.0

    spatial_kwargs = {"layout": "interleaved", "base_chunk": 136, "down_chunk": 34, "block_k_down": 128, "block_m_down": 16}
    w, y = spatial_down_base_matmul(x, c, a, **spatial_kwargs)
    ref = w + y.to(dtype) @ b

    cases = []
    for block_m in (16, 32, 64):
        for block_n in (64, 128, 256):
            cases.append((block_m, block_n))

    rows = []
    for block_m, block_n in cases:
        out = lora_up_add(w, y, b, block_m=block_m, block_n=block_n)
        torch.testing.assert_close(ref, out, atol=2e-2, rtol=2e-2)
        up_time = benchmark(lambda bm=block_m, bn=block_n: lora_up_add(w, y, b, block_m=bm, block_n=bn))
        full_time = benchmark(
            lambda bm=block_m, bn=block_n: (
                lambda wy: lora_up_add(wy[0], wy[1], b, block_m=bm, block_n=bn)
            )(spatial_down_base_matmul(x, c, a, **spatial_kwargs))
        )
        rows.append((block_m, block_n, up_time, full_time))

    rows.sort(key=lambda row: row[3])
    print("Spatial config: interleaved base_chunk=136 down_chunk=34 block_k_down=128 block_m_down=16")
    print("-" * 72)
    print(f"{'BLOCK_M':>8} {'BLOCK_N':>8} {'UpOnly(us)':>12} {'Spatial+Up(us)':>16}")
    print("-" * 72)
    for block_m, block_n, up_time, full_time in rows:
        print(f"{block_m:>8} {block_n:>8} {up_time:>12.2f} {full_time:>16.2f}")
    print("-" * 72)


if __name__ == "__main__":
    main()
