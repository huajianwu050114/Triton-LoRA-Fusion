import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.lora_up_add import lora_up_add
from kernels.spatial_down_base import spatial_down_base_matmul


def benchmark(func, iters=100, warmup=16):
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

    out_ref = x @ c + (x @ a) @ b
    serial_full = benchmark(lambda: x @ c + (x @ a) @ b)

    cases = []
    for block_n_base in (64, 128, 256):
        for block_k_base in (32, 64, 128):
            for base_chunk, down_chunk in ((120, 30), (128, 32), (136, 34), (144, 36), (160, 40)):
                cases.append((block_n_base, block_k_base, base_chunk, down_chunk))

    rows = []
    for block_n_base, block_k_base, base_chunk, down_chunk in cases:
        kwargs = {
            "layout": "interleaved",
            "base_chunk": base_chunk,
            "down_chunk": down_chunk,
            "block_m_base": 64,
            "block_n_base": block_n_base,
            "block_k_base": block_k_base,
            "block_m_down": 16,
            "block_k_down": 128,
        }
        try:
            w, y = spatial_down_base_matmul(x, c, a, **kwargs)
            out = lora_up_add(w, y, b)
            torch.testing.assert_close(out_ref, out, atol=2e-2, rtol=2e-2)
            base_down_time = benchmark(lambda kwargs=kwargs: spatial_down_base_matmul(x, c, a, **kwargs))
            full_time = benchmark(
                lambda kwargs=kwargs: (
                    lambda wy: lora_up_add(wy[0], wy[1], b)
                )(spatial_down_base_matmul(x, c, a, **kwargs))
            )
            rows.append((full_time, base_down_time, block_n_base, block_k_base, base_chunk, down_chunk))
        except Exception as exc:
            print(
                f"SKIP block_n={block_n_base} block_k={block_k_base} "
                f"base_chunk={base_chunk} down_chunk={down_chunk}: {exc}"
            )

    rows.sort(key=lambda row: row[0])
    print(f"serial_full: {serial_full:.2f} us")
    print("-" * 112)
    print(
        f"{'Full(us)':>10} {'Base+Down(us)':>14} {'Speedup':>10} {'BN_BASE':>8} "
        f"{'BK_BASE':>8} {'base_chunk':>11} {'down_chunk':>11}"
    )
    print("-" * 112)
    for full_time, base_down_time, block_n_base, block_k_base, base_chunk, down_chunk in rows[:24]:
        print(
            f"{full_time:>10.2f} {base_down_time:>14.2f} {serial_full / full_time:>9.2f}x "
            f"{block_n_base:>8} {block_k_base:>8} {base_chunk:>11} {down_chunk:>11}"
        )
    print("-" * 112)


if __name__ == "__main__":
    main()
