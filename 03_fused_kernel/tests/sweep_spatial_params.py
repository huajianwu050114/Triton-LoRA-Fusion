import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.lora_up_add import lora_up_add
from kernels.spatial_down_base import spatial_down_base_matmul


def benchmark(func, iters=60, warmup=8):
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
    device = "cuda"
    dtype = torch.float16

    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=dtype, device=device) / 10.0
    c = torch.randn((K, N), dtype=dtype, device=device) / 10.0
    a = torch.randn((K, R), dtype=dtype, device=device) / 10.0
    b = torch.randn((R, N), dtype=dtype, device=device) / 10.0

    w_ref = x @ c
    y_ref = (x @ a).float()
    out_ref = w_ref + y_ref.to(dtype) @ b

    serial_base_down = benchmark(lambda: (x @ c, x @ a))
    serial_full = benchmark(lambda: x @ c + (x @ a) @ b)

    cases = [
        ("contiguous_k64_m16", {"layout": "contiguous", "block_k_down": 64, "block_m_down": 16}),
        ("contiguous_k128_m16", {"layout": "contiguous", "block_k_down": 128, "block_m_down": 16}),
        ("contiguous_k256_m16", {"layout": "contiguous", "block_k_down": 256, "block_m_down": 16}),
        ("interleaved136_34_k64_m16", {"layout": "interleaved", "base_chunk": 136, "down_chunk": 34, "block_k_down": 64, "block_m_down": 16}),
        ("interleaved136_34_k128_m16", {"layout": "interleaved", "base_chunk": 136, "down_chunk": 34, "block_k_down": 128, "block_m_down": 16}),
        ("interleaved136_34_k256_m16", {"layout": "interleaved", "base_chunk": 136, "down_chunk": 34, "block_k_down": 256, "block_m_down": 16}),
        ("three136_k64_m16", {"layout": "three_range", "base_chunk": 136, "block_k_down": 64, "block_m_down": 16}),
        ("three136_k128_m16", {"layout": "three_range", "base_chunk": 136, "block_k_down": 128, "block_m_down": 16}),
        ("three136_k256_m16", {"layout": "three_range", "base_chunk": 136, "block_k_down": 256, "block_m_down": 16}),
        ("three136_k128_m32", {"layout": "three_range", "base_chunk": 136, "block_k_down": 128, "block_m_down": 32}),
        ("three136_k256_m32", {"layout": "three_range", "base_chunk": 136, "block_k_down": 256, "block_m_down": 32}),
        ("three160_k128_m32", {"layout": "three_range", "base_chunk": 160, "block_k_down": 128, "block_m_down": 32}),
        ("fullk_contiguous_k64_m16", {"layout": "contiguous", "block_k_down": 64, "block_m_down": 16, "down_mode": "full_k"}),
        ("fullk_contiguous_k128_m16", {"layout": "contiguous", "block_k_down": 128, "block_m_down": 16, "down_mode": "full_k"}),
        ("fullk_contiguous_k256_m16", {"layout": "contiguous", "block_k_down": 256, "block_m_down": 16, "down_mode": "full_k"}),
        ("fullk_three136_k64_m16", {"layout": "three_range", "base_chunk": 136, "block_k_down": 64, "block_m_down": 16, "down_mode": "full_k"}),
        ("fullk_three136_k128_m16", {"layout": "three_range", "base_chunk": 136, "block_k_down": 128, "block_m_down": 16, "down_mode": "full_k"}),
        ("fullk_three136_k256_m16", {"layout": "three_range", "base_chunk": 136, "block_k_down": 256, "block_m_down": 16, "down_mode": "full_k"}),
        ("fullk_three136_k128_m32", {"layout": "three_range", "base_chunk": 136, "block_k_down": 128, "block_m_down": 32, "down_mode": "full_k"}),
        ("fullk_three136_k256_m32", {"layout": "three_range", "base_chunk": 136, "block_k_down": 256, "block_m_down": 32, "down_mode": "full_k"}),
        ("fullk_three160_k128_m32", {"layout": "three_range", "base_chunk": 160, "block_k_down": 128, "block_m_down": 32, "down_mode": "full_k"}),
    ]

    rows = []
    for name, kwargs in cases:
        w, y = spatial_down_base_matmul(x, c, a, **kwargs)
        torch.testing.assert_close(w_ref, w, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(y_ref, y, atol=2e-2, rtol=2e-2)
        out = w + y.to(dtype) @ b
        torch.testing.assert_close(out_ref, out, atol=2e-2, rtol=2e-2)
        out_tri = lora_up_add(w, y, b)
        torch.testing.assert_close(out_ref, out_tri, atol=2e-2, rtol=2e-2)

        base_down_time = benchmark(lambda kwargs=kwargs: spatial_down_base_matmul(x, c, a, **kwargs))
        torch_full_time = benchmark(
            lambda kwargs=kwargs: (
                lambda wy: wy[0] + wy[1].to(dtype) @ b
            )(spatial_down_base_matmul(x, c, a, **kwargs))
        )
        triton_full_time = benchmark(
            lambda kwargs=kwargs: (
                lambda wy: lora_up_add(wy[0], wy[1], b)
            )(spatial_down_base_matmul(x, c, a, **kwargs))
        )
        rows.append((
            name,
            base_down_time,
            serial_base_down / base_down_time,
            torch_full_time,
            serial_full / torch_full_time,
            triton_full_time,
            serial_full / triton_full_time,
        ))

    rows.sort(key=lambda item: item[5])

    print(f"serial_base_down: {serial_base_down:.2f} us")
    print(f"serial_full     : {serial_full:.2f} us")
    print("-" * 136)
    print(f"{'Case':<32} {'Base+Down(us)':>14} {'BD speedup':>12} {'TorchFull(us)':>14} {'Torch spd':>10} {'TriFull(us)':>12} {'Tri spd':>10}")
    print("-" * 136)
    for name, bd_time, bd_speed, torch_time, torch_speed, tri_time, tri_speed in rows:
        print(f"{name:<32} {bd_time:>14.2f} {bd_speed:>12.2f}x {torch_time:>14.2f} {torch_speed:>10.2f}x {tri_time:>12.2f} {tri_speed:>10.2f}x")
    print("-" * 108)


if __name__ == "__main__":
    main()
