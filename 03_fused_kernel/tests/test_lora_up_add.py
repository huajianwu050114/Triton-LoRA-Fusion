import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from kernels.lora_up_add import lora_up_add


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


def main():
    M, N, R = 64, 28672, 8
    device = "cuda"

    torch.manual_seed(0)
    w = torch.randn((M, N), dtype=torch.float16, device=device) / 10.0
    y = torch.randn((M, R), dtype=torch.float32, device=device) / 10.0
    b = torch.randn((R, N), dtype=torch.float16, device=device) / 10.0

    ref = w + y.to(torch.float16) @ b
    out = lora_up_add(w, y, b)
    torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)

    torch_time = benchmark(lambda: w + y.to(torch.float16) @ b)
    triton_time = benchmark(lambda: lora_up_add(w, y, b))

    print("LoRA Up+Add correctness PASSED")
    print(f"Torch up+add latency : {torch_time:.2f} us")
    print(f"Triton up+add latency: {triton_time:.2f} us")
    print(f"Speedup              : {torch_time / triton_time:.2f}x")


if __name__ == "__main__":
    main()
