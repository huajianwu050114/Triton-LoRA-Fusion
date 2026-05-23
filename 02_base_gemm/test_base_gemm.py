import torch
from matmul_kernel import triton_matmul

def test_correctness_and_speed():
    M, K, N = 64, 4096, 28672
    device = "cuda"
    
    # 缩减随机数范围，避免 fp16 在 K=4096 时溢出
    X = torch.randn((M, K), dtype=torch.float16, device=device) / 10.0
    C = torch.randn((K, N), dtype=torch.float16, device=device) / 10.0
    
    # 1. 验证正确性
    ref_out = X @ C
    tri_out = triton_matmul(X, C)
    
    try:
        torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)
        print("✅ Correctness Check PASSED! Triton output matches PyTorch.")
    except Exception as e:
        print("❌ Correctness Check FAILED!")
        print(e)
        return

    # 2. 测量各自的性能
    # Warmup
    for _ in range(10):
        _ = X @ C
        _ = triton_matmul(X, C)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        _ = X @ C
    end_event.record()
    torch.cuda.synchronize()
    pytorch_time = start_event.elapsed_time(end_event) / 100 * 1000 # us
    
    # Benchmark Triton
    start_event.record()
    for _ in range(100):
        _ = triton_matmul(X, C)
    end_event.record()
    torch.cuda.synchronize()
    triton_time = start_event.elapsed_time(end_event) / 100 * 1000 # us
    
    print(f"PyTorch (cuBLAS) Latency: {pytorch_time:.2f} us")
    print(f"Triton Base GEMM Latency: {triton_time:.2f} us")
    print(f"Performance Ratio (Triton/cuBLAS): {triton_time / pytorch_time * 100:.1f}%")

if __name__ == "__main__":
    test_correctness_and_speed()