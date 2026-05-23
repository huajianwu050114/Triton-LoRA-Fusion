import torch
from fused_lora_gemm import fused_lora_matmul

def test_fusion_performance():
    M, K, N, r = 64, 4096, 28672, 8
    device = "cuda"
    
    # 初始化数据 (除以 10.0 防止 FP16 溢出)
    X = torch.randn((M, K), dtype=torch.float16, device=device) / 10.0
    C = torch.randn((K, N), dtype=torch.float16, device=device) / 10.0
    A = torch.randn((K, r), dtype=torch.float16, device=device) / 10.0
    B = torch.randn((r, N), dtype=torch.float16, device=device) / 10.0
    
    # --- 1. 正确性验证 ---
    # 原生串行答案
    W_ref = X @ C
    Y_ref = X @ A
    Z_ref = Y_ref @ B
    O_ref = W_ref + Z_ref
    
    # 融合算子答案 (先用 cuBLAS 算完极其轻量的降维，把 Y 传给融合算子)
    Y_tri = X @ A 
    O_tri = fused_lora_matmul(X, C, Y_tri, B)
    
    try:
        torch.testing.assert_close(O_ref, O_tri, atol=1e-2, rtol=1e-2)
        print("✅ Fused Kernel Correctness Check PASSED!")
    except Exception as e:
        print("❌ Fused Kernel Correctness Check FAILED!")
        print(e)
        return

    # --- 2. 性能对决 ---
    # Warmup
    for _ in range(10):
        _ = (X @ C) + (X @ A) @ B
        _ = fused_lora_matmul(X, C, X @ A, B)
    torch.cuda.synchronize()
    
    # Benchmark 原生串行总和
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        O = (X @ C) + (X @ A) @ B
    end.record()
    torch.cuda.synchronize()
    baseline_time = start.elapsed_time(end) / 100 * 1000
    
    # Benchmark 融合策略总和
    start.record()
    for _ in range(100):
        Y = X @ A  # 先执行降维 (14us 左右)
        O = fused_lora_matmul(X, C, Y, B) # 融合 Base 和 Up
    end.record()
    torch.cuda.synchronize()
    fused_time = start.elapsed_time(end) / 100 * 1000
    
    print("-" * 40)
    print(f"Baseline Total Latency : {baseline_time:.2f} us")
    print(f"Fused Strategy Latency : {fused_time:.2f} us")
    print(f"Speedup                : {baseline_time / fused_time:.2f}x")
    print(f"Saved Time             : {baseline_time - fused_time:.2f} us (HBM Read/Write Eliminated)")
    print("-" * 40)

if __name__ == "__main__":
    test_fusion_performance()