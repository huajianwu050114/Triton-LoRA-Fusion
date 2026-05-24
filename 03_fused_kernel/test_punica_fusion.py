import torch
import triton
from fused_punica_base_gemm import fused_sgmv_base_expand_kernel

def test_sgmv_fusion():
    M, K_BASE, N, r = 64, 4096, 28672, 8
    num_loras = 2 # 假设当前 Batch 有两个不同的 LoRA 请求
    device = "cuda"
    
    # --- 1. 模拟网络张量 ---
    X = torch.randn((M, K_BASE), dtype=torch.float16, device=device) / 10.0
    C = torch.randn((K_BASE, N), dtype=torch.float16, device=device) / 10.0
    Y = torch.randn((M, r), dtype=torch.float16, device=device) / 10.0  # (降维结果已准备好)
    B_weights = torch.randn((num_loras, N, r), dtype=torch.float16, device=device) / 10.0
    
    # --- 2. 模拟 SGLang 的 Segment 调度信息 ---
    # 假设前 30 个 token 用 lora_0，后 34 个 token 用 lora_1
    seg_indptr = torch.tensor([0, 30, 64], dtype=torch.int32, device=device)
    weight_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    lora_ranks = torch.tensor([8, 8], dtype=torch.int32, device=device)
    permutation = torch.arange(M, dtype=torch.int32, device=device) # 物理位置映射
    scalings = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    num_segs = 2
    
    # --- 3. 计算原生参考值 (模拟分离执行) ---
    W_ref = X @ C
    Z_ref = torch.zeros_like(W_ref)
    # Punica 逻辑：分段计算
    Z_ref[:30] = Y[:30] @ B_weights[0].T
    Z_ref[30:] = Y[30:] @ B_weights[1].T
    O_ref = W_ref + Z_ref
    
    # --- 4. 运行我们的融合算子 ---
    O_tri = torch.zeros_like(O_ref)
    
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_N']), 
        1,       # slice_id (此处不分片)
        num_segs # segment 数量
    )
    
    fused_sgmv_base_expand_kernel[grid](
        X, C, X.stride(0), X.stride(1), C.stride(0), C.stride(1),
        K_BASE, 64, # BLOCK_K_BASE
        Y, B_weights, O_tri,
        Y.stride(0), Y.stride(1),
        B_weights.stride(0), B_weights.stride(1), B_weights.stride(2),
        O_tri.stride(0), O_tri.stride(1),
        seg_indptr, weight_indices, lora_ranks, permutation, scalings, num_segs,
        N, r, 
        BLOCK_M=64, BLOCK_N=128, BLOCK_K_LORA=16 # 补齐 TensorCore 需要的 16 维度
    )
    
    try:
        torch.testing.assert_close(O_ref, O_tri, atol=1e-2, rtol=1e-2)
        print("✅ Punica SGMV Fusion Correctness Check PASSED!")
    except Exception as e:
        print("❌ Fusion FAILED!")
        print(e)

if __name__ == "__main__":
    test_sgmv_fusion()