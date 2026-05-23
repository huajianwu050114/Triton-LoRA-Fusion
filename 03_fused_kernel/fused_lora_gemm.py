import torch
import triton
import triton.language as tl

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)

@triton.jit
def fused_lora_kernel(
    # Base Layer Pointers
    x_ptr, c_ptr, 
    # LoRA Expand Pointers (Y = X @ A, 已经在外部算好; lora_b 是升维权重)
    y_ptr, lora_b_ptr, 
    # Output Pointer
    out_ptr,
    
    # Dimensions
    M, N, K, r: tl.constexpr,
    
    # Strides
    stride_xm, stride_xk,
    stride_ck, stride_cn,
    stride_ym, stride_yr,
    stride_br, stride_bn,
    stride_outm, stride_outn,
    
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    
    # --- Part 1: Base GEMM (X @ C) ---
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    c_ptrs = c_ptr + (offs_k[:, None] * stride_ck + offs_n[None, :] * stride_cn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        mask_c = (offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < N)
        
        x_val = tl.load(x_ptrs, mask=mask_x, other=0.0)
        c_val = tl.load(c_ptrs, mask=mask_c, other=0.0)
        
        accumulator += tl.dot(x_val, c_val)
        
        x_ptrs += BLOCK_SIZE_K * stride_xk
        c_ptrs += BLOCK_SIZE_K * stride_ck

    # --- Part 2: 融合 LoRA Expand (Y @ B) ---
    # 因为 r (通常为 8 或 16) 非常小，可以直接作为一个 Block 一次性 load 进来，不需要内层循环！
    BLOCK_SIZE_R: tl.constexpr = 16
    offs_r = tl.arange(0, BLOCK_SIZE_R)
    
    # y_ptrs shape: [BLOCK_SIZE_M, r]
    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_r[None, :] * stride_yr)
    # lora_b_ptrs shape: [r, BLOCK_SIZE_N]
    lora_b_ptrs = lora_b_ptr + (offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn)
    
    mask_y = (offs_m[:, None] < M) & (offs_r[None, :] < r)
    mask_b = (offs_r[:, None] < r) & (offs_n[None, :] < N)
    
    y_val = tl.load(y_ptrs, mask=mask_y, other=0.0)
    lora_b_val = tl.load(lora_b_ptrs, mask=mask_b, other=0.0)
    
    # 神奇的时刻：直接在仍处于 FP32 精度的 accumulator 上累加 LoRA 的结果
    # 彻底省去了 Base GEMM 写入显存再读出来的巨大开销
    accumulator += tl.dot(y_val, lora_b_val)

    # --- Part 3: 一次性写回 ---
    out = accumulator.to(tl.float16)
    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (offs_outm[:, None] * stride_outm + offs_outn[None, :] * stride_outn)
    mask_out = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    
    tl.store(out_ptrs, out, mask=mask_out)

def fused_lora_matmul(x, c, y, lora_b):
    """
    x: [M, K] - 输入
    c: [K, N] - 主干权重 Base
    y: [M, r] - 降维后的中间结果 (已经算好)
    lora_b: [r, N] - 升维权重 LoRA Up
    """
    assert x.is_contiguous() and c.is_contiguous()
    assert y.is_contiguous() and lora_b.is_contiguous()
    
    M, K = x.shape
    _, N = c.shape
    _, r = y.shape
    
    # 分配最终输出显存
    out = torch.empty((M, N), device=x.device, dtype=torch.float16)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # 启动融合 Kernel
    fused_lora_kernel[grid](
        x, c, y, lora_b, out,
        M, N, K, r,
        x.stride(0), x.stride(1),
        c.stride(0), c.stride(1),
        y.stride(0), y.stride(1),
        lora_b.stride(0), lora_b.stride(1),
        out.stride(0), out.stride(1)
    )
    return out