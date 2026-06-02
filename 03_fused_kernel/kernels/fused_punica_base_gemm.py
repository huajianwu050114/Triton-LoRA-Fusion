import torch
import triton
import triton.language as tl

@triton.jit
def fused_sgmv_base_expand_kernel(
    # --- Base GEMM inputs ---
    base_x,          # [M, K_base] 主干输入
    base_c,          # [K_base, N] 主干权重
    base_x_stride_0, base_x_stride_1,
    base_c_stride_0, base_c_stride_1,
    K_BASE: tl.constexpr,
    BLOCK_K_BASE: tl.constexpr,
    
    # --- Punica SGMV inputs ---
    y,               # [M, MAX_RANK] 降维输出
    weights,         # [num_loras, N, MAX_RANK] 升维权重
    output,          # [M, N] 最终输出
    y_stride_0, y_stride_1,
    w_stride_0, w_stride_1, w_stride_2,
    out_stride_0, out_stride_1,
    
    # --- Segment / Routing Info ---
    seg_indptr,      # Segment 边界指针
    weight_indices,  # 映射到哪个 LoRA id
    lora_ranks,      # 每个 LoRA 的实际 rank
    permutation,     # Token 的物理位置映射
    scalings,        # LoRA scaling (alpha / r)
    num_segs,
    
    # --- Meta parameters ---
    N_DIM: tl.constexpr,
    MAX_RANK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K_LORA: tl.constexpr,
):
    # 1. 解析 Segment 
    pid_s = tl.program_id(axis=2)
    if pid_s >= num_segs:
        return

    w_index = tl.load(weight_indices + pid_s)
    cur_rank = tl.load(lora_ranks + w_index)
    scaling = tl.load(scalings + w_index)
    
    seg_start = tl.load(seg_indptr + pid_s)
    seg_end = tl.load(seg_indptr + pid_s + 1)
    cur_rank = tl.minimum(MAX_RANK, cur_rank)

    # bias
    s_offset_logical = tl.arange(0, BLOCK_M) + seg_start
    s_offset_physical = tl.load(
        permutation + s_offset_logical, mask=s_offset_logical < seg_end
    )

    pid_n = tl.program_id(axis=0)
    n_offset = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    
    # ==============================================================
    # step1 Base GEMM (X @ C)
    # ==============================================================
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    base_k_offset = tl.arange(0, BLOCK_K_BASE)
    
    base_x_ptrs = base_x + (s_offset_physical[:, None] * base_x_stride_0 + base_k_offset[None, :] * base_x_stride_1)
    base_c_ptrs = base_c + (base_k_offset[:, None] * base_c_stride_0 + n_offset[None, :] * base_c_stride_1)
    
    for k in range(0, tl.cdiv(K_BASE, BLOCK_K_BASE)):
        mask_x = (s_offset_logical[:, None] < seg_end) & (base_k_offset[None, :] < K_BASE - k * BLOCK_K_BASE)
        mask_c = (base_k_offset[:, None] < K_BASE - k * BLOCK_K_BASE) & (n_offset[None, :] < N_DIM)
        
        x_tile = tl.load(base_x_ptrs, mask=mask_x, other=0.0)
        c_tile = tl.load(base_c_ptrs, mask=mask_c, other=0.0)
        accumulator += tl.dot(x_tile, c_tile)
        
        base_x_ptrs += BLOCK_K_BASE * base_x_stride_1
        base_c_ptrs += BLOCK_K_BASE * base_c_stride_0

    # ==============================================================
    # step2. SGMV Expand (Y @ LoRA_B)
    # ==============================================================
    if cur_rank > 0:
        lora_k_offset = tl.arange(0, BLOCK_K_LORA)
        y_ptrs = y + (s_offset_physical[:, None] * y_stride_0 + lora_k_offset[None, :] * y_stride_1)
        w_ptrs = weights + w_index * w_stride_0 + (lora_k_offset[:, None] * w_stride_2 + n_offset[None, :] * w_stride_1)
        
        lora_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(cur_rank, BLOCK_K_LORA)):
            mask_y = (s_offset_logical[:, None] < seg_end) & (lora_k_offset[None, :] < cur_rank - k * BLOCK_K_LORA)
            mask_w = (lora_k_offset[:, None] < cur_rank - k * BLOCK_K_LORA) & (n_offset[None, :] < N_DIM)
            
            y_tile = tl.load(y_ptrs, mask=mask_y, other=0.0)
            w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
            lora_acc += tl.dot(y_tile, w_tile)
            
            y_ptrs += BLOCK_K_LORA * y_stride_1
            w_ptrs += BLOCK_K_LORA * w_stride_2
            
        # Punica SGMV add to Base GEMM acc
        accumulator += lora_acc * scaling

    # ==============================================================
    # step3. write back to HBM
    # ==============================================================
    output_ptr = output + (s_offset_physical[:, None] * out_stride_0 + n_offset[None, :] * out_stride_1)
    output_mask = (s_offset_logical[:, None] < seg_end) & (n_offset[None, :] < N_DIM)
    
    tl.store(output_ptr, accumulator.to(output.dtype.element_ty), mask=output_mask)