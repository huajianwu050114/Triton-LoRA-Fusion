import torch
import triton
import triton.language as tl


@triton.jit
def spatial_down_base_kernel(
    x_ptr,
    c_ptr,
    a_ptr,
    w_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_ck,
    stride_cn,
    stride_ak,
    stride_ar,
    stride_wm,
    stride_wn,
    stride_ym,
    stride_yr,
    BLOCK_M_BASE: tl.constexpr,
    BLOCK_N_BASE: tl.constexpr,
    BLOCK_K_BASE: tl.constexpr,
    BLOCK_M_DOWN: tl.constexpr,
    BLOCK_K_DOWN: tl.constexpr,
    BLOCK_R_PAD: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    BASE_CHUNK: tl.constexpr,
    DOWN_CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)

    num_base_m = tl.cdiv(M, BLOCK_M_BASE)
    num_base_n = tl.cdiv(N, BLOCK_N_BASE)
    num_base_tiles = num_base_m * num_base_n
    num_down_m = tl.cdiv(M, BLOCK_M_DOWN)
    num_down_tiles = num_down_m * tl.cdiv(K, BLOCK_K_DOWN)

    if INTERLEAVE:
        cycle = BASE_CHUNK + DOWN_CHUNK
        group_id = pid // cycle
        local_id = pid - group_id * cycle
        is_base = local_id < BASE_CHUNK
        base_task = group_id * BASE_CHUNK + local_id
        down_task = group_id * DOWN_CHUNK + (local_id - BASE_CHUNK)
    else:
        is_base = pid < num_base_tiles
        base_task = pid
        down_task = pid - num_base_tiles

    if is_base:
        if base_task >= num_base_tiles:
            return

        pid_m = base_task % num_base_m
        pid_n = base_task // num_base_m

        base_offs_m = pid_m * BLOCK_M_BASE + tl.arange(0, BLOCK_M_BASE)
        base_offs_n = pid_n * BLOCK_N_BASE + tl.arange(0, BLOCK_N_BASE)
        base_offs_k = tl.arange(0, BLOCK_K_BASE)

        base_x_ptrs = x_ptr + base_offs_m[:, None] * stride_xm + base_offs_k[None, :] * stride_xk
        base_c_ptrs = c_ptr + base_offs_k[:, None] * stride_ck + base_offs_n[None, :] * stride_cn

        acc = tl.zeros((BLOCK_M_BASE, BLOCK_N_BASE), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K_BASE):
            base_k_mask = base_offs_k < K - k0
            x = tl.load(
                base_x_ptrs,
                mask=(base_offs_m[:, None] < M) & base_k_mask[None, :],
                other=0.0,
            )
            c = tl.load(
                base_c_ptrs,
                mask=base_k_mask[:, None] & (base_offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(x, c)
            base_x_ptrs += BLOCK_K_BASE * stride_xk
            base_c_ptrs += BLOCK_K_BASE * stride_ck

        base_w_ptrs = w_ptr + base_offs_m[:, None] * stride_wm + base_offs_n[None, :] * stride_wn
        tl.store(
            base_w_ptrs,
            acc.to(tl.float16),
            mask=(base_offs_m[:, None] < M) & (base_offs_n[None, :] < N),
        )
    else:
        if down_task >= num_down_tiles:
            return

        pid_m = down_task % num_down_m
        pid_k = down_task // num_down_m

        down_offs_m = pid_m * BLOCK_M_DOWN + tl.arange(0, BLOCK_M_DOWN)
        down_offs_k = pid_k * BLOCK_K_DOWN + tl.arange(0, BLOCK_K_DOWN)
        down_offs_r = tl.arange(0, BLOCK_R_PAD)

        down_x_ptrs = x_ptr + down_offs_m[:, None] * stride_xm + down_offs_k[None, :] * stride_xk
        down_a_ptrs = a_ptr + down_offs_k[:, None] * stride_ak + down_offs_r[None, :] * stride_ar

        x = tl.load(
            down_x_ptrs,
            mask=(down_offs_m[:, None] < M) & (down_offs_k[None, :] < K),
            other=0.0,
        )
        a = tl.load(
            down_a_ptrs,
            mask=(down_offs_k[:, None] < K) & (down_offs_r[None, :] < R),
            other=0.0,
        )
        partial = tl.dot(x, a)

        down_y_ptrs = y_ptr + down_offs_m[:, None] * stride_ym + down_offs_r[None, :] * stride_yr
        tl.atomic_add(
            down_y_ptrs,
            partial,
            sem="relaxed",
            mask=(down_offs_m[:, None] < M) & (down_offs_r[None, :] < R),
        )


def _grid_size(num_base_tiles, num_down_tiles, layout, base_chunk, down_chunk):
    if layout == "contiguous":
        return num_base_tiles + num_down_tiles
    if layout == "interleaved":
        groups = max(triton.cdiv(num_base_tiles, base_chunk), triton.cdiv(num_down_tiles, down_chunk))
        return groups * (base_chunk + down_chunk)
    raise ValueError(f"Unsupported layout: {layout}")


def spatial_down_base_matmul(
    x,
    c,
    a,
    *,
    layout="contiguous",
    base_chunk=80,
    down_chunk=20,
    block_m_base=64,
    block_n_base=128,
    block_k_base=64,
    block_m_down=16,
    block_k_down=64,
    block_r_pad=16,
):
    """
    Spatial prototype:
    - base tiles compute W = X @ C
    - down tiles compute partial X @ A over K chunks and atomic-add into Y

    layout="contiguous" launches [all base][all down].
    layout="interleaved" launches repeated [base_chunk base][down_chunk down].
    """
    assert x.is_contiguous() and c.is_contiguous() and a.is_contiguous()
    M, K = x.shape
    K_c, N = c.shape
    K_a, R = a.shape
    assert K == K_c == K_a
    assert block_r_pad >= R
    assert base_chunk > 0 and down_chunk > 0

    w = torch.empty((M, N), device=x.device, dtype=torch.float16)
    y = torch.empty((M, R), device=x.device, dtype=torch.float32)
    y.zero_()

    num_base_tiles = triton.cdiv(M, block_m_base) * triton.cdiv(N, block_n_base)
    num_down_tiles = triton.cdiv(M, block_m_down) * triton.cdiv(K, block_k_down)
    grid = (_grid_size(num_base_tiles, num_down_tiles, layout, base_chunk, down_chunk),)

    spatial_down_base_kernel[grid](
        x,
        c,
        a,
        w,
        y,
        M,
        N,
        K,
        R,
        x.stride(0),
        x.stride(1),
        c.stride(0),
        c.stride(1),
        a.stride(0),
        a.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M_BASE=block_m_base,
        BLOCK_N_BASE=block_n_base,
        BLOCK_K_BASE=block_k_base,
        BLOCK_M_DOWN=block_m_down,
        BLOCK_K_DOWN=block_k_down,
        BLOCK_R_PAD=block_r_pad,
        INTERLEAVE=layout == "interleaved",
        BASE_CHUNK=base_chunk,
        DOWN_CHUNK=down_chunk,
    )
    return w, y
