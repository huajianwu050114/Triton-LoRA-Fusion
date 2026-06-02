import torch
import triton
import triton.language as tl


@triton.jit
def lora_up_add_kernel(
    w_ptr,
    y_ptr,
    b_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    stride_wm,
    stride_wn,
    stride_ym,
    stride_yr,
    stride_br,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R_PAD: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_r = tl.arange(0, BLOCK_R_PAD)

    w = tl.load(
        w_ptr + offs_m[:, None] * stride_wm + offs_n[None, :] * stride_wn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    ).to(tl.float32)

    y = tl.load(
        y_ptr + offs_m[:, None] * stride_ym + offs_r[None, :] * stride_yr,
        mask=(offs_m[:, None] < M) & (offs_r[None, :] < R),
        other=0.0,
    ).to(tl.float16)
    b = tl.load(
        b_ptr + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn,
        mask=(offs_r[:, None] < R) & (offs_n[None, :] < N),
        other=0.0,
    )

    out = w + tl.dot(y, b)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        out.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def lora_up_add(w, y, b, *, block_m=32, block_n=256, block_r_pad=16):
    assert w.is_contiguous() and y.is_contiguous() and b.is_contiguous()
    M, N = w.shape
    M_y, R = y.shape
    R_b, N_b = b.shape
    assert M == M_y and R == R_b and N == N_b
    assert block_r_pad >= R

    out = torch.empty_like(w)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    lora_up_add_kernel[grid](
        w,
        y,
        b,
        out,
        M,
        N,
        R,
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R_PAD=block_r_pad,
    )
    return out
