"""
SGLang MergedColumnParallelLinear Forward Pass Patch
该补丁展示了如何拦截原生线性层的前向传播，植入我们的 Epilogue Fusion 算子。
"""
import torch
from sglang.srt.lora.backend.triton_backend import sgemm_lora_a_fwd
from fused_punica_base_gemm import fused_sgmv_base_expand_kernel

def patched_forward(self, input_: torch.Tensor):
    """
    替换原生 SGLang 中 Llama 模型的 gate_up_proj 前向传播逻辑
    """
    # 获取当前的 LoRA 批处理信息
    lora_backend = self.lora_backend
    batch_info = getattr(lora_backend, "sgemm_batch_info", None) if lora_backend else None

    # =====================================================================
    # 🚀 快速通道 (Fast-Path)：命中我们融合算子的条件
    # =====================================================================
    # 假设我们针对 M <= 64 的小 Batch 推理场景进行了极致特化
    if lora_backend and batch_info and input_.shape[0] <= 64:
        # 1. 独立执行 LoRA 降维 (Y = X @ A)
        # 这步极快，完全复用 SGLang 极其优秀的 sgemm_lora_a_fwd 算子
        lora_a_output = sgemm_lora_a_fwd(
            input_, 
            self.lora_a_weights, 
            batch_info, 
            stack_num=2
        )
        
        # 2. 准备输出张量
        output_dim = self.lora_b_weights.shape[-2] // 2
        final_output = torch.empty((input_.shape[0], self.weight.shape[0]), dtype=input_.dtype, device=input_.device)
        
        # 3. 💣 拦截点：绝不调用原生 base_layer 计算！
        # 直接启动我们的终极融合算子，将 Base 权重和 LoRA 权重一起传进去！
        grid = lambda META: (triton.cdiv(final_output.shape[1], META['BLOCK_N']), 1, batch_info.num_segments)
        
        fused_sgmv_base_expand_kernel[grid](
            base_x=input_, 
            base_c=self.weight, # 直接拿主干权重去算，根本不产生 base_output
            base_x_stride_0=input_.stride(0), base_x_stride_1=input_.stride(1),
            base_c_stride_0=self.weight.stride(0), base_c_stride_1=self.weight.stride(1),
            K_BASE=input_.shape[1],
            BLOCK_K_BASE=64,
            
            y=lora_a_output, 
            weights=self.lora_b_weights, 
            output=final_output,
            # ... 省略繁杂的 stride 参数 ...
            
            seg_indptr=batch_info.seg_indptr,
            weight_indices=batch_info.weight_indices,
            lora_ranks=batch_info.lora_ranks,
            permutation=batch_info.permutation,
            scalings=batch_info.scalings,
            num_segs=batch_info.num_segments,
            
            N_DIM=final_output.shape[1], MAX_RANK=self.lora_b_weights.shape[-1],
            BLOCK_M=64, BLOCK_N=128, BLOCK_K_LORA=16
        )
        return final_output

    # =====================================================================
    # 🐢 优雅回退 (Graceful Fallback)：原生逻辑
    # =====================================================================
    # 如果不满足融合条件（比如超大 Batch），乖乖走原生的耗时逻辑
    base_output = self.base_layer(input_) # 巨大的显存开销在这里产生
    
    if lora_backend:
        return lora_backend.run_gate_up_lora(
            input_, self.lora_a_weights, self.lora_b_weights, base_output=base_output
        )
        
    return base_output