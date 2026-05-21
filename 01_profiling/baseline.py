import torch
import time

# 1. init
M, K, N, r = 64, 4096, 28672, 8
dtype = torch.float16
device = "cuda"

print(f"Profiling gate_up_proj: M={M}, K={K}, N={N}, r={r}")

X = torch.randn((M, K), dtype=dtype, device=device)
C = torch.randn((K, N), dtype=dtype, device=device) # Base Layer
A = torch.randn((K, r), dtype=dtype, device=device) # LoRA Down
B = torch.randn((r, N), dtype=dtype, device=device) # LoRA Up

# warm up
for _ in range(10):
    W = X @ C
    Y = X @ A
    Z = Y @ B
    O = W + Z
torch.cuda.synchronize()

# time test
def benchmark(func, iters=100):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        func()
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / iters * 1000  # us

print(f"Base GEMM  (X @ C)   : {benchmark(lambda: X @ C):.2f} us")
print(f"LoRA Down  (X @ A)   : {benchmark(lambda: X @ A):.2f} us")
print(f"LoRA Up    (Y @ B)   : {benchmark(lambda: (X @ A) @ B):.2f} us") 
print(f"Total Seq  (W + Z)   : {benchmark(lambda: X @ C + (X @ A) @ B):.2f} us")

# use profile
print("\nRunning PyTorch Profiler...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/lora_profile'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for i in range(1 + 1 + 3): # wait + warmup + active
        W = X @ C
        Y = X @ A
        Z = Y @ B
        O = W + Z
        prof.step()

print("Trace saved to ./log/lora_profile. \nOpen Chrome and go to chrome://tracing to view the JSON file.")