#!/usr/bin/env python3
"""Profile Qwen3VLModel decode step (1 token) to find the next bottleneck
after the SwiGLU fusion.  Uses torch.profiler to get per-kernel timings."""

import torch
import sys
import importlib.util

sys.path.insert(0, "..")
spec = importlib.util.spec_from_file_location("qwen3_vl", "../models/qwen3_vl.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

m = mod.Qwen3VLModel(n_layers=4).to(dtype=torch.float16).cuda().eval()
ids = torch.randint(0, 151936, (1, 1), device="cuda")

# Warm up Triton JIT for the fused path
with torch.no_grad():
    for _ in range(10):
        m(ids)
torch.cuda.synchronize()

# Profile
with torch.no_grad():
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(50):
            m(ids)

# Aggregate by op name
events = prof.key_averages(group_by_input_shape=True)
kernels = []
total_time = 0.0
for ev in events:
    if ev.device_time_total > 0:
        kernels.append((ev.device_time_total / 50, ev.device_time_total,
                        ev.count, str(ev.key), str(ev.input_shapes)))
        total_time += ev.device_time_total / 50

# Print top 20
kernels.sort(key=lambda x: x[0], reverse=True)
print(f"Total GPU time per decode step (4-layer): {total_time:.1f} µs")
print()
print(f"{'µs/step':>10} {'calls':>6} {'pct':>6} {'name':<50} {'shape'}")
print("-" * 130)
cum = 0.0
for us, _, cnt, name, shape in kernels[:25]:
    pct = us / total_time * 100
    cum += pct
    name = (name[:48] + "..") if len(name) > 50 else name
    shape = (shape[:40] + "..") if len(shape) > 42 else shape
    print(f"{us:>10.1f} {cnt:>6} {pct:>5.1f}% {name:<50} {shape}")
    if cum > 99.5:
        break
