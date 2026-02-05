# CUDA Primer (Beginner Friendly)

This is a compact walkthrough of the CUDA programming model and how to get productive fast.

## 1) CUDA mental model

- A GPU kernel is a function that runs on many threads in parallel.
- Threads are grouped into **blocks**; blocks form a **grid**.
- Threads in a block can share **shared memory** and synchronize; blocks cannot synchronize with each other inside a kernel.
- You choose the grid/block sizes; the GPU schedules blocks onto SMs (streaming multiprocessors).

Key idea: you are trading **lots of simple threads** for high throughput, so you want to maximize parallel work and memory coalescing.

## 2) Memory hierarchy (the short version)

- **Global memory**: large, high latency. Coalesced access matters a lot.
- **Shared memory**: on-chip, much faster, visible to threads in the same block.
- **Registers**: fastest, per-thread, limited.
- **Constant/texture**: cached read-only paths useful for constant tables.

Rule of thumb: minimize global memory traffic and make accesses contiguous.

## 3) First hands-on exercises (minimal but effective)

These are the CUDA samples to start with. Build/run them before touching project code:

1) `vectorAdd` — understand grid/block indexing and global memory access.
2) `matrixMul` — see tiling with shared memory.
3) `simpleCudaGraphs` — understand kernel launch overhead and graph capture (optional).

After each sample, answer:
- What is the kernel doing per thread?
- What is the memory access pattern?
- How is occupancy affected by block size?

## 4) CUDA compilation (bare minimum)

- `.cu` files compiled by `nvcc`.
- Host code is C++ (or Rust FFI), device code uses CUDA extensions.
- Your build will link against the CUDA runtime and driver libraries.

## 5) Beginner checklist

- Can you explain how a thread index maps to data?
- Can you choose block sizes and explain why?
- Can you identify coalesced vs strided global access?
- Can you use shared memory to reduce global reads?

## 6) Official docs to keep open

- CUDA C++ Programming Guide
- CUDA Best Practices Guide
- CUDA Samples (GitHub)
- CUDA Toolkit Installation Guide

(Links are in the references section below.)

## References

- CUDA C++ Programming Guide (programming model, memory hierarchy)
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices Guide (performance, memory access patterns)
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- CUDA Samples (hands-on examples)
  https://github.com/NVIDIA/cuda-samples
- CUDA Toolkit Installation Guide for Linux
  https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
