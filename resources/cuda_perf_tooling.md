# CUDA Performance + Tooling

This module teaches you how to measure and optimize GPU kernels.

## 1) What to optimize (in order)

1. Correctness (always)
2. Memory bandwidth and access patterns
3. Kernel launch overhead
4. Occupancy / latency hiding
5. Arithmetic intensity

## 2) Nsight Systems vs Nsight Compute

- **Nsight Systems**: system-level view. Good for seeing CPU/GPU overlap, kernel launch timing, and where time is spent.
- **Nsight Compute**: kernel-level view. Good for instruction mix, memory throughput, occupancy, and bottlenecks.

## 3) Basic Nsight Compute workflow

1. Run your kernel under `ncu`.
2. Look at:
   - Memory throughput (global, L2, shared)
   - Achieved occupancy
   - Warp stall reasons
3. Apply one change at a time and measure again.

## 4) Occupancy basics

Occupancy depends on:
- Registers per thread
- Shared memory per block
- Block size

Higher occupancy is not always faster; it just helps hide latency.

## 5) Practical measurement checklist

- Is global memory access coalesced?
- Are you doing avoidable host<->device copies?
- Is the kernel launch overhead bigger than the kernel work?
- Can you batch work to increase GPU utilization?

## References

- Nsight Compute Documentation
  https://docs.nvidia.com/nsight-compute/
- Nsight Systems Documentation
  https://docs.nvidia.com/nsight-systems/
- Occupancy Calculator overview (CUDA documentation)
  https://docs.nvidia.com/cuda/cuda-occupancy-calculator/
