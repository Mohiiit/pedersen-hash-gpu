# Project Walkthrough (pedersen-hash-gpu)

This is a code-oriented map from CUDA concepts to this repository.

## 1) High-level flow

1. Inputs are pairs of `Felt` values (Starknet field elements).
2. We split each input into low/high scalars.
3. We perform fixed-point scalar multiplication for each component.
4. We sum the resulting points and return the x-coordinate.

## 2) Where things live

- `src/pedersen/` — CPU reference logic and helpers.
- `cuda/pedersen.cu` — CUDA kernels, field ops, and batch hashing.
- `src/gpu/` — Rust <-> CUDA bridge (memory, launch, error handling).
- `src/field/` and `src/curve/` — field arithmetic and curve math.

## 3) Data layout (important for CUDA)

- Field elements are stored as 4 x 64-bit limbs, little-endian.
- Kernel inputs are arrays of limbs for `a` and `b`.
- Outputs are limbs for the resulting x-coordinate.

If you change this layout, you must update both Rust and CUDA sides.

## 4) Debugging workflow

- Start with CPU reference hash and compare to GPU output.
- Test with small batch sizes (1, 2, 4, 8) before scaling up.
- Use `RUST_LOG=debug` (if available) and add temporary kernel prints only for small runs.

## 5) Performance workflow

- Make sure batching is large enough to amortize kernel launch overhead.
- Use Nsight Compute to find memory bottlenecks.
- Try different block sizes and measure, don’t guess.

## 6) Suggested small exercises

1. Change batch size, re-run benchmarks, record throughput.
2. Add a debug mode that compares GPU and CPU for random vectors.
3. Experiment with block sizes in `cuda/pedersen.cu` and measure.
