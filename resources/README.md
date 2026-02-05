# Learning Resources: CUDA + Pedersen Hash + This Project

This directory is a beginner-friendly, project-specific learning path. It is split into short modules so you can go from zero to productive without drowning in theory.

## How to use this resource

1. Skim the learning path in `cuda_primer.md`.
2. Do the hands-on CUDA samples (listed in `cuda_primer.md`).
3. Read `pedersen.md` to understand the hash and curve the project implements.
4. Read `project_notes.md` to understand how this repo maps CUDA concepts to real code.
5. Use `cuda_perf_tooling.md` to profile and optimize kernels.

## Files

- `cuda_primer.md` — CUDA fundamentals, mental models, and beginner exercises.
- `cuda_perf_tooling.md` — profiling, bottlenecks, Nsight Compute/Systems, occupancy.
- `pedersen.md` — Pedersen hash and the STARK curve details used here.
- `project_notes.md` — walkthrough of this repo and how data flows through the CUDA kernels.

## Suggested 7‑day plan (2–3 hours/day)

Day 1: `cuda_primer.md` (concepts + first 2 samples)
Day 2: `cuda_primer.md` (memory model + samples)
Day 3: `cuda_perf_tooling.md` (Nsight basics + one profile)
Day 4: `pedersen.md` (hash/curve + map to code)
Day 5: `project_notes.md` (kernel + data layout)
Day 6: Re-run samples, try a small change in our kernel
Day 7: Profile and write down a micro-optimization plan
