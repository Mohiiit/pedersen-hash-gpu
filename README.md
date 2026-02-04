# pedersen-hash-gpu

GPU-accelerated Pedersen hashing for Starknet's bonsai-trie.

## Results (2026-02-04)

| GPU | GPU Batch | GPU Throughput | CPU Throughput (batch 10) | Speedup |
|-----|-----------|----------------|---------------------------|---------|
| RTX 3060 | 16,384 | 286.14 Kelem/s | 230 elem/s | ~1,244× |
| RTX 3080 | 16,384 | 712.28 Kelem/s | 298 elem/s | ~2,390× |

Speedup is GPU throughput vs CPU batch‑10 throughput on the same instance.

## Overview

This crate provides high-performance Pedersen hash computation using CUDA, designed to accelerate Merkle tree operations in Starknet's state management. It targets 10-100× throughput improvement for batched Pedersen hashing by parallelizing elliptic curve scalar multiplications across thousands of CUDA cores.

## Features

- **STARK Field Arithmetic**: Complete 251-bit prime field operations
- **Elliptic Curve Operations**: Point addition, doubling, and scalar multiplication on the STARK curve
- **Pedersen Hash**: Both single-hash and batched implementations
- **GPU Acceleration**: CUDA kernels for parallel hash computation (optional, requires `cuda` feature)
- **Differential Testing**: Property-based testing against reference implementation

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pedersen-hash-gpu = "0.1"
```

For GPU acceleration, enable the `cuda` feature:

```toml
[dependencies]
pedersen-hash-gpu = { version = "0.1", features = ["cuda"] }
```

## Quick Start

```rust
use pedersen_hash_gpu::pedersen::pedersen_hash;
use starknet_types_core::felt::Felt;

let a = Felt::from(314u64);
let b = Felt::from(159u64);
let hash = pedersen_hash(&a, &b);
```

### Batch Hashing

```rust
use pedersen_hash_gpu::pedersen::pedersen_hash_batch;
use starknet_types_core::felt::Felt;

let pairs = vec![
    (Felt::from(1u64), Felt::from(2u64)),
    (Felt::from(3u64), Felt::from(4u64)),
    // ... thousands more
];

let hashes = pedersen_hash_batch(&pairs);
```

## Architecture

### STARK Curve Parameters

The implementation operates on the STARK curve:
- **Prime**: P = 2²⁵¹ + 17·2¹⁹² + 1
- **Curve**: y² = x³ + x + β (Weierstrass form with α=1)
- **Order**: n = 0x0800000000000010ffffffffffffffffb781126dcae7b2321e66a241adc64d2f

### Pedersen Hash Formula

```
H(a, b) = [P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄]_x
```

Where:
- P₀, P₁, P₂, P₃, P₄ are generator points derived from π
- a_low, b_low are the lower 248 bits
- a_high, b_high are the upper 4 bits
- [P]_x denotes the x-coordinate

### Module Structure

```
src/
├── lib.rs          # Crate root and public API
├── error.rs        # Error types
├── field/          # STARK prime field arithmetic
│   ├── constants.rs
│   ├── element.rs
│   └── ops.rs
├── curve/          # Elliptic curve operations
│   ├── affine.rs
│   ├── projective.rs
│   ├── constants.rs
│   └── scalar_mul.rs
├── pedersen/       # Pedersen hash implementation
│   ├── constants.rs
│   └── hash.rs
└── gpu/            # CUDA acceleration (optional)
    └── mod.rs
```

## Benchmarks

Run benchmarks with:

```bash
CPU_BENCH_MAX=10 cargo bench --features bench
```

For GPU benchmarks:

```bash
CPU_BENCH_MAX=0 CUDA_ARCH=sm_86 cargo bench --features "bench,cuda,cudarc/cuda-12040"
```

### Measured Performance (2026-02-04)

Correctness validated on each instance via:
- `cargo test --features "cuda,cudarc/cuda-12040" --test gpu_field_debug -- --ignored`
- `cargo test --features "cuda,cudarc/cuda-12040" --test gpu_pedersen -- --ignored`

**RTX 3060 (Vast.ai, on-demand)**

Environment:
- OS: Ubuntu 22.04 (image: `nvidia/cuda:12.4.0-devel-ubuntu22.04`)
- CPU: Intel Xeon E5-2690 v4 @ 2.60GHz (56 vCPUs)
- RAM: 220 GiB
- GPU: NVIDIA GeForce RTX 3060 (12 GB)
- Driver: 550.144.03
- CUDA Toolkit: 12.4

CPU (single-threaded, `CPU_BENCH_MAX=10`):
- `pedersen_hash_single`: ~178 µs
- Batch 1: 4.42 ms (~226 elem/s)
- Batch 10: 43.4 ms (~230 elem/s)

GPU (`cargo bench --features "bench,cuda,cudarc/cuda-12040"`):
- Batch 1,024: 19.47 ms (~52.60 Kelem/s)
- Batch 4,096: 19.75 ms (~207.38 Kelem/s)
- Batch 16,384: 57.26 ms (~286.14 Kelem/s)

GPU utilization during benchmarks: peak 100% SM (verified via `nvidia-smi dmon`).

**RTX 3080 (Vast.ai, on-demand)**

Environment:
- OS: Ubuntu 22.04 (image: `nvidia/cuda:12.4.0-devel-ubuntu22.04`)
- CPU: Intel Core i3-9100 @ 3.60GHz (4 vCPUs)
- RAM: 15 GiB
- GPU: NVIDIA GeForce RTX 3080
- Driver: 550.144.03
- CUDA Toolkit: 12.4

CPU (single-threaded, `CPU_BENCH_MAX=10`):
- `pedersen_hash_single`: ~139 µs
- Batch 1: 3.40 ms (~294 elem/s)
- Batch 10: 33.5 ms (~298 elem/s)

GPU (`CPU_BENCH_MAX=0` to skip CPU batches):
- Batch 1,024: 22.21 ms (~46.11 Kelem/s)
- Batch 4,096: 21.23 ms (~192.97 Kelem/s)
- Batch 16,384: 23.00 ms (~712.28 Kelem/s)

GPU utilization during benchmarks: peak 100% SM (verified via `nvidia-smi dmon`).

## Development

### Building

```bash
# CPU-only build
cargo build --release

# With CUDA support (requires CUDA toolkit)
cargo build --release --features cuda
```

### Testing

```bash
# Run all tests
cargo test

# Run with logging
RUST_LOG=debug cargo test

# Property-based tests
cargo test --release -- --ignored proptest
```

### GPU Testing (CUDA)

GPU tests are ignored by default. On a CUDA-capable machine, run:

```bash
# Compile CUDA kernel with nvcc and run ignored GPU tests
CUDA_ARCH=sm_80 cargo test --features cuda -- --ignored
```

If `nvcc` is not on your PATH:

```bash
NVCC=/path/to/nvcc CUDA_ARCH=sm_80 cargo test --features cuda -- --ignored
```

Notes:
- `CUDA_ARCH` should match your GPU (e.g., `sm_75`, `sm_80`, `sm_90`).
- On macOS, you can still run CPU-only tests with `cargo test`.

### Differential Testing

The crate includes differential tests against `starknet-crypto`:

```bash
cargo test differential
```

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## References

- [Starknet Cryptography Docs](https://docs.starknet.io/architecture/cryptography/)
- [StarkEx Pedersen Hash Specification](https://docs.starkware.co/starkex/crypto/pedersen-hash-function.html)
- [cuZK Paper](https://eprint.iacr.org/2022/1321) - GPU MSM optimization
