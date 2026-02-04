# bonsai-trie-gpu

GPU-accelerated Pedersen hashing for Starknet's bonsai-trie.

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
bonsai-trie-gpu = "0.1"
```

For GPU acceleration, enable the `cuda` feature:

```toml
[dependencies]
bonsai-trie-gpu = { version = "0.1", features = ["cuda"] }
```

## Quick Start

```rust
use bonsai_trie_gpu::pedersen::pedersen_hash;
use starknet_types_core::felt::Felt;

let a = Felt::from(314u64);
let b = Felt::from(159u64);
let hash = pedersen_hash(&a, &b);
```

### Batch Hashing

```rust
use bonsai_trie_gpu::pedersen::pedersen_hash_batch;
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
cargo bench --features bench
```

### Expected Performance

| Batch Size | CPU (single-threaded) | GPU (RTX 4090) | Speedup |
|------------|----------------------|----------------|---------|
| 1          | ~13 µs               | ~50 µs         | 0.3×    |
| 1,000      | ~13 ms               | ~1 ms          | 13×     |
| 10,000     | ~130 ms              | ~5 ms          | 26×     |
| 100,000    | ~1.3 s               | ~25 ms         | 52×     |

GPU becomes beneficial for batch sizes > 1,024 hashes.

### Measured Performance (2026-02-04)

**Environment**
- Instance: Vast.ai (on-demand)
- OS: Ubuntu 22.04.4 LTS
- CPU: Intel Xeon E5-2680 v4 @ 2.40GHz (56 vCPUs)
- RAM: 62 GiB
- GPU: NVIDIA GeForce RTX 3060 (12 GB)
- Driver: 550.163.01
- CUDA Toolkit: 12.4

**CPU (single-threaded, `cargo bench --bench pedersen`, `CPU_BENCH_MAX=1000`)**

| Batch Size | Time (approx) | Throughput (approx) |
|------------|---------------|---------------------|
| 1          | 4.52 ms       | 221 elem/s          |
| 10         | 46.1 ms       | 217 elem/s          |
| 100        | 469 ms        | 213 elem/s          |
| 1,000      | 4.73 s        | 212 elem/s          |

`pedersen_hash_single`: ~178 µs

**GPU**

GPU benchmarking is currently blocked by a correctness mismatch in the CUDA field arithmetic; results will be added once the kernel passes differential tests.

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

## Roadmap

- [x] Phase 1: Field arithmetic module
- [x] Phase 2: Elliptic curve operations
- [x] Phase 3: Pedersen hash (CPU)
- [ ] Phase 4: CUDA kernel implementation
- [ ] Phase 5: Batch optimization with Pippenger's algorithm
- [ ] Phase 6: Integration with bonsai-trie

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## References

- [Starknet Cryptography Docs](https://docs.starknet.io/architecture/cryptography/)
- [StarkEx Pedersen Hash Specification](https://docs.starkware.co/starkex/crypto/pedersen-hash-function.html)
- [cuZK Paper](https://eprint.iacr.org/2022/1321) - GPU MSM optimization
