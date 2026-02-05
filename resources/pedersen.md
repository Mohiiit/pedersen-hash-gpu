# Pedersen Hash (Starknet) Basics

This project implements the Starknet-style Pedersen hash over the STARK-friendly elliptic curve. This section is enough to understand the math that the CUDA kernels are executing.

## 1) Pedersen hash formula (Starknet variant)

The hash takes two field elements `a` and `b`. Each is split into low and high bits, and used as scalars for fixed curve points:

```
H(a, b) = x( P0
            + a_low  * P1 + a_high * P2
            + b_low  * P3 + b_high * P4 )
```

- `a_low` is the low 248 bits, `a_high` is the high 4 bits.
- Same for `b`.
- The output is the **x-coordinate** of the resulting curve point.

## 2) The STARK curve

The hash uses a specific curve defined over the prime field:

- Field prime: p = 2^251 + 17*2^192 + 1
- Curve: y^2 = x^3 + x + beta

This is the curve used by StarkWare/STARK systems.

## 3) Why fixed points matter

Pedersen uses fixed points (P0..P4). That means we can precompute or structure kernels to optimize scalar multiplication with these constants, which is one reason GPU parallelization is viable.

## 4) How this maps to code in this repo

- `src/pedersen/` implements scalar splitting + point accumulation.
- `cuda/pedersen.cu` contains the CUDA kernels that batch the scalar multiplications.
- `src/gpu/` bridges Rust to CUDA and manages memory transfers.

## References

- StarkWare Pedersen hash definition
  https://docs.starkware.co/starkex/crypto/pedersen-hash-function
- StarkWare STARK curve parameters
  https://docs.starkware.co/starkex/crypto/stark-curve
