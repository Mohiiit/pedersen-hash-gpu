//! # bonsai-trie-gpu
//!
//! GPU-accelerated Pedersen hashing for Starknet's bonsai-trie.
//!
//! This crate provides high-performance Pedersen hash computation using CUDA,
//! designed to accelerate Merkle tree operations in Starknet's state management.
//!
//! ## Features
//!
//! - **STARK Field Arithmetic**: Complete 251-bit prime field operations
//! - **Elliptic Curve Operations**: Point addition, doubling, and scalar multiplication
//! - **Pedersen Hash**: Both single-hash and batched implementations
//! - **GPU Acceleration**: CUDA kernels for parallel hash computation (optional)
//!
//! ## Quick Start
//!
//! ```rust
//! use bonsai_trie_gpu::pedersen::pedersen_hash;
//! use starknet_types_core::felt::Felt;
//!
//! let a = Felt::from(314u64);
//! let b = Felt::from(159u64);
//! let hash = pedersen_hash(&a, &b);
//! ```
//!
//! ## Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`field`]: STARK prime field arithmetic (F_p where p = 2^251 + 17·2^192 + 1)
//! - [`curve`]: Elliptic curve operations on the STARK curve
//! - [`pedersen`]: Pedersen hash implementation with generator points
//! - [`gpu`]: CUDA acceleration layer (requires `cuda` feature)

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod curve;
pub mod field;
pub mod pedersen;

#[cfg(feature = "cuda")]
pub mod gpu;

mod error;

pub use error::Error;
pub use field::FieldElement;
pub use curve::{AffinePoint, ProjectivePoint};
pub use pedersen::{pedersen_hash, pedersen_hash_array};

/// Re-export of Felt for convenience
pub use starknet_types_core::felt::Felt;
