//! Pedersen hash implementation for Starknet.
//!
//! The Pedersen hash function is defined as:
//!
//! ```text
//! H(a, b) = [P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄]_x
//! ```
//!
//! Where:
//! - P₀, P₁, P₂, P₃, P₄ are generator points on the STARK curve
//! - a_low = a & ((1 << 248) - 1) (lower 248 bits of a)
//! - a_high = a >> 248 (upper 4 bits of a)
//! - b_low, b_high are defined similarly
//! - [P]_x denotes the x-coordinate of point P

mod constants;
mod hash;

pub use constants::*;
pub use hash::{pedersen_hash, pedersen_hash_array, pedersen_hash_batch};
