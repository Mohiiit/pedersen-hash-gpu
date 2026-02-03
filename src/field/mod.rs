//! STARK prime field arithmetic.
//!
//! This module implements arithmetic operations over the STARK prime field F_p,
//! where p = 2^251 + 17·2^192 + 1.
//!
//! The field element representation uses a 4x64-bit limb layout for efficient
//! computation, with Montgomery form used for multiplication.

mod constants;
mod element;
mod ops;

pub use constants::*;
pub use element::FieldElement;
