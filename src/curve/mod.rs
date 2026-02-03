//! Elliptic curve operations on the STARK curve.
//!
//! The STARK curve is a Weierstrass curve over the STARK prime field:
//! y² = x³ + αx + β (mod P)
//!
//! where:
//! - α = 1
//! - β = 0x06f21413efbe40de150e596d72f7a8c5609ad26c15c915c1f4cdfcb99cee9e89
//!
//! This module provides:
//! - Affine point representation (x, y)
//! - Projective/Jacobian point representation (X, Y, Z) for efficient computation
//! - Point addition, doubling, and scalar multiplication

mod affine;
mod constants;
mod projective;
mod scalar_mul;

pub use affine::AffinePoint;
pub use constants::*;
pub use projective::ProjectivePoint;
pub use scalar_mul::scalar_mul;
