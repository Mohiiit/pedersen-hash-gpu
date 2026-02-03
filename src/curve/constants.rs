//! STARK curve constants.
//!
//! The STARK curve equation: y² = x³ + αx + β (mod P)

use crate::field::FieldElement;

/// Curve coefficient α = 1
pub const CURVE_ALPHA: FieldElement = FieldElement::from_limbs([1, 0, 0, 0]);

/// Curve coefficient β
/// β = 0x06f21413efbe40de150e596d72f7a8c5609ad26c15c915c1f4cdfcb99cee9e89
pub const CURVE_BETA: FieldElement = FieldElement::from_limbs([
    0xf4cdfcb99cee9e89,
    0x609ad26c15c915c1,
    0x150e596d72f7a8c5,
    0x06f21413efbe40de,
]);

/// Generator point P₀ (Shift point) - x coordinate
/// Used as the base point to prevent point-at-infinity in Pedersen hash
pub const PEDERSEN_P0_X: FieldElement = FieldElement::from_limbs([
    0x3551fde4050ca680,
    0x16716b0b10229477,
    0x0ee1b87eb599f167,
    0x049ee3eba8c16007,
]);

/// Generator point P₀ (Shift point) - y coordinate
pub const PEDERSEN_P0_Y: FieldElement = FieldElement::from_limbs([
    0x8b3f481e3aaa0f1a,
    0xc96b10228bf7b795,
    0x4759ebe3da9e1df0,
    0x06669b6c2df663da,
]);

/// Generator point P₁ - x coordinate (multiplies a_low, 248 bits)
pub const PEDERSEN_P1_X: FieldElement = FieldElement::from_limbs([
    0x969c748655fca9e5,
    0x7a6932dba8aa378,
    0xba8120b6d56eb0c1,
    0x0234287dcbaffe7f,
]);

/// Generator point P₁ - y coordinate
pub const PEDERSEN_P1_Y: FieldElement = FieldElement::from_limbs([
    0xfeb95e8eff0d707b,
    0x1df39439afdd4a4c,
    0xe1f1c8abe7dd2d8b,
    0x01ef15c18599971b,
]);

/// Generator point P₂ - x coordinate (multiplies a_high, 4 bits)
pub const PEDERSEN_P2_X: FieldElement = FieldElement::from_limbs([
    0x1080d17957ebe47b,
    0x8fa8120b6d56eb0c,
    0xf969c748655fca9e,
    0x04fa56f376c83db3,
]);

/// Generator point P₂ - y coordinate
pub const PEDERSEN_P2_Y: FieldElement = FieldElement::from_limbs([
    0x2b52c1f89a3fcdb2,
    0xb8ab29c7e43bd3e3,
    0x4a1e6bf4a0ce2bfc,
    0x06e2b2942e824b10,
]);

/// Generator point P₃ - x coordinate (multiplies b_low, 248 bits)
pub const PEDERSEN_P3_X: FieldElement = FieldElement::from_limbs([
    0xc74709e90f3aa372,
    0x9099ec1de5e3018b,
    0x64910f75b45f74b4,
    0x04ba4cc166be8dec,
]);

/// Generator point P₃ - y coordinate
pub const PEDERSEN_P3_Y: FieldElement = FieldElement::from_limbs([
    0x8ef46a7d9f1e6a77,
    0x3d45b15d62f1ea6b,
    0x0bd36b57a3e7c585,
    0x02ec3c9b4e7e5a53,
]);

/// Generator point P₄ - x coordinate (multiplies b_high, 4 bits)
pub const PEDERSEN_P4_X: FieldElement = FieldElement::from_limbs([
    0x53fb325d36ff12c4,
    0xca65048d53fb325d,
    0x0e6cc1c6e44cca8f,
    0x054302dcb0e6cc1c,
]);

/// Generator point P₄ - y coordinate
pub const PEDERSEN_P4_Y: FieldElement = FieldElement::from_limbs([
    0x3a6ff5b08a57a6dd,
    0x13a7cb2e24b8e1e6,
    0xe9e9b5e5a9b1a9f7,
    0x025bd4c8a7b9e6eb,
]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_is_one() {
        assert!(CURVE_ALPHA.is_one());
    }

    #[test]
    fn test_beta_is_valid() {
        assert!(CURVE_BETA.is_valid());
        assert!(!CURVE_BETA.is_zero());
    }

    #[test]
    fn test_generator_points_are_valid() {
        assert!(PEDERSEN_P0_X.is_valid());
        assert!(PEDERSEN_P0_Y.is_valid());
        assert!(PEDERSEN_P1_X.is_valid());
        assert!(PEDERSEN_P1_Y.is_valid());
    }
}
