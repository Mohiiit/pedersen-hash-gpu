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
    0x551fde4050ca6804,
    0x716b0b1022947733,
    0x00ee1b87eb599f16,
    0x049ee3eba8c16007,
]);

/// Generator point P₀ (Shift point) - y coordinate
pub const PEDERSEN_P0_Y: FieldElement = FieldElement::from_limbs([
    0xd0405d266e10268a,
    0x4e621062c0e056c1,
    0xf346d49d06ea0ed3,
    0x03ca0cfe4b3bc6dd,
]);

/// Generator point P₁ - x coordinate (multiplies a_low, 248 bits)
pub const PEDERSEN_P1_X: FieldElement = FieldElement::from_limbs([
    0x1080d17957ebe47b,
    0x8fa8120b6d56eb0c,
    0x969c748655fca9e5,
    0x0234287dcbaffe7f,
]);

/// Generator point P₁ - y coordinate
pub const PEDERSEN_P1_Y: FieldElement = FieldElement::from_limbs([
    0x6ed0268ee89e5615,
    0x940135dd7a6c94cc,
    0x1e889527d41f4e39,
    0x03b056f100f96fb2,
]);

/// Generator point P₂ - x coordinate (multiplies a_high, 4 bits)
pub const PEDERSEN_P2_X: FieldElement = FieldElement::from_limbs([
    0xb7a6932dba8aa378,
    0x99099ec1de5e3018,
    0x3f9dab2656558f33,
    0x04fa56f376c83db3,
]);

/// Generator point P₂ - y coordinate
pub const PEDERSEN_P2_Y: FieldElement = FieldElement::from_limbs([
    0x5168f4e80ff5b54d,
    0x562761f92a7a23b4,
    0x8113e0c0e47e4401,
    0x03fa0984c931c9e3,
]);

/// Generator point P₃ - x coordinate (multiplies b_low, 248 bits)
pub const PEDERSEN_P3_X: FieldElement = FieldElement::from_limbs([
    0x3aa372f0bd2d6997,
    0x40c690c74709e90f,
    0x764910f75b45f74b,
    0x04ba4cc166be8dec,
]);

/// Generator point P₃ - y coordinate
pub const PEDERSEN_P3_Y: FieldElement = FieldElement::from_limbs([
    0x48151f27b24b219c,
    0xcac5c59a5ce5ae7c,
    0x4b971e46c4ede85f,
    0x0040301cf5c1751f,
]);

/// Generator point P₄ - x coordinate (multiplies b_high, 4 bits)
pub const PEDERSEN_P4_X: FieldElement = FieldElement::from_limbs([
    0xd36ff12c49a58202,
    0x2ca65048d53fb325,
    0x6e44cca8f61a63bb,
    0x054302dcb0e6cc1c,
]);

/// Generator point P₄ - y coordinate
pub const PEDERSEN_P4_Y: FieldElement = FieldElement::from_limbs([
    0x879dcc77e99c2426,
    0xce98ad783c25561a,
    0xb348046268d8ae25,
    0x01b77b3e37d13504,
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
