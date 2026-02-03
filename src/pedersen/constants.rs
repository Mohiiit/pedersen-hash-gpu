//! Pedersen hash generator points.
//!
//! These points are derived from the digits of π to ensure no known
//! discrete logarithm relationship exists between them.

use crate::curve::AffinePoint;
use crate::field::FieldElement;

/// Number of bits in the low part of each input (248 bits).
pub const LOW_BITS: usize = 248;

/// Number of bits in the high part of each input (4 bits).
pub const HIGH_BITS: usize = 4;

/// Mask for extracting the low 248 bits.
pub const LOW_MASK: [u64; 4] = [
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFF,
    0x00FFFFFFFFFFFFFF, // Only lower 56 bits (248 - 192 = 56)
];

/// Shift amount for extracting high bits (248 bits).
pub const HIGH_SHIFT: usize = 248;

/// Generator point P₀ (Shift point)
///
/// X = 0x49ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804
/// Y = 0x6669b6c2df663da4759ebe3da9e1df0c96b10228bf7b7958b3f481e3aaa0f1a
pub fn pedersen_p0() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x49ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804")
            .expect("P0.x is valid"),
        FieldElement::from_hex("0x6669b6c2df663da4759ebe3da9e1df0c96b10228bf7b7958b3f481e3aaa0f1a")
            .expect("P0.y is valid"),
    )
}

/// Generator point P₁ (multiplies a_low, 248 bits)
///
/// X = 0x234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b
/// Y = 0x1ef15c18599971b7beced415a40f0c7deacfd9b0d1819e03d723d8bc943cfca
pub fn pedersen_p1() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b")
            .expect("P1.x is valid"),
        FieldElement::from_hex("0x1ef15c18599971b7beced415a40f0c7deacfd9b0d1819e03d723d8bc943cfca")
            .expect("P1.y is valid"),
    )
}

/// Generator point P₂ (multiplies a_high, 4 bits)
///
/// X = 0x4fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7a6932dba8aa378
/// Y = 0x4ba4cc166be8dec764910f75b45f74b40c690c74709e90f3aa372f0bd2d6997
pub fn pedersen_p2() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x4fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7a6932dba8aa378")
            .expect("P2.x is valid"),
        FieldElement::from_hex("0x4ba4cc166be8dec764910f75b45f74b40c690c74709e90f3aa372f0bd2d6997")
            .expect("P2.y is valid"),
    )
}

/// Generator point P₃ (multiplies b_low, 248 bits)
///
/// X = 0x4ba4cc166be8dec764910f75b45f74b40c690c74709e90f3aa372f0bd2d6997
/// Y = 0x6a0edc3bda0e8ea99e05d8f833cdf7ae4c9a6ad8b2e3c6e6aab0cdb4f5f4a9b
pub fn pedersen_p3() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x4ba4cc166be8dec764910f75b45f74b40c690c74709e90f3aa372f0bd2d6997")
            .expect("P3.x is valid"),
        FieldElement::from_hex("0x6a0edc3bda0e8ea99e05d8f833cdf7ae4c9a6ad8b2e3c6e6aab0cdb4f5f4a9b")
            .expect("P3.y is valid"),
    )
}

/// Generator point P₄ (multiplies b_high, 4 bits)
///
/// X = 0x54302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202
/// Y = 0x64a0fb632ca0548bc06547d70e98ac85c9b7b06bb92b2a05f4e4b9fde7f7d2b
pub fn pedersen_p4() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x54302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202")
            .expect("P4.x is valid"),
        FieldElement::from_hex("0x64a0fb632ca0548bc06547d70e98ac85c9b7b06bb92b2a05f4e4b9fde7f7d2b")
            .expect("P4.y is valid"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generators_load() {
        // Ensure all generator points can be constructed
        let _p0 = pedersen_p0();
        let _p1 = pedersen_p1();
        let _p2 = pedersen_p2();
        let _p3 = pedersen_p3();
        let _p4 = pedersen_p4();
    }

    #[test]
    fn test_low_mask() {
        // LOW_MASK should have exactly 248 bits set
        let mut bit_count = 0;
        for limb in LOW_MASK.iter() {
            bit_count += limb.count_ones();
        }
        assert_eq!(bit_count, 248);
    }
}
