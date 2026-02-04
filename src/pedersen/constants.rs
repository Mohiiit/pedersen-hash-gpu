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
/// X = 0x049ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804
/// Y = 0x03ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a
pub fn pedersen_p0() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x049ee3eba8c1600700ee1b87eb599f16716b0b1022947733551fde4050ca6804")
            .expect("P0.x is valid"),
        FieldElement::from_hex("0x03ca0cfe4b3bc6ddf346d49d06ea0ed34e621062c0e056c1d0405d266e10268a")
            .expect("P0.y is valid"),
    )
}

/// Generator point P₁ (multiplies a_low, 248 bits)
///
/// X = 0x0234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b
/// Y = 0x03b056f100f96fb21e889527d41f4e39940135dd7a6c94cc6ed0268ee89e5615
pub fn pedersen_p1() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x0234287dcbaffe7f969c748655fca9e58fa8120b6d56eb0c1080d17957ebe47b")
            .expect("P1.x is valid"),
        FieldElement::from_hex("0x03b056f100f96fb21e889527d41f4e39940135dd7a6c94cc6ed0268ee89e5615")
            .expect("P1.y is valid"),
    )
}

/// Generator point P₂ (multiplies a_high, 4 bits)
///
/// X = 0x04fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7a6932dba8aa378
/// Y = 0x03fa0984c931c9e38113e0b6ba5a2556eff35160c2c197e20f6f0ee12f3f2e66
pub fn pedersen_p2() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x04fa56f376c83db33f9dab2656558f3399099ec1de5e3018b7a6932dba8aa378")
            .expect("P2.x is valid"),
        FieldElement::from_hex("0x03fa0984c931c9e38113e0b6ba5a2556eff35160c2c197e20f6f0ee12f3f2e66")
            .expect("P2.y is valid"),
    )
}

/// Generator point P₃ (multiplies b_low, 248 bits)
///
/// X = 0x04ba4cc166be8dec764910f75b45f74b40c690c74709e90f3aa372f0bd2d6997
/// Y = 0x0040301cf5c1751f4b971e46c4ede85fcac5c59a5ce5ae7c48151f27b24b219c
pub fn pedersen_p3() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x04ba4cc166be8dec764910f75b45f74b40c690c74709e90f3aa372f0bd2d6997")
            .expect("P3.x is valid"),
        FieldElement::from_hex("0x0040301cf5c1751f4b971e46c4ede85fcac5c59a5ce5ae7c48151f27b24b219c")
            .expect("P3.y is valid"),
    )
}

/// Generator point P₄ (multiplies b_high, 4 bits)
///
/// X = 0x054302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202
/// Y = 0x01b77b3e37d13504bed68c4f1d6f6ca0f2f62fbd0e61cdd30491cf9039a72533
pub fn pedersen_p4() -> AffinePoint {
    AffinePoint::new_unchecked(
        FieldElement::from_hex("0x054302dcb0e6cc1c6e44cca8f61a63bb2ca65048d53fb325d36ff12c49a58202")
            .expect("P4.x is valid"),
        FieldElement::from_hex("0x01b77b3e37d13504bed68c4f1d6f6ca0f2f62fbd0e61cdd30491cf9039a72533")
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
