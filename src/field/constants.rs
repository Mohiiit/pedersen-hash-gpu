//! STARK field constants.
//!
//! The STARK prime field is defined over:
//! P = 2^251 + 17·2^192 + 1
//!   = 0x800000000000011000000000000000000000000000000000000000000000001
//!
//! This prime was chosen for its efficiency in both native and SNARK contexts.

use super::FieldElement;

/// The STARK prime: P = 2^251 + 17·2^192 + 1
///
/// Represented as 4 x 64-bit limbs in little-endian order.
/// limbs[0] is the least significant.
pub const STARK_PRIME: [u64; 4] = [
    0x0000000000000001, // limb 0 (least significant)
    0x0000000000000000, // limb 1
    0x0000000000000000, // limb 2
    0x0800000000000011, // limb 3 (most significant, 2^251 + 17*2^192 contributions)
];

/// The STARK prime as a hex string for reference.
pub const STARK_PRIME_HEX: &str =
    "0x800000000000011000000000000000000000000000000000000000000000001";

/// Number of bits in the STARK prime.
pub const STARK_PRIME_BITS: usize = 252;

/// Number of 64-bit limbs needed to represent a field element.
pub const NUM_LIMBS: usize = 4;

/// R = 2^256 mod P (Montgomery constant)
/// Used for converting to/from Montgomery form.
pub const MONTGOMERY_R: [u64; 4] = [
    0xffffffffffffffe1, // These values are computed as 2^256 mod P
    0xffffffffffffffff,
    0xffffffffffffffff,
    0x07fffffffffffdf0,
];

/// R^2 mod P (for Montgomery multiplication)
pub const MONTGOMERY_R2: [u64; 4] = [
    0xfffffd737e000401,
    0x00000001330fffff,
    0xffffffffff6f8000,
    0x07ffd4ab5e008810,
];

/// -P^(-1) mod 2^64 (Montgomery constant for reduction)
pub const MONTGOMERY_INV: u64 = 0xffffffffffffffff;

/// The curve order n (number of points on the STARK curve)
/// n = 0x0800000000000010ffffffffffffffffb781126dcae7b2321e66a241adc64d2f
pub const CURVE_ORDER: [u64; 4] = [
    0x1e66a241adc64d2f,
    0xb781126dcae7b232,
    0xffffffffffffffff,
    0x0800000000000010,
];

/// Zero element in the field.
pub const ZERO: FieldElement = FieldElement::from_limbs([0, 0, 0, 0]);

/// One element in the field.
pub const ONE: FieldElement = FieldElement::from_limbs([1, 0, 0, 0]);

/// Two element in the field.
pub const TWO: FieldElement = FieldElement::from_limbs([2, 0, 0, 0]);

/// Three element in the field.
pub const THREE: FieldElement = FieldElement::from_limbs([3, 0, 0, 0]);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_is_correct() {
        // Verify P = 2^251 + 17*2^192 + 1
        // 2^251 = 0x0800000000000000 << 192 bits = in limb 3
        // 17*2^192 = 0x11 << 128 bits = in limb 2
        // 1 = in limb 0

        // Check limb structure
        assert_eq!(STARK_PRIME[0], 1); // +1 term
        assert_eq!(STARK_PRIME[1], 0);
        assert_eq!(STARK_PRIME[2], 0); // limb 2 is zero
        assert_eq!(STARK_PRIME[3], 0x0800000000000011); // 2^251 + 17*2^192 terms
    }

    #[test]
    fn test_prime_bit_count() {
        // The most significant limb should have bits set for 2^251
        // 2^251 in a 64-bit context starting at bit 192 means
        // bit 251-192 = bit 59 should be set
        assert!(STARK_PRIME[3] >> 59 == 1);
        assert_eq!(STARK_PRIME_BITS, 252);
    }
}
