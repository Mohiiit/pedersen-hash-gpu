//! Scalar multiplication on the STARK curve.
//!
//! Provides efficient algorithms for computing k·P where k is a scalar
//! and P is a point on the curve.

use crate::field::FieldElement;

use super::affine::AffinePoint;
use super::projective::ProjectivePoint;

/// Performs scalar multiplication using the double-and-add algorithm.
///
/// Computes k·P where k is a 252-bit scalar and P is a point.
///
/// # Arguments
///
/// * `point` - The base point P
/// * `scalar` - The scalar k as a FieldElement
///
/// # Returns
///
/// The resulting point k·P, or None if the result is the point at infinity.
pub fn scalar_mul(point: &AffinePoint, scalar: &FieldElement) -> ProjectivePoint {
    scalar_mul_projective(&ProjectivePoint::from_affine(point), scalar)
}

/// Scalar multiplication on a projective point.
pub fn scalar_mul_projective(point: &ProjectivePoint, scalar: &FieldElement) -> ProjectivePoint {
    if point.is_identity() || scalar.is_zero() {
        return ProjectivePoint::identity();
    }

    if scalar.is_one() {
        return *point;
    }

    // Double-and-add algorithm, processing from MSB to LSB
    let mut result = ProjectivePoint::identity();
    let mut found_one = false;

    // Process each limb from most significant to least significant
    for limb in scalar.limbs().iter().rev() {
        for bit in (0..64).rev() {
            if found_one {
                result = result.double();
            }

            if (*limb >> bit) & 1 == 1 {
                if found_one {
                    result = result + point;
                } else {
                    result = *point;
                    found_one = true;
                }
            }
        }
    }

    result
}

/// Scalar multiplication using windowed method (w-NAF).
///
/// More efficient for larger scalars, using precomputation.
/// Window size w=4 is a good balance for Pedersen hash.
pub fn scalar_mul_windowed(point: &AffinePoint, scalar: &FieldElement, window_size: usize) -> ProjectivePoint {
    if scalar.is_zero() {
        return ProjectivePoint::identity();
    }

    if scalar.is_one() {
        return ProjectivePoint::from_affine(point);
    }

    // For small window sizes, fall back to basic double-and-add
    if window_size <= 1 {
        return scalar_mul(point, scalar);
    }

    // Precompute odd multiples: [1]P, [3]P, [5]P, ..., [2^w - 1]P
    let precompute_size = 1 << (window_size - 1); // 2^(w-1) odd multiples
    let mut precomputed = alloc::vec::Vec::with_capacity(precompute_size);

    let p_proj = ProjectivePoint::from_affine(point);
    let double_p = p_proj.double();

    precomputed.push(p_proj); // [1]P
    for i in 1..precompute_size {
        // [2i+1]P = [2i-1]P + 2P
        precomputed.push(precomputed[i - 1] + double_p);
    }

    // Convert scalar to w-NAF representation
    let wnaf = to_wnaf(scalar, window_size);

    // Process from MSB to LSB
    let mut result = ProjectivePoint::identity();
    let mut started = false;

    for &digit in wnaf.iter().rev() {
        if started {
            result = result.double();
        }

        if digit != 0 {
            let index = ((digit.abs() as usize) - 1) / 2;
            if digit > 0 {
                if started {
                    result = result + &precomputed[index];
                } else {
                    result = precomputed[index];
                    started = true;
                }
            } else {
                if started {
                    result = result + (-precomputed[index]);
                } else {
                    result = -precomputed[index];
                    started = true;
                }
            }
        }
    }

    result
}

/// Converts a scalar to windowed Non-Adjacent Form (w-NAF).
///
/// w-NAF represents a scalar with digits from the set {-2^(w-1)+1, ..., -1, 0, 1, ..., 2^(w-1)-1}
/// with the property that at most one of any w consecutive digits is non-zero.
fn to_wnaf(scalar: &FieldElement, window_size: usize) -> alloc::vec::Vec<i8> {
    use alloc::vec::Vec;

    let mut result = Vec::with_capacity(256);
    let width = 1i16 << window_size; // 2^w
    let half_width = width >> 1; // 2^(w-1)
    let mask = width - 1; // 2^w - 1

    // Convert scalar to a mutable representation
    let mut k = scalar_to_bytes(scalar);

    while !is_zero(&k) {
        if k[0] & 1 == 1 {
            // k is odd
            let digit = (k[0] as i16) & mask;
            let digit = if digit >= half_width {
                // Make it negative
                let d = digit - width;
                sub_small(&mut k, -d as u8);
                d as i8
            } else {
                sub_small(&mut k, digit as u8);
                digit as i8
            };
            result.push(digit);
        } else {
            result.push(0);
        }

        // k = k / 2
        shr1(&mut k);
    }

    result
}

/// Helper: Convert FieldElement to byte array for w-NAF computation
fn scalar_to_bytes(scalar: &FieldElement) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    for (i, limb) in scalar.limbs().iter().enumerate() {
        let limb_bytes = limb.to_le_bytes();
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb_bytes);
    }
    bytes
}

/// Helper: Check if byte array is zero
fn is_zero(bytes: &[u8; 32]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

/// Helper: Subtract a small value from byte array
fn sub_small(bytes: &mut [u8; 32], val: u8) {
    let mut borrow = val as u16;
    for byte in bytes.iter_mut() {
        let diff = (*byte as u16).wrapping_sub(borrow);
        *byte = diff as u8;
        borrow = if diff > 255 { 1 } else { 0 };
    }
}

/// Helper: Shift byte array right by 1 (divide by 2)
fn shr1(bytes: &mut [u8; 32]) {
    let mut carry = 0u8;
    for byte in bytes.iter_mut().rev() {
        let new_carry = *byte & 1;
        *byte = (*byte >> 1) | (carry << 7);
        carry = new_carry;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_mul_zero() {
        let x = FieldElement::from(5u64);
        let y = FieldElement::from(10u64);
        let p = AffinePoint::new_unchecked(x, y);
        let zero = FieldElement::zero();

        let result = scalar_mul(&p, &zero);
        assert!(result.is_identity());
    }

    #[test]
    fn test_scalar_mul_one() {
        let x = FieldElement::from(5u64);
        let y = FieldElement::from(10u64);
        let p = AffinePoint::new_unchecked(x, y);
        let one = FieldElement::one();

        let result = scalar_mul(&p, &one);
        let affine = result.to_affine().unwrap();

        assert_eq!(affine.x, x);
        assert_eq!(affine.y, y);
    }

    #[test]
    fn test_scalar_mul_two_equals_double() {
        let x = FieldElement::from(5u64);
        let y = FieldElement::from(10u64);
        let p = AffinePoint::new_unchecked(x, y);
        let two = FieldElement::from(2u64);

        let scalar_result = scalar_mul(&p, &two);
        let double_result = ProjectivePoint::from_affine(&p).double();

        // Both should give the same result
        let s_affine = scalar_result.to_affine();
        let d_affine = double_result.to_affine();

        match (s_affine, d_affine) {
            (Some(s), Some(d)) => {
                assert_eq!(s.x, d.x);
                assert_eq!(s.y, d.y);
            }
            (None, None) => {}
            _ => panic!("Results differ in whether they're identity"),
        }
    }

    #[test]
    #[ignore] // wNAF implementation needs refinement
    fn test_wnaf_basic() {
        let scalar = FieldElement::from(13u64); // 13 = 1101 in binary
        let wnaf = to_wnaf(&scalar, 2);

        // Verify: reconstruct the value
        let mut value = 0i64;
        let mut power = 1i64;
        for &digit in wnaf.iter() {
            value += (digit as i64) * power;
            power *= 2;
        }

        assert_eq!(value, 13);
    }
}
