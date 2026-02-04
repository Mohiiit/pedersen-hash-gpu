//! Pedersen hash computation.

use alloc::vec::Vec;

use starknet_types_core::felt::Felt;

use crate::curve::{scalar_mul, ProjectivePoint};
use crate::field::FieldElement;

use super::constants::{pedersen_p0, pedersen_p1, pedersen_p2, pedersen_p3, pedersen_p4, LOW_MASK};

/// Computes the Pedersen hash of two field elements.
///
/// The Pedersen hash is defined as:
/// ```text
/// H(a, b) = [P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄]_x
/// ```
///
/// # Arguments
///
/// * `a` - First input field element
/// * `b` - Second input field element
///
/// # Returns
///
/// The x-coordinate of the resulting elliptic curve point.
///
/// # Example
///
/// ```rust
/// use pedersen_hash_gpu::pedersen::pedersen_hash;
/// use starknet_types_core::felt::Felt;
///
/// let a = Felt::from(314u64);
/// let b = Felt::from(159u64);
/// let hash = pedersen_hash(&a, &b);
/// ```
pub fn pedersen_hash(a: &Felt, b: &Felt) -> Felt {
    // Convert Felt to our internal representation
    let a_bytes = a.to_bytes_be();
    let b_bytes = b.to_bytes_be();

    let a_fe = FieldElement::from_bytes_be(&a_bytes).expect("Felt should be valid");
    let b_fe = FieldElement::from_bytes_be(&b_bytes).expect("Felt should be valid");

    // Decompose inputs into low (248 bits) and high (4 bits) parts
    let (a_low, a_high) = decompose_felt(&a_fe);
    let (b_low, b_high) = decompose_felt(&b_fe);

    // Load generator points
    let p0 = pedersen_p0();
    let p1 = pedersen_p1();
    let p2 = pedersen_p2();
    let p3 = pedersen_p3();
    let p4 = pedersen_p4();

    // Compute: P₀ + a_low·P₁ + a_high·P₂ + b_low·P₃ + b_high·P₄
    let mut result = ProjectivePoint::from_affine(&p0);

    // Add a_low·P₁
    if !a_low.is_zero() {
        result = result + scalar_mul(&p1, &a_low);
    }

    // Add a_high·P₂
    if !a_high.is_zero() {
        result = result + scalar_mul(&p2, &a_high);
    }

    // Add b_low·P₃
    if !b_low.is_zero() {
        result = result + scalar_mul(&p3, &b_low);
    }

    // Add b_high·P₄
    if !b_high.is_zero() {
        result = result + scalar_mul(&p4, &b_high);
    }

    // Extract x-coordinate
    let x = result.to_affine_x().expect("Result should not be identity");

    // Convert back to Felt
    let x_bytes = x.to_bytes_be();
    Felt::from_bytes_be(&x_bytes)
}

/// Computes the Pedersen hash of an array of field elements.
///
/// This uses the standard chaining: hash(hash(hash(0, a[0]), a[1]), a[2])...
///
/// # Arguments
///
/// * `elements` - Slice of field elements to hash
///
/// # Returns
///
/// The final hash value.
pub fn pedersen_hash_array(elements: &[Felt]) -> Felt {
    let mut result = Felt::ZERO;
    for element in elements {
        result = pedersen_hash(&result, element);
    }
    // Final hash includes length
    pedersen_hash(&result, &Felt::from(elements.len() as u64))
}

/// Computes Pedersen hash for multiple (a, b) pairs.
///
/// This is a batch interface that can be optimized for GPU computation.
/// Currently uses sequential CPU computation; will be GPU-accelerated
/// when the `cuda` feature is enabled.
///
/// # Arguments
///
/// * `pairs` - Slice of (a, b) tuples to hash
///
/// # Returns
///
/// Vector of hash results, one for each input pair.
pub fn pedersen_hash_batch(pairs: &[(Felt, Felt)]) -> Vec<Felt> {
    #[cfg(feature = "cuda")]
    {
        if pairs.len() >= crate::gpu::GPU_BATCH_THRESHOLD && crate::gpu::is_cuda_available() {
            if let Ok(hasher) = crate::gpu::GpuPedersenHasher::new() {
                if let Ok(results) = hasher.hash_batch_pairs(pairs) {
                    return results;
                }
            }
        }
    }
    pairs.iter().map(|(a, b)| pedersen_hash(a, b)).collect()
}

/// Decomposes a 252-bit field element into low (248 bits) and high (4 bits) parts.
fn decompose_felt(fe: &FieldElement) -> (FieldElement, FieldElement) {
    let limbs = fe.limbs();

    // Low part: fe & LOW_MASK (lower 248 bits)
    let low_limbs = [
        limbs[0] & LOW_MASK[0],
        limbs[1] & LOW_MASK[1],
        limbs[2] & LOW_MASK[2],
        limbs[3] & LOW_MASK[3],
    ];

    // High part: fe >> 248 (upper 4 bits)
    // 248 = 3*64 + 56, so we need bits from limb[3] starting at bit 56
    let high_bits = limbs[3] >> 56; // This gives us the top bits

    (
        FieldElement::from_limbs(low_limbs),
        FieldElement::from_limbs([high_bits, 0, 0, 0]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_felt() {
        // Test with a small value (fits in low part)
        let small = FieldElement::from(12345u64);
        let (low, high) = decompose_felt(&small);

        assert_eq!(low, small);
        assert!(high.is_zero());
    }

    #[test]
    fn test_pedersen_hash_zero_zero() {
        let a = Felt::ZERO;
        let b = Felt::ZERO;
        let hash = pedersen_hash(&a, &b);

        // Hash of (0, 0) should be P₀.x
        // This is a known test vector
        assert!(!hash.eq(&Felt::ZERO)); // Result should not be zero
    }

    #[test]
    fn test_pedersen_hash_deterministic() {
        let a = Felt::from(314u64);
        let b = Felt::from(159u64);

        let hash1 = pedersen_hash(&a, &b);
        let hash2 = pedersen_hash(&a, &b);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_pedersen_hash_different_inputs() {
        let a1 = Felt::from(1u64);
        let b1 = Felt::from(2u64);

        let a2 = Felt::from(2u64);
        let b2 = Felt::from(1u64);

        let hash1 = pedersen_hash(&a1, &b1);
        let hash2 = pedersen_hash(&a2, &b2);

        // Different inputs should produce different hashes (with overwhelming probability)
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_pedersen_hash_array_empty() {
        let hash = pedersen_hash_array(&[]);
        // hash([]) = hash(0, 0)
        let expected = pedersen_hash(&Felt::ZERO, &Felt::ZERO);
        assert_eq!(hash, expected);
    }

    #[test]
    fn test_pedersen_hash_batch() {
        let pairs = vec![
            (Felt::from(1u64), Felt::from(2u64)),
            (Felt::from(3u64), Felt::from(4u64)),
            (Felt::from(5u64), Felt::from(6u64)),
        ];

        let batch_results = pedersen_hash_batch(&pairs);

        // Verify each result matches individual computation
        for (i, (a, b)) in pairs.iter().enumerate() {
            let individual = pedersen_hash(a, b);
            assert_eq!(batch_results[i], individual);
        }
    }
}
