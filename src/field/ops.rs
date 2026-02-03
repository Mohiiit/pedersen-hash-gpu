//! Additional field operations for GPU-friendly computation.
//!
//! This module contains helper functions that will be mirrored in CUDA kernels.

use super::FieldElement;

/// Batch addition of field elements.
///
/// Computes a[i] + b[i] for all i.
pub fn batch_add(a: &[FieldElement], b: &[FieldElement]) -> alloc::vec::Vec<FieldElement> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}

/// Batch multiplication of field elements.
///
/// Computes a[i] * b[i] for all i.
pub fn batch_mul(a: &[FieldElement], b: &[FieldElement]) -> alloc::vec::Vec<FieldElement> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).collect()
}

/// Batch squaring of field elements.
pub fn batch_square(a: &[FieldElement]) -> alloc::vec::Vec<FieldElement> {
    a.iter().map(|x| x.square()).collect()
}

/// Batch inversion using Montgomery's trick.
///
/// Computes the inverse of all elements using only 3n-3 multiplications
/// and a single inversion, instead of n inversions.
pub fn batch_inverse(elements: &[FieldElement]) -> crate::error::Result<alloc::vec::Vec<FieldElement>> {
    use alloc::vec::Vec;

    if elements.is_empty() {
        return Ok(Vec::new());
    }

    let n = elements.len();

    // Check for zeros
    for (i, e) in elements.iter().enumerate() {
        if e.is_zero() {
            return Err(crate::error::Error::Other(
                alloc::format!("zero element at index {}", i)
            ));
        }
    }

    // Compute prefix products: products[i] = elements[0] * ... * elements[i]
    let mut products: Vec<FieldElement> = Vec::with_capacity(n);
    products.push(elements[0]);
    for i in 1..n {
        products.push(products[i - 1] * elements[i]);
    }

    // Compute inverse of the total product
    let mut inv_total = products[n - 1].inverse()?;

    // Compute inverses using the relation:
    // elements[i]^(-1) = products[i-1] * inv_suffix
    // where inv_suffix = (elements[i+1] * ... * elements[n-1])^(-1)
    let mut inverses: Vec<FieldElement> = alloc::vec![FieldElement::zero(); n];

    for i in (1..n).rev() {
        inverses[i] = products[i - 1] * inv_total;
        inv_total = inv_total * elements[i];
    }
    inverses[0] = inv_total;

    Ok(inverses)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_add() {
        let a = alloc::vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
        ];
        let b = alloc::vec![
            FieldElement::from(10u64),
            FieldElement::from(20u64),
            FieldElement::from(30u64),
        ];

        let result = batch_add(&a, &b);
        assert_eq!(result[0], FieldElement::from(11u64));
        assert_eq!(result[1], FieldElement::from(22u64));
        assert_eq!(result[2], FieldElement::from(33u64));
    }

    #[test]
    fn test_batch_inverse() {
        let elements = alloc::vec![
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(5u64),
            FieldElement::from(7u64),
        ];

        let inverses = batch_inverse(&elements).unwrap();

        // Verify each inverse
        for (e, inv) in elements.iter().zip(inverses.iter()) {
            let product = *e * *inv;
            assert!(product.is_one(), "Expected product to be 1, got {:?}", product);
        }
    }

    #[test]
    fn test_batch_inverse_with_zero() {
        let elements = alloc::vec![
            FieldElement::from(2u64),
            FieldElement::zero(),
            FieldElement::from(5u64),
        ];

        let result = batch_inverse(&elements);
        assert!(result.is_err());
    }
}
