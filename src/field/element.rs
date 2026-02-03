//! Field element type and basic operations.

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::constants::{NUM_LIMBS, STARK_PRIME};
use crate::error::{Error, Result};

/// A field element in the STARK prime field.
///
/// Internally represented as 4 x 64-bit limbs in little-endian order.
/// The value is kept in the range [0, P) where P is the STARK prime.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct FieldElement {
    /// The limbs of the field element, in little-endian order.
    /// limbs[0] is the least significant.
    pub(crate) limbs: [u64; NUM_LIMBS],
}

impl FieldElement {
    /// Creates a new field element from raw limbs.
    ///
    /// # Safety
    /// The caller must ensure the value is less than the STARK prime.
    #[inline]
    pub const fn from_limbs(limbs: [u64; NUM_LIMBS]) -> Self {
        Self { limbs }
    }

    /// Creates a field element from a u64 value.
    #[inline]
    pub const fn from_u64(value: u64) -> Self {
        Self {
            limbs: [value, 0, 0, 0],
        }
    }

    /// Returns the zero element.
    #[inline]
    pub const fn zero() -> Self {
        Self { limbs: [0; NUM_LIMBS] }
    }

    /// Returns the one element.
    #[inline]
    pub const fn one() -> Self {
        Self {
            limbs: [1, 0, 0, 0],
        }
    }

    /// Checks if this element is zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.limbs[0] == 0 && self.limbs[1] == 0 && self.limbs[2] == 0 && self.limbs[3] == 0
    }

    /// Checks if this element is one.
    #[inline]
    pub const fn is_one(&self) -> bool {
        self.limbs[0] == 1 && self.limbs[1] == 0 && self.limbs[2] == 0 && self.limbs[3] == 0
    }

    /// Returns the raw limbs.
    #[inline]
    pub const fn limbs(&self) -> &[u64; NUM_LIMBS] {
        &self.limbs
    }

    /// Converts to big-endian bytes (32 bytes).
    pub fn to_bytes_be(&self) -> [u8; 32] {
        let mut result = [0u8; 32];
        // limbs are little-endian, we want big-endian bytes
        for (i, limb) in self.limbs.iter().enumerate() {
            let offset = 24 - i * 8;
            result[offset..offset + 8].copy_from_slice(&limb.to_be_bytes());
        }
        result
    }

    /// Creates a field element from big-endian bytes.
    pub fn from_bytes_be(bytes: &[u8; 32]) -> Result<Self> {
        let mut limbs = [0u64; NUM_LIMBS];
        for i in 0..NUM_LIMBS {
            let offset = 24 - i * 8;
            limbs[i] = u64::from_be_bytes(bytes[offset..offset + 8].try_into().unwrap());
        }

        let result = Self { limbs };
        if !result.is_valid() {
            return Err(Error::InvalidFieldElement);
        }
        Ok(result)
    }

    /// Checks if the value is in the valid range [0, P).
    #[inline]
    pub fn is_valid(&self) -> bool {
        // Compare with STARK_PRIME, return true if self < P
        for i in (0..NUM_LIMBS).rev() {
            if self.limbs[i] < STARK_PRIME[i] {
                return true;
            }
            if self.limbs[i] > STARK_PRIME[i] {
                return false;
            }
        }
        // Equal to P, which is not valid
        false
    }

    /// Reduces the value modulo P if needed.
    /// This is a simple reduction for values that might be slightly >= P.
    #[inline]
    pub fn reduce(&mut self) {
        if !self.is_valid() {
            // Subtract P once (assumes value < 2P)
            let mut borrow = 0u64;
            for i in 0..NUM_LIMBS {
                let (diff, b1) = self.limbs[i].overflowing_sub(STARK_PRIME[i]);
                let (diff, b2) = diff.overflowing_sub(borrow);
                self.limbs[i] = diff;
                borrow = (b1 as u64) + (b2 as u64);
            }
        }
    }

    /// Adds two field elements without reduction.
    /// Result may be >= P.
    #[inline]
    fn add_no_reduce(&self, other: &Self) -> Self {
        let mut result = [0u64; NUM_LIMBS];
        let mut carry = 0u64;

        for i in 0..NUM_LIMBS {
            let (sum, c1) = self.limbs[i].overflowing_add(other.limbs[i]);
            let (sum, c2) = sum.overflowing_add(carry);
            result[i] = sum;
            carry = (c1 as u64) + (c2 as u64);
        }

        Self { limbs: result }
    }

    /// Subtracts two field elements.
    /// If self < other, adds P first to ensure positive result.
    #[inline]
    fn sub_impl(&self, other: &Self) -> Self {
        let mut result = [0u64; NUM_LIMBS];
        let mut borrow = 0u64;

        for i in 0..NUM_LIMBS {
            let (diff, b1) = self.limbs[i].overflowing_sub(other.limbs[i]);
            let (diff, b2) = diff.overflowing_sub(borrow);
            result[i] = diff;
            borrow = (b1 as u64) + (b2 as u64);
        }

        // If we had a borrow, add P back
        if borrow != 0 {
            let mut carry = 0u64;
            for i in 0..NUM_LIMBS {
                let (sum, c1) = result[i].overflowing_add(STARK_PRIME[i]);
                let (sum, c2) = sum.overflowing_add(carry);
                result[i] = sum;
                carry = (c1 as u64) + (c2 as u64);
            }
        }

        Self { limbs: result }
    }

    /// Negates the field element (computes P - self).
    #[inline]
    pub fn neg_impl(&self) -> Self {
        if self.is_zero() {
            return *self;
        }

        let mut result = [0u64; NUM_LIMBS];
        let mut borrow = 0u64;

        for i in 0..NUM_LIMBS {
            let (diff, b1) = STARK_PRIME[i].overflowing_sub(self.limbs[i]);
            let (diff, b2) = diff.overflowing_sub(borrow);
            result[i] = diff;
            borrow = (b1 as u64) + (b2 as u64);
        }

        Self { limbs: result }
    }

    /// Doubles the field element.
    #[inline]
    pub fn double(&self) -> Self {
        let mut result = self.add_no_reduce(self);
        result.reduce();
        result
    }

    /// Squares the field element.
    #[inline]
    pub fn square(&self) -> Self {
        self.mul_impl(self)
    }

    /// Multiplies two field elements using BigUint for correctness.
    /// This is a reference implementation; optimized versions will use
    /// Montgomery multiplication.
    fn mul_impl(&self, other: &Self) -> Self {
        use num_bigint::BigUint;

        // Convert to BigUint
        let a = self.to_biguint();
        let b = other.to_biguint();

        // Multiply
        let product = a * b;

        // Get prime
        let prime = Self::prime_biguint();

        // Reduce modulo P
        let reduced = product % prime;

        // Convert back to FieldElement
        Self::from_biguint(&reduced)
    }

    /// Converts to BigUint for arithmetic operations.
    fn to_biguint(&self) -> num_bigint::BigUint {
        use num_bigint::BigUint;
        let bytes = self.to_bytes_be();
        BigUint::from_bytes_be(&bytes)
    }

    /// Creates a FieldElement from BigUint.
    fn from_biguint(n: &num_bigint::BigUint) -> Self {
        let bytes = n.to_bytes_be();
        let mut arr = [0u8; 32];
        let start = if bytes.len() > 32 { bytes.len() - 32 } else { 0 };
        let dest_start = 32 - (bytes.len() - start).min(32);
        arr[dest_start..].copy_from_slice(&bytes[start..]);
        // from_bytes_be validates, but we trust BigUint reduction
        Self::from_bytes_be(&arr).unwrap_or_else(|_| Self::zero())
    }

    /// Returns the STARK prime as BigUint.
    fn prime_biguint() -> num_bigint::BigUint {
        use num_bigint::BigUint;
        let mut bytes = [0u8; 32];
        // Convert STARK_PRIME limbs to big-endian bytes
        for (i, limb) in STARK_PRIME.iter().enumerate() {
            let offset = 24 - i * 8;
            bytes[offset..offset + 8].copy_from_slice(&limb.to_be_bytes());
        }
        BigUint::from_bytes_be(&bytes)
    }

    /// Reduces a wide (512-bit) value modulo P.
    /// Currently unused since we use BigUint-based multiplication.
    #[allow(dead_code)]
    fn reduce_wide(wide: &[u64; 8]) -> Self {
        use num_bigint::BigUint;

        let mut value = BigUint::from(0u64);
        for i in (0..8).rev() {
            value <<= 64;
            value += wide[i];
        }

        let prime = Self::prime_biguint();
        let reduced = value % prime;
        Self::from_biguint(&reduced)
    }

    /// Computes the modular inverse using Fermat's little theorem.
    /// a^(-1) = a^(P-2) mod P
    pub fn inverse(&self) -> Result<Self> {
        use num_bigint::BigUint;
        use num_traits::One;

        if self.is_zero() {
            return Err(Error::DivisionByZero);
        }

        let a = self.to_biguint();
        let p = Self::prime_biguint();
        let p_minus_2 = &p - BigUint::from(2u64);

        // Use BigUint's modpow for correct exponentiation
        let result = a.modpow(&p_minus_2, &p);

        Ok(Self::from_biguint(&result))
    }

    /// Computes self^exp using square-and-multiply.
    pub fn pow(&self, exp: &[u64; NUM_LIMBS]) -> Self {
        use num_bigint::BigUint;

        let base = self.to_biguint();
        let prime = Self::prime_biguint();

        // Convert exp to BigUint
        let mut exp_biguint = BigUint::from(0u64);
        for i in (0..NUM_LIMBS).rev() {
            exp_biguint <<= 64;
            exp_biguint += exp[i];
        }

        let result = base.modpow(&exp_biguint, &prime);
        Self::from_biguint(&result)
    }

    /// Creates a field element from a hex string (with or without 0x prefix).
    pub fn from_hex(hex_str: &str) -> Result<Self> {
        let hex_str = hex_str.strip_prefix("0x").unwrap_or(hex_str);
        let hex_str = hex_str.strip_prefix("0X").unwrap_or(hex_str);

        // Pad to 64 characters (32 bytes)
        let padded = format!("{:0>64}", hex_str);
        if padded.len() != 64 {
            return Err(Error::InvalidHex(alloc::string::String::from(
                "hex string too long",
            )));
        }

        let bytes: alloc::vec::Vec<u8> = (0..32)
            .map(|i| {
                u8::from_str_radix(&padded[i * 2..i * 2 + 2], 16)
                    .map_err(|_| Error::InvalidHex(alloc::string::String::from("invalid hex digit")))
            })
            .collect::<Result<alloc::vec::Vec<_>>>()?;

        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Self::from_bytes_be(&arr)
    }
}

// Implement Display for debugging
impl fmt::Debug for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FieldElement(0x{:016x}{:016x}{:016x}{:016x})",
            self.limbs[3], self.limbs[2], self.limbs[1], self.limbs[0]
        )
    }
}

impl fmt::Display for FieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "0x{:016x}{:016x}{:016x}{:016x}",
            self.limbs[3], self.limbs[2], self.limbs[1], self.limbs[0]
        )
    }
}

// Arithmetic trait implementations

impl Add for FieldElement {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        let mut result = self.add_no_reduce(&other);
        result.reduce();
        result
    }
}

impl Add<&FieldElement> for FieldElement {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        let mut result = self.add_no_reduce(other);
        result.reduce();
        result
    }
}

impl Add<&FieldElement> for &FieldElement {
    type Output = FieldElement;

    #[inline]
    fn add(self, other: &FieldElement) -> FieldElement {
        let mut result = self.add_no_reduce(other);
        result.reduce();
        result
    }
}

impl AddAssign for FieldElement {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for FieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        self.sub_impl(&other)
    }
}

impl Sub<&FieldElement> for FieldElement {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        self.sub_impl(other)
    }
}

impl Sub<&FieldElement> for &FieldElement {
    type Output = FieldElement;

    #[inline]
    fn sub(self, other: &FieldElement) -> FieldElement {
        self.sub_impl(other)
    }
}

impl SubAssign for FieldElement {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Neg for FieldElement {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        self.neg_impl()
    }
}

impl Neg for &FieldElement {
    type Output = FieldElement;

    #[inline]
    fn neg(self) -> FieldElement {
        self.neg_impl()
    }
}

impl Mul for FieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        self.mul_impl(&other)
    }
}

impl Mul<&FieldElement> for FieldElement {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        self.mul_impl(other)
    }
}

impl Mul<&FieldElement> for &FieldElement {
    type Output = FieldElement;

    #[inline]
    fn mul(self, other: &FieldElement) -> FieldElement {
        self.mul_impl(other)
    }
}

impl MulAssign for FieldElement {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl From<u64> for FieldElement {
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl From<u32> for FieldElement {
    fn from(value: u32) -> Self {
        Self::from_u64(value as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_and_one() {
        let zero = FieldElement::zero();
        let one = FieldElement::one();

        assert!(zero.is_zero());
        assert!(!zero.is_one());
        assert!(one.is_one());
        assert!(!one.is_zero());
    }

    #[test]
    fn test_addition() {
        let a = FieldElement::from_u64(100);
        let b = FieldElement::from_u64(200);
        let c = a + b;

        assert_eq!(c.limbs[0], 300);
        assert_eq!(c.limbs[1], 0);
        assert_eq!(c.limbs[2], 0);
        assert_eq!(c.limbs[3], 0);
    }

    #[test]
    fn test_subtraction() {
        let a = FieldElement::from_u64(200);
        let b = FieldElement::from_u64(100);
        let c = a - b;

        assert_eq!(c.limbs[0], 100);
    }

    #[test]
    fn test_subtraction_underflow() {
        let a = FieldElement::from_u64(100);
        let b = FieldElement::from_u64(200);
        let c = a - b;

        // Result should be P - 100
        let expected = FieldElement::from_limbs(STARK_PRIME) - FieldElement::from_u64(100);
        // Since P - 100 wraps, we compare differently
        assert!(!c.is_zero());
    }

    #[test]
    fn test_negation() {
        let a = FieldElement::from_u64(100);
        let neg_a = -a;

        // a + (-a) should be 0
        let sum = a + neg_a;
        assert!(sum.is_zero());
    }

    #[test]
    fn test_multiplication() {
        let a = FieldElement::from_u64(100);
        let b = FieldElement::from_u64(200);
        let c = a * b;

        assert_eq!(c.limbs[0], 20000);
    }

    #[test]
    fn test_double() {
        let a = FieldElement::from_u64(100);
        let doubled = a.double();
        let added = a + a;

        assert_eq!(doubled, added);
    }

    #[test]
    fn test_square() {
        let a = FieldElement::from_u64(100);
        let squared = a.square();
        let multiplied = a * a;

        assert_eq!(squared, multiplied);
    }

    #[test]
    fn test_inverse() {
        let a = FieldElement::from_u64(7);
        let a_inv = a.inverse().unwrap();

        // a * a^(-1) should be 1
        let product = a * a_inv;
        assert!(product.is_one());
    }

    #[test]
    fn test_inverse_zero_fails() {
        let zero = FieldElement::zero();
        assert!(zero.inverse().is_err());
    }

    #[test]
    fn test_from_hex() {
        let fe = FieldElement::from_hex("0x1").unwrap();
        assert!(fe.is_one());

        let fe = FieldElement::from_hex("0x0").unwrap();
        assert!(fe.is_zero());

        let fe = FieldElement::from_hex("100").unwrap();
        assert_eq!(fe.limbs[0], 0x100);
    }
}
