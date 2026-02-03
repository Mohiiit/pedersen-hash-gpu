//! Projective (Jacobian) point representation on the STARK curve.
//!
//! Jacobian coordinates represent a point (x, y) as (X, Y, Z) where:
//! - x = X / Z²
//! - y = Y / Z³
//!
//! This representation avoids costly field inversions during point operations,
//! requiring only a single inversion when converting back to affine form.

use core::fmt;
use core::ops::{Add, AddAssign, Neg};

use crate::field::FieldElement;

use super::affine::AffinePoint;
use super::constants::CURVE_ALPHA;

/// A point on the STARK curve in Jacobian projective coordinates.
///
/// Represents the point (X/Z², Y/Z³) in affine coordinates.
/// The point at infinity is represented by Z = 0.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ProjectivePoint {
    /// X coordinate (numerator for x)
    pub x: FieldElement,
    /// Y coordinate (numerator for y)
    pub y: FieldElement,
    /// Z coordinate (shared denominator factor)
    pub z: FieldElement,
}

impl ProjectivePoint {
    /// Creates a new projective point from coordinates.
    #[inline]
    pub const fn new(x: FieldElement, y: FieldElement, z: FieldElement) -> Self {
        Self { x, y, z }
    }

    /// Returns the point at infinity (identity element).
    #[inline]
    pub const fn identity() -> Self {
        Self {
            x: FieldElement::one(),
            y: FieldElement::one(),
            z: FieldElement::zero(),
        }
    }

    /// Checks if this is the point at infinity.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.z.is_zero()
    }

    /// Creates a projective point from an affine point.
    #[inline]
    pub fn from_affine(point: &AffinePoint) -> Self {
        Self {
            x: point.x,
            y: point.y,
            z: FieldElement::one(),
        }
    }

    /// Converts to affine coordinates.
    ///
    /// Returns None if this is the point at infinity.
    pub fn to_affine(&self) -> Option<AffinePoint> {
        if self.is_identity() {
            return None;
        }

        // x = X / Z²
        // y = Y / Z³
        let z_inv = self.z.inverse().ok()?;
        let z_inv_squared = z_inv.square();
        let z_inv_cubed = z_inv_squared * z_inv;

        let x = self.x * z_inv_squared;
        let y = self.y * z_inv_cubed;

        Some(AffinePoint::new_unchecked(x, y))
    }

    /// Returns only the x-coordinate in affine form.
    ///
    /// This is useful for Pedersen hash which only needs the x-coordinate.
    pub fn to_affine_x(&self) -> Option<FieldElement> {
        if self.is_identity() {
            return None;
        }

        let z_inv = self.z.inverse().ok()?;
        let z_inv_squared = z_inv.square();

        Some(self.x * z_inv_squared)
    }

    /// Doubles the point in Jacobian coordinates.
    ///
    /// Uses the formula for curves with a = 1:
    /// - S = 4·X·Y²
    /// - M = 3·X² + a·Z⁴ = 3·X² + Z⁴ (since a = 1)
    /// - X' = M² - 2·S
    /// - Y' = M·(S - X') - 8·Y⁴
    /// - Z' = 2·Y·Z
    pub fn double(&self) -> Self {
        if self.is_identity() {
            return *self;
        }

        // If Y = 0, the result is the identity
        if self.y.is_zero() {
            return Self::identity();
        }

        // Y² and Y⁴
        let y_squared = self.y.square();
        let y_fourth = y_squared.square();

        // S = 4·X·Y²
        let s = (self.x * y_squared).double().double();

        // Z² and Z⁴
        let z_squared = self.z.square();
        let z_fourth = z_squared.square();

        // M = 3·X² + a·Z⁴ (a = 1 for STARK curve)
        let x_squared = self.x.square();
        let three_x_squared = x_squared + x_squared + x_squared;
        let m = three_x_squared + CURVE_ALPHA * z_fourth;

        // X' = M² - 2·S
        let m_squared = m.square();
        let two_s = s.double();
        let x_prime = m_squared - two_s;

        // Y' = M·(S - X') - 8·Y⁴
        let eight_y_fourth = y_fourth.double().double().double();
        let y_prime = m * (s - x_prime) - eight_y_fourth;

        // Z' = 2·Y·Z
        let z_prime = (self.y * self.z).double();

        Self {
            x: x_prime,
            y: y_prime,
            z: z_prime,
        }
    }

    /// Adds an affine point to this projective point.
    ///
    /// This is more efficient than adding two projective points because
    /// the affine point has Z = 1.
    ///
    /// Uses the "add-2007-bl" formula from EFD.
    pub fn add_affine(&self, other: &AffinePoint) -> Self {
        if self.is_identity() {
            return Self::from_affine(other);
        }

        // Z1² and Z1³
        let z1_squared = self.z.square();
        let z1_cubed = z1_squared * self.z;

        // U2 = X2·Z1²
        let u2 = other.x * z1_squared;

        // S2 = Y2·Z1³
        let s2 = other.y * z1_cubed;

        // H = U2 - X1
        let h = u2 - self.x;

        // If H = 0, either doubling or identity
        if h.is_zero() {
            if (s2 - self.y).is_zero() {
                // Same point, double
                return self.double();
            } else {
                // Point and its negation, return identity
                return Self::identity();
            }
        }

        // R = S2 - Y1
        let r = s2 - self.y;

        // H², H³
        let h_squared = h.square();
        let h_cubed = h_squared * h;

        // X3 = R² - H³ - 2·X1·H²
        let x1_h_squared = self.x * h_squared;
        let x_prime = r.square() - h_cubed - x1_h_squared.double();

        // Y3 = R·(X1·H² - X3) - Y1·H³
        let y_prime = r * (x1_h_squared - x_prime) - self.y * h_cubed;

        // Z3 = Z1·H
        let z_prime = self.z * h;

        Self {
            x: x_prime,
            y: y_prime,
            z: z_prime,
        }
    }

    /// Adds two projective points.
    ///
    /// Uses the standard Jacobian addition formula.
    pub fn add(&self, other: &Self) -> Self {
        if self.is_identity() {
            return *other;
        }
        if other.is_identity() {
            return *self;
        }

        // Z1², Z1³, Z2², Z2³
        let z1_squared = self.z.square();
        let z1_cubed = z1_squared * self.z;
        let z2_squared = other.z.square();
        let z2_cubed = z2_squared * other.z;

        // U1 = X1·Z2², U2 = X2·Z1²
        let u1 = self.x * z2_squared;
        let u2 = other.x * z1_squared;

        // S1 = Y1·Z2³, S2 = Y2·Z1³
        let s1 = self.y * z2_cubed;
        let s2 = other.y * z1_cubed;

        // H = U2 - U1
        let h = u2 - u1;

        // If H = 0, either doubling or identity
        if h.is_zero() {
            if (s2 - s1).is_zero() {
                // Same point, double
                return self.double();
            } else {
                // Point and its negation, return identity
                return Self::identity();
            }
        }

        // R = S2 - S1
        let r = s2 - s1;

        // H², H³
        let h_squared = h.square();
        let h_cubed = h_squared * h;

        // X3 = R² - H³ - 2·U1·H²
        let u1_h_squared = u1 * h_squared;
        let x_prime = r.square() - h_cubed - u1_h_squared.double();

        // Y3 = R·(U1·H² - X3) - S1·H³
        let y_prime = r * (u1_h_squared - x_prime) - s1 * h_cubed;

        // Z3 = Z1·Z2·H
        let z_prime = self.z * other.z * h;

        Self {
            x: x_prime,
            y: y_prime,
            z: z_prime,
        }
    }

    /// Negates the point (reflects across x-axis).
    #[inline]
    pub fn negate(&self) -> Self {
        Self {
            x: self.x,
            y: -self.y,
            z: self.z,
        }
    }
}

impl fmt::Debug for ProjectivePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectivePoint")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}

impl Default for ProjectivePoint {
    fn default() -> Self {
        Self::identity()
    }
}

impl Neg for ProjectivePoint {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        self.negate()
    }
}

impl Neg for &ProjectivePoint {
    type Output = ProjectivePoint;

    #[inline]
    fn neg(self) -> ProjectivePoint {
        self.negate()
    }
}

impl Add for ProjectivePoint {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        ProjectivePoint::add(&self, &other)
    }
}

impl Add<&ProjectivePoint> for ProjectivePoint {
    type Output = Self;

    #[inline]
    fn add(self, other: &ProjectivePoint) -> Self {
        ProjectivePoint::add(&self, other)
    }
}

impl Add<&AffinePoint> for ProjectivePoint {
    type Output = Self;

    #[inline]
    fn add(self, other: &AffinePoint) -> Self {
        self.add_affine(other)
    }
}

impl AddAssign for ProjectivePoint {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl AddAssign<&AffinePoint> for ProjectivePoint {
    #[inline]
    fn add_assign(&mut self, other: &AffinePoint) {
        *self = self.add_affine(other);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = ProjectivePoint::identity();
        assert!(id.is_identity());
    }

    #[test]
    fn test_from_affine_round_trip() {
        let x = FieldElement::from(5u64);
        let y = FieldElement::from(10u64);
        let affine = AffinePoint::new_unchecked(x, y);

        let projective = ProjectivePoint::from_affine(&affine);
        let back = projective.to_affine().unwrap();

        assert_eq!(back.x, affine.x);
        assert_eq!(back.y, affine.y);
    }

    #[test]
    fn test_add_identity() {
        let x = FieldElement::from(5u64);
        let y = FieldElement::from(10u64);
        let p = ProjectivePoint::from_affine(&AffinePoint::new_unchecked(x, y));
        let id = ProjectivePoint::identity();

        // p + identity = p
        let result = p + id;
        let affine_result = result.to_affine().unwrap();
        assert_eq!(affine_result.x, x);
        assert_eq!(affine_result.y, y);

        // identity + p = p
        let result = id + p;
        let affine_result = result.to_affine().unwrap();
        assert_eq!(affine_result.x, x);
        assert_eq!(affine_result.y, y);
    }

    #[test]
    fn test_negation() {
        let x = FieldElement::from(5u64);
        let y = FieldElement::from(10u64);
        let p = ProjectivePoint::from_affine(&AffinePoint::new_unchecked(x, y));
        let neg_p = -p;

        let affine_neg = neg_p.to_affine().unwrap();
        assert_eq!(affine_neg.x, x);
        assert_eq!(affine_neg.y, -y);
    }
}
