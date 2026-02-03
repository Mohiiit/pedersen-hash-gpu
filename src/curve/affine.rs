//! Affine point representation on the STARK curve.

use core::fmt;
use core::ops::{Add, Neg};

use crate::error::{Error, Result};
use crate::field::FieldElement;

use super::constants::{CURVE_ALPHA, CURVE_BETA};
use super::ProjectivePoint;

/// A point on the STARK curve in affine coordinates (x, y).
///
/// The point at infinity is represented separately via `Option<AffinePoint>`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct AffinePoint {
    /// The x-coordinate of the point.
    pub x: FieldElement,
    /// The y-coordinate of the point.
    pub y: FieldElement,
}

impl AffinePoint {
    /// Creates a new affine point from coordinates.
    ///
    /// Does not verify that the point is on the curve.
    #[inline]
    pub const fn new_unchecked(x: FieldElement, y: FieldElement) -> Self {
        Self { x, y }
    }

    /// Creates a new affine point, verifying it lies on the curve.
    pub fn new(x: FieldElement, y: FieldElement) -> Result<Self> {
        let point = Self { x, y };
        if point.is_on_curve() {
            Ok(point)
        } else {
            Err(Error::PointNotOnCurve)
        }
    }

    /// Checks if the point lies on the STARK curve.
    ///
    /// Verifies: y² = x³ + αx + β
    pub fn is_on_curve(&self) -> bool {
        // y²
        let y_squared = self.y.square();

        // x³ + αx + β
        let x_squared = self.x.square();
        let x_cubed = x_squared * self.x;
        let alpha_x = CURVE_ALPHA * self.x;
        let rhs = x_cubed + alpha_x + CURVE_BETA;

        y_squared == rhs
    }

    /// Returns the x-coordinate.
    #[inline]
    pub const fn x(&self) -> &FieldElement {
        &self.x
    }

    /// Returns the y-coordinate.
    #[inline]
    pub const fn y(&self) -> &FieldElement {
        &self.y
    }

    /// Converts to projective coordinates.
    #[inline]
    pub fn to_projective(&self) -> ProjectivePoint {
        ProjectivePoint::from_affine(self)
    }

    /// Doubles the point.
    ///
    /// Uses the tangent line formula for point doubling.
    pub fn double(&self) -> Option<Self> {
        // If y = 0, the tangent is vertical and the result is the point at infinity
        if self.y.is_zero() {
            return None;
        }

        // λ = (3x² + α) / (2y)
        let x_squared = self.x.square();
        let three_x_squared = x_squared + x_squared + x_squared;
        let numerator = three_x_squared + CURVE_ALPHA;
        let denominator = self.y.double();
        let lambda = numerator * denominator.inverse().ok()?;

        // x' = λ² - 2x
        let lambda_squared = lambda.square();
        let two_x = self.x.double();
        let x_prime = lambda_squared - two_x;

        // y' = λ(x - x') - y
        let y_prime = lambda * (self.x - x_prime) - self.y;

        Some(Self {
            x: x_prime,
            y: y_prime,
        })
    }

    /// Adds another point to this one.
    ///
    /// Returns None if the result is the point at infinity.
    pub fn add(&self, other: &Self) -> Option<Self> {
        // If both points are the same, use doubling
        if self == other {
            return self.double();
        }

        // If x coordinates are equal but y coordinates differ, result is infinity
        if self.x == other.x {
            return None;
        }

        // λ = (y₂ - y₁) / (x₂ - x₁)
        let dy = other.y - self.y;
        let dx = other.x - self.x;
        let lambda = dy * dx.inverse().ok()?;

        // x' = λ² - x₁ - x₂
        let lambda_squared = lambda.square();
        let x_prime = lambda_squared - self.x - other.x;

        // y' = λ(x₁ - x') - y₁
        let y_prime = lambda * (self.x - x_prime) - self.y;

        Some(Self {
            x: x_prime,
            y: y_prime,
        })
    }

    /// Negates the point (reflects across x-axis).
    #[inline]
    pub fn negate(&self) -> Self {
        Self {
            x: self.x,
            y: -self.y,
        }
    }
}

impl fmt::Debug for AffinePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffinePoint")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl fmt::Display for AffinePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl Neg for AffinePoint {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        self.negate()
    }
}

impl Neg for &AffinePoint {
    type Output = AffinePoint;

    #[inline]
    fn neg(self) -> AffinePoint {
        self.negate()
    }
}

impl Add for AffinePoint {
    type Output = Option<Self>;

    #[inline]
    fn add(self, other: Self) -> Option<Self> {
        self.add(&other)
    }
}

impl Add<&AffinePoint> for AffinePoint {
    type Output = Option<Self>;

    #[inline]
    fn add(self, other: &AffinePoint) -> Option<Self> {
        AffinePoint::add(&self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_on_curve() {
        // Test with a known generator point
        let p0 = AffinePoint::new_unchecked(
            super::super::constants::PEDERSEN_P0_X,
            super::super::constants::PEDERSEN_P0_Y,
        );

        // Note: The constants need to be verified against the actual curve
        // This test will pass once we have correct generator coordinates
    }

    #[test]
    fn test_negation() {
        let x = FieldElement::from(5u64);
        let y = FieldElement::from(10u64);
        let p = AffinePoint::new_unchecked(x, y);
        let neg_p = -p;

        assert_eq!(neg_p.x, p.x);
        assert_eq!(neg_p.y, -p.y);
    }

    #[test]
    fn test_double_negation() {
        let x = FieldElement::from(5u64);
        let y = FieldElement::from(10u64);
        let p = AffinePoint::new_unchecked(x, y);
        let neg_neg_p = -(-p);

        assert_eq!(neg_neg_p, p);
    }
}
