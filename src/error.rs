//! Error types for bonsai-trie-gpu operations.

use alloc::string::String;
use core::fmt;

/// Errors that can occur during GPU-accelerated Pedersen hashing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Field element is not in the valid range [0, P)
    InvalidFieldElement,

    /// Point is not on the STARK curve
    PointNotOnCurve,

    /// Attempted to invert zero in the field
    DivisionByZero,

    /// Invalid hex string for parsing field elements
    InvalidHex(String),

    /// CUDA-related error (only with `cuda` feature)
    #[cfg(feature = "cuda")]
    CudaError(String),

    /// Batch size is invalid (e.g., empty or mismatched lengths)
    InvalidBatchSize,

    /// Generic error with message
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidFieldElement => {
                write!(f, "field element is not in valid range [0, P)")
            }
            Error::PointNotOnCurve => {
                write!(f, "point is not on the STARK curve")
            }
            Error::DivisionByZero => {
                write!(f, "attempted to divide by zero")
            }
            Error::InvalidHex(s) => {
                write!(f, "invalid hex string: {}", s)
            }
            #[cfg(feature = "cuda")]
            Error::CudaError(s) => {
                write!(f, "CUDA error: {}", s)
            }
            Error::InvalidBatchSize => {
                write!(f, "invalid batch size")
            }
            Error::Other(s) => {
                write!(f, "{}", s)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Result type alias for bonsai-trie-gpu operations.
pub type Result<T> = core::result::Result<T, Error>;
