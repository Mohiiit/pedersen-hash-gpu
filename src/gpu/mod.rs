//! GPU acceleration layer for Pedersen hashing.
//!
//! This module provides CUDA-accelerated batch Pedersen hash computation.
//! It is only available when the `cuda` feature is enabled.
//!
//! # Architecture
//!
//! The GPU implementation uses:
//! - 4x64-bit limb representation for field elements
//! - Jacobian coordinates for elliptic curve points
//! - Batch processing with async memory transfers
//!
//! # Usage
//!
//! ```rust,ignore
//! use bonsai_trie_gpu::gpu::GpuPedersenHasher;
//!
//! let hasher = GpuPedersenHasher::new()?;
//! let results = hasher.hash_batch(&inputs_a, &inputs_b)?;
//! ```

#[cfg(feature = "cuda")]
mod context;
#[cfg(feature = "cuda")]
mod kernel;
#[cfg(feature = "cuda")]
mod hasher;

#[cfg(feature = "cuda")]
pub use context::GpuContext;
#[cfg(feature = "cuda")]
pub use hasher::GpuPedersenHasher;

/// Minimum batch size for GPU to be more efficient than CPU.
///
/// For smaller batches, CPU computation is typically faster due to
/// GPU memory transfer overhead.
pub const GPU_BATCH_THRESHOLD: usize = 1024;

/// Default CUDA block size for Pedersen hash kernel.
pub const DEFAULT_BLOCK_SIZE: u32 = 256;

/// Configuration for GPU computation.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Minimum batch size to use GPU (below this, fall back to CPU).
    pub batch_threshold: usize,

    /// CUDA block size for kernel launches.
    pub block_size: u32,

    /// Device ID to use (0 for first GPU).
    pub device_id: u32,

    /// Whether to use pinned (page-locked) memory for faster transfers.
    pub use_pinned_memory: bool,

    /// Whether to use multiple CUDA streams for overlapping.
    pub use_async_streams: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            batch_threshold: GPU_BATCH_THRESHOLD,
            block_size: DEFAULT_BLOCK_SIZE,
            device_id: 0,
            use_pinned_memory: true,
            use_async_streams: true,
        }
    }
}

/// Checks if CUDA is available on the system.
#[cfg(feature = "cuda")]
pub fn is_cuda_available() -> bool {
    cudarc::driver::CudaDevice::new(0).is_ok()
}

/// Checks if CUDA is available on the system.
#[cfg(not(feature = "cuda"))]
pub fn is_cuda_available() -> bool {
    false
}
