//! CUDA device context handling.

use std::sync::Arc;

use cudarc::driver::CudaDevice;

use crate::error::{Error, Result};

/// GPU context for CUDA operations.
pub struct GpuContext {
    /// Shared CUDA device handle.
    pub device: Arc<CudaDevice>,
}

impl GpuContext {
    /// Create a new CUDA context for the given device ID.
    pub fn new(device_id: u32) -> Result<Self> {
        let device = CudaDevice::new(device_id as usize).map_err(cuda_err)?;
        Ok(Self {
            device: Arc::new(device),
        })
    }
}

fn cuda_err<E: core::fmt::Display>(err: E) -> Error {
    Error::CudaError(err.to_string())
}
