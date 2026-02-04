//! CUDA kernel loading and launch helpers.

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::vec::Vec;

use crate::error::{Error, Result};

const MODULE_NAME: &str = "pedersen";
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pedersen.ptx"));

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CudaFieldElement {
    pub limbs: [u64; 4],
}

unsafe impl DeviceRepr for CudaFieldElement {}
unsafe impl ValidAsZeroBits for CudaFieldElement {}

pub struct PedersenKernel;

impl PedersenKernel {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self> {
        let ptx = Ptx::from_src(PTX);
        device
            .load_ptx(ptx, MODULE_NAME, &["pedersen_hash_batch", "warmup_kernel"])
            .map_err(cuda_err)?;
        Ok(Self)
    }

    pub fn warmup(&self, device: &Arc<CudaDevice>) -> Result<()> {
        let func = device
            .get_func(MODULE_NAME, "warmup_kernel")
            .ok_or_else(|| Error::CudaError("missing warmup_kernel".to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut params: Vec<*mut core::ffi::c_void> = Vec::new();
            func.launch(cfg, &mut params)
                .map_err(cuda_err)?;
        }
        Ok(())
    }

    pub fn launch_batch(
        &self,
        device: &Arc<CudaDevice>,
        inputs_a: &CudaSlice<CudaFieldElement>,
        inputs_b: &CudaSlice<CudaFieldElement>,
        outputs: &mut CudaSlice<CudaFieldElement>,
        n: usize,
        block_size: u32,
    ) -> Result<()> {
        let grid_x = ((n as u32) + block_size - 1) / block_size;
        let func = device
            .get_func(MODULE_NAME, "pedersen_hash_batch")
            .ok_or_else(|| Error::CudaError("missing pedersen_hash_batch".to_string()))?;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(cfg, (inputs_a, inputs_b, outputs, n as i32))
                .map_err(cuda_err)?;
        }
        Ok(())
    }
}

fn cuda_err<E: core::fmt::Display>(err: E) -> Error {
    Error::CudaError(err.to_string())
}
