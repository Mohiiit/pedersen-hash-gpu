//! CUDA kernel loading and launch helpers.

use cudarc::driver::{CudaDevice, CudaSlice, DeviceRepr, LaunchAsync};

use crate::error::{Error, Result};

const MODULE_NAME: &str = "pedersen";
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pedersen.ptx"));

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CudaFieldElement {
    pub limbs: [u64; 4],
}

unsafe impl DeviceRepr for CudaFieldElement {}

pub struct PedersenKernel;

impl PedersenKernel {
    pub fn load(device: &CudaDevice) -> Result<Self> {
        device
            .load_ptx(PTX, MODULE_NAME, &["pedersen_hash_batch", "warmup_kernel"])
            .map_err(cuda_err)?;
        Ok(Self)
    }

    pub fn warmup(&self, device: &CudaDevice) -> Result<()> {
        let func = device.get_func(MODULE_NAME, "warmup_kernel").map_err(cuda_err)?;
        unsafe {
            func.launch([1, 1, 1], [1, 1, 1], ())
                .map_err(cuda_err)?;
        }
        Ok(())
    }

    pub fn launch_batch(
        &self,
        device: &CudaDevice,
        inputs_a: &CudaSlice<CudaFieldElement>,
        inputs_b: &CudaSlice<CudaFieldElement>,
        outputs: &mut CudaSlice<CudaFieldElement>,
        n: usize,
        block_size: u32,
    ) -> Result<()> {
        let grid_x = ((n as u32) + block_size - 1) / block_size;
        let func = device.get_func(MODULE_NAME, "pedersen_hash_batch").map_err(cuda_err)?;
        unsafe {
            func.launch(
                [grid_x, 1, 1],
                [block_size, 1, 1],
                (inputs_a, inputs_b, outputs, n as i32),
            )
            .map_err(cuda_err)?;
        }
        Ok(())
    }
}

fn cuda_err<E: core::fmt::Display>(err: E) -> Error {
    Error::CudaError(err.to_string())
}
