//! High-level GPU Pedersen hashing interface.

use crate::error::{Error, Result};
use crate::field::FieldElement;
use crate::gpu::{GpuConfig, GpuContext};
use crate::pedersen::pedersen_hash;
use crate::Felt;

use super::kernel::{CudaFieldElement, PedersenKernel};

/// GPU-backed Pedersen hasher.
pub struct GpuPedersenHasher {
    context: GpuContext,
    kernel: PedersenKernel,
    config: GpuConfig,
}

impl GpuPedersenHasher {
    /// Create a hasher with default configuration.
    pub fn new() -> Result<Self> {
        Self::new_with_config(GpuConfig::default())
    }

    /// Create a hasher with a specific configuration.
    pub fn new_with_config(config: GpuConfig) -> Result<Self> {
        let context = GpuContext::new(config.device_id)?;
        let kernel = PedersenKernel::load(&context.device)?;
        kernel.warmup(&context.device)?;
        Ok(Self {
            context,
            kernel,
            config,
        })
    }

    /// Hash a batch of input pairs (a, b).
    pub fn hash_batch_pairs(&self, pairs: &[(Felt, Felt)]) -> Result<Vec<Felt>> {
        if pairs.is_empty() {
            return Err(Error::InvalidBatchSize);
        }

        let mut inputs_a = Vec::with_capacity(pairs.len());
        let mut inputs_b = Vec::with_capacity(pairs.len());
        for (a, b) in pairs.iter() {
            inputs_a.push(*a);
            inputs_b.push(*b);
        }

        self.hash_batch(&inputs_a, &inputs_b)
    }

    /// Hash a batch of inputs with separate a and b arrays.
    pub fn hash_batch(&self, inputs_a: &[Felt], inputs_b: &[Felt]) -> Result<Vec<Felt>> {
        if inputs_a.len() != inputs_b.len() || inputs_a.is_empty() {
            return Err(Error::InvalidBatchSize);
        }

        let n = inputs_a.len();
        let mut host_a = Vec::with_capacity(n);
        let mut host_b = Vec::with_capacity(n);

        for (a, b) in inputs_a.iter().zip(inputs_b.iter()) {
            host_a.push(CudaFieldElement::from_field(&felt_to_field(a)?));
            host_b.push(CudaFieldElement::from_field(&felt_to_field(b)?));
        }

        let device = &self.context.device;
        let d_a = device.htod_copy(host_a.as_slice()).map_err(cuda_err)?;
        let d_b = device.htod_copy(host_b.as_slice()).map_err(cuda_err)?;
        let mut d_out = device
            .alloc_zeros::<CudaFieldElement>(n)
            .map_err(cuda_err)?;

        self.kernel
            .launch_batch(device, &d_a, &d_b, &mut d_out, n, self.config.block_size)?;

        device.synchronize().map_err(cuda_err)?;
        let out_host: Vec<CudaFieldElement> = device.dtoh_sync_copy(&d_out).map_err(cuda_err)?;

        let mut results = Vec::with_capacity(n);
        for out in out_host.iter() {
            let field = out.to_field();
            let bytes = field.to_bytes_be();
            results.push(Felt::from_bytes_be(&bytes));
        }

        Ok(results)
    }

    /// CPU fallback for small batches or when GPU is unavailable.
    pub fn hash_batch_cpu(pairs: &[(Felt, Felt)]) -> Vec<Felt> {
        pairs.iter().map(|(a, b)| pedersen_hash(a, b)).collect()
    }
}

impl CudaFieldElement {
    fn from_field(field: &FieldElement) -> Self {
        Self {
            limbs: *field.limbs(),
        }
    }

    fn to_field(&self) -> FieldElement {
        FieldElement::from_limbs(self.limbs)
    }
}

fn felt_to_field(felt: &Felt) -> Result<FieldElement> {
    let bytes = felt.to_bytes_be();
    FieldElement::from_bytes_be(&bytes).map_err(|_| Error::InvalidFieldElement)
}

fn cuda_err<E: core::fmt::Display>(err: E) -> Error {
    Error::CudaError(err.to_string())
}
