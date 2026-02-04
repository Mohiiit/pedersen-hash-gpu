#![cfg(feature = "cuda")]

use pedersen_hash_gpu::gpu::GpuPedersenHasher;
use pedersen_hash_gpu::pedersen::pedersen_hash;
use starknet_types_core::felt::Felt;

use rand::prelude::*;

fn generate_pairs(n: usize) -> Vec<(Felt, Felt)> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| {
            let mut a_bytes: [u8; 32] = rng.gen();
            let mut b_bytes: [u8; 32] = rng.gen();
            // Ensure values are in valid range by clearing top bits
            a_bytes[0] &= 0x07;
            b_bytes[0] &= 0x07;
            (Felt::from_bytes_be(&a_bytes), Felt::from_bytes_be(&b_bytes))
        })
        .collect()
}

#[test]
#[ignore]
fn test_gpu_matches_cpu_small_batch() {
    let pairs = generate_pairs(256);
    let hasher = GpuPedersenHasher::new().expect("failed to create GPU hasher");

    let gpu = hasher
        .hash_batch_pairs(&pairs)
        .expect("GPU hash failed");
    let cpu: Vec<Felt> = pairs.iter().map(|(a, b)| pedersen_hash(a, b)).collect();

    assert_eq!(gpu, cpu);
}
