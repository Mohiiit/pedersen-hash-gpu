//! Benchmarks for Pedersen hash implementations.
//!
//! Run with: cargo bench --features bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use starknet_types_core::felt::Felt;

use bonsai_trie_gpu::pedersen::{pedersen_hash, pedersen_hash_batch};

#[cfg(feature = "cuda")]
use bonsai_trie_gpu::gpu::GpuPedersenHasher;

/// Generate random test data for benchmarks.
fn generate_test_data(n: usize) -> Vec<(Felt, Felt)> {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();

    (0..n)
        .map(|_| {
            let a_bytes: [u8; 32] = rng.gen();
            let b_bytes: [u8; 32] = rng.gen();
            // Ensure values are in valid range by clearing top bits
            let mut a = a_bytes;
            let mut b = b_bytes;
            a[0] &= 0x07; // Clear top 5 bits to ensure < 2^251
            b[0] &= 0x07;
            (Felt::from_bytes_be(&a), Felt::from_bytes_be(&b))
        })
        .collect()
}

fn bench_single_hash(c: &mut Criterion) {
    let a = Felt::from(314u64);
    let b = Felt::from(159u64);

    c.bench_function("pedersen_hash_single", |bencher| {
        bencher.iter(|| pedersen_hash(black_box(&a), black_box(&b)))
    });
}

fn bench_batch_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("pedersen_hash_batch");

    for size in [1, 10, 100, 1000, 10000].iter() {
        let data = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("cpu", size), &data, |b, data| {
            b.iter(|| pedersen_hash_batch(black_box(data)))
        });
    }

    group.finish();
}

#[cfg(feature = "reference")]
fn bench_reference_comparison(c: &mut Criterion) {
    // Compare with starknet-crypto reference implementation
    let a = Felt::from(314u64);
    let b = Felt::from(159u64);

    let mut group = c.benchmark_group("pedersen_comparison");

    group.bench_function("our_impl", |bencher| {
        bencher.iter(|| pedersen_hash(black_box(&a), black_box(&b)))
    });

    // Note: starknet-crypto uses different types, conversion needed
    // This benchmark is a placeholder until we add proper conversion
    group.bench_function("reference_starknet_crypto", |bencher| {
        // Convert to starknet-crypto types
        let a_sc = starknet_crypto::Felt::from_raw([
            a.to_bytes_be()[24..32].try_into().map(u64::from_be_bytes).unwrap_or(0),
            a.to_bytes_be()[16..24].try_into().map(u64::from_be_bytes).unwrap_or(0),
            a.to_bytes_be()[8..16].try_into().map(u64::from_be_bytes).unwrap_or(0),
            a.to_bytes_be()[0..8].try_into().map(u64::from_be_bytes).unwrap_or(0),
        ]);
        let b_sc = starknet_crypto::Felt::from_raw([
            b.to_bytes_be()[24..32].try_into().map(u64::from_be_bytes).unwrap_or(0),
            b.to_bytes_be()[16..24].try_into().map(u64::from_be_bytes).unwrap_or(0),
            b.to_bytes_be()[8..16].try_into().map(u64::from_be_bytes).unwrap_or(0),
            b.to_bytes_be()[0..8].try_into().map(u64::from_be_bytes).unwrap_or(0),
        ]);

        bencher.iter(|| {
            starknet_crypto::pedersen_hash(black_box(&a_sc), black_box(&b_sc))
        })
    });

    group.finish();
}

#[cfg(feature = "cuda")]
fn bench_gpu_batch_sizes(c: &mut Criterion) {
    if !bonsai_trie_gpu::gpu::is_cuda_available() {
        return;
    }

    let hasher = GpuPedersenHasher::new().expect("failed to create GPU hasher");
    let mut group = c.benchmark_group("pedersen_hash_batch_gpu");

    for size in [1024usize, 4096, 16384].iter() {
        let data = generate_test_data(*size);

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("gpu", size), &data, |b, data| {
            b.iter(|| {
                hasher
                    .hash_batch_pairs(black_box(data))
                    .expect("GPU hash failed")
            })
        });
    }

    group.finish();
}

#[cfg(all(feature = "cuda", feature = "reference"))]
criterion_group!(
    benches,
    bench_single_hash,
    bench_batch_sizes,
    bench_reference_comparison,
    bench_gpu_batch_sizes,
);

#[cfg(all(feature = "cuda", not(feature = "reference")))]
criterion_group!(
    benches,
    bench_single_hash,
    bench_batch_sizes,
    bench_gpu_batch_sizes,
);

#[cfg(all(not(feature = "cuda"), feature = "reference"))]
criterion_group!(
    benches,
    bench_single_hash,
    bench_batch_sizes,
    bench_reference_comparison,
);

#[cfg(all(not(feature = "cuda"), not(feature = "reference")))]
criterion_group!(
    benches,
    bench_single_hash,
    bench_batch_sizes,
);
criterion_main!(benches);
