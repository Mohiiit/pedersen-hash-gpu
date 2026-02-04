#![cfg(feature = "cuda")]

use cudarc::driver::{CudaDevice, DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use cudarc::nvrtc::Ptx;
use num_bigint::BigUint;
use num_traits::Num;
use std::sync::Arc;

use bonsai_trie_gpu::field::{FieldElement, MONTGOMERY_R, STARK_PRIME_HEX};

const MODULE_NAME: &str = "pedersen";
const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pedersen.ptx"));

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CudaFieldElement {
    limbs: [u64; 4],
}

unsafe impl DeviceRepr for CudaFieldElement {}
unsafe impl ValidAsZeroBits for CudaFieldElement {}

fn fe_to_biguint(fe: &FieldElement) -> BigUint {
    BigUint::from_bytes_be(&fe.to_bytes_be())
}

fn biguint_to_fe(val: &BigUint) -> FieldElement {
    let mut bytes = val.to_bytes_be();
    if bytes.len() > 32 {
        bytes = bytes[bytes.len() - 32..].to_vec();
    }
    if bytes.len() < 32 {
        let mut padded = vec![0u8; 32 - bytes.len()];
        padded.extend_from_slice(&bytes);
        bytes = padded;
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes[..32]);
    FieldElement::from_bytes_be(&arr).expect("valid field element")
}

fn prime_biguint() -> BigUint {
    BigUint::from_str_radix(STARK_PRIME_HEX.trim_start_matches("0x"), 16).unwrap()
}

fn mont_r_biguint() -> BigUint {
    BigUint::from_bytes_be(&FieldElement::from_limbs(MONTGOMERY_R).to_bytes_be())
}

fn to_hex(fe: &FieldElement) -> String {
    fe.to_bytes_be()
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<String>()
}

fn load_kernels(device: &Arc<CudaDevice>) {
    let ptx = Ptx::from_src(PTX);
    device
        .load_ptx(
            ptx,
            MODULE_NAME,
            &[
                "field_copy_batch",
                "field_to_mont_batch",
                "field_from_mont_batch",
                "field_mul_batch",
                "mul64hi_batch",
                "mul_add_batch",
                "field_mul_debug",
            ],
        )
        .expect("load ptx");
}

fn launch_cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (((n as u32) + 255) / 256, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[test]
#[ignore]
fn test_field_copy_kernel() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);
    let func = device.get_func(MODULE_NAME, "field_copy_batch").unwrap();

    let inputs = [
        FieldElement::zero(),
        FieldElement::one(),
        FieldElement::from_u64(123456789),
    ];
    let host: Vec<CudaFieldElement> = inputs
        .iter()
        .map(|f| CudaFieldElement { limbs: *f.limbs() })
        .collect();
    let n = host.len();
    let d_in = device.htod_copy(host.clone()).expect("copy in");
    let mut d_out = device.alloc_zeros::<CudaFieldElement>(n).expect("alloc out");

    unsafe {
        func.launch(launch_cfg(n), (&d_in, &mut d_out, n as i32))
            .expect("launch");
    }
    device.synchronize().expect("sync");
    let out = device.dtoh_sync_copy(&d_out).expect("dtoh");

    for (idx, output) in out.iter().enumerate() {
        assert_eq!(host[idx].limbs, output.limbs, "copy mismatch at {idx}");
    }
}

#[test]
#[ignore]
fn test_mul64hi_kernel() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);
    let func = device.get_func(MODULE_NAME, "mul64hi_batch").unwrap();

    let inputs_a: Vec<u64> = vec![
        0,
        1,
        2,
        0xffffffffffffffff,
        0x0123456789abcdef,
    ];
    let inputs_b: Vec<u64> = vec![
        0,
        1,
        3,
        0xffffffffffffffff,
        0xfedcba9876543210,
    ];
    let n = inputs_a.len();

    let d_a = device.htod_copy(inputs_a.clone()).expect("copy a");
    let d_b = device.htod_copy(inputs_b.clone()).expect("copy b");
    let mut d_lo = device.alloc_zeros::<u64>(n).expect("alloc lo");
    let mut d_hi = device.alloc_zeros::<u64>(n).expect("alloc hi");

    unsafe {
        func.launch(launch_cfg(n), (&d_a, &d_b, &mut d_lo, &mut d_hi, n as i32))
            .expect("launch");
    }
    device.synchronize().expect("sync");

    let out_lo = device.dtoh_sync_copy(&d_lo).expect("dtoh lo");
    let out_hi = device.dtoh_sync_copy(&d_hi).expect("dtoh hi");

    for idx in 0..n {
        let prod = (inputs_a[idx] as u128) * (inputs_b[idx] as u128);
        let exp_lo = prod as u64;
        let exp_hi = (prod >> 64) as u64;
        if out_lo[idx] != exp_lo || out_hi[idx] != exp_hi {
            eprintln!("mul64hi mismatch at {idx}");
            eprintln!("  a: 0x{:016x}", inputs_a[idx]);
            eprintln!("  b: 0x{:016x}", inputs_b[idx]);
            eprintln!("  expect lo/hi: 0x{:016x} / 0x{:016x}", exp_lo, exp_hi);
            eprintln!("  got    lo/hi: 0x{:016x} / 0x{:016x}", out_lo[idx], out_hi[idx]);
        }
        assert_eq!(out_lo[idx], exp_lo, "lo mismatch at {idx}");
        assert_eq!(out_hi[idx], exp_hi, "hi mismatch at {idx}");
    }
}

#[test]
#[ignore]
fn test_mul_add_kernel() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);
    let func = device.get_func(MODULE_NAME, "mul_add_batch").unwrap();

    let n = 256;
    let mut inputs_a = vec![0u64; n];
    let mut inputs_b = vec![0u64; n];
    let mut inputs_acc = vec![0u64; n];
    let mut inputs_carry = vec![0u64; n];

    let mut seed = 0x1234_5678_9abc_def0u64;
    for i in 0..n {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        inputs_a[i] = seed;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        inputs_b[i] = seed;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        inputs_acc[i] = seed;
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        inputs_carry[i] = seed;
    }

    let d_a = device.htod_copy(inputs_a.clone()).expect("copy a");
    let d_b = device.htod_copy(inputs_b.clone()).expect("copy b");
    let d_acc = device.htod_copy(inputs_acc.clone()).expect("copy acc");
    let d_carry = device.htod_copy(inputs_carry.clone()).expect("copy carry");
    let mut d_sum = device.alloc_zeros::<u64>(n).expect("alloc sum");
    let mut d_out_carry = device.alloc_zeros::<u64>(n).expect("alloc carry");

    unsafe {
        func.launch(
            launch_cfg(n),
            (&d_a, &d_b, &d_acc, &d_carry, &mut d_sum, &mut d_out_carry, n as i32),
        )
        .expect("launch");
    }
    device.synchronize().expect("sync");

    let out_sum = device.dtoh_sync_copy(&d_sum).expect("dtoh sum");
    let out_carry = device.dtoh_sync_copy(&d_out_carry).expect("dtoh carry");

    for i in 0..n {
        let prod = (inputs_a[i] as u128) * (inputs_b[i] as u128)
            + (inputs_acc[i] as u128)
            + (inputs_carry[i] as u128);
        let exp_sum = prod as u64;
        let exp_carry = (prod >> 64) as u64;
        if out_sum[i] != exp_sum || out_carry[i] != exp_carry {
            eprintln!("mul_add mismatch at {i}");
            eprintln!("  a: 0x{:016x}", inputs_a[i]);
            eprintln!("  b: 0x{:016x}", inputs_b[i]);
            eprintln!("  acc: 0x{:016x}", inputs_acc[i]);
            eprintln!("  carry: 0x{:016x}", inputs_carry[i]);
            eprintln!(
                "  expect sum/carry: 0x{:016x} / 0x{:016x}",
                exp_sum, exp_carry
            );
            eprintln!(
                "  got    sum/carry: 0x{:016x} / 0x{:016x}",
                out_sum[i], out_carry[i]
            );
        }
        assert_eq!(out_sum[i], exp_sum, "sum mismatch at {i}");
        assert_eq!(out_carry[i], exp_carry, "carry mismatch at {i}");
    }
}

fn mul_add_cpu(a: u64, b: u64, acc: u64, carry: u64) -> (u64, u64) {
    let prod = (a as u128) * (b as u128);
    let lo = prod as u64;
    let hi = (prod >> 64) as u64;
    let (sum1, c1) = acc.overflowing_add(lo);
    let (sum2, c2) = sum1.overflowing_add(carry);
    let carry_out = hi.wrapping_add(c1 as u64).wrapping_add(c2 as u64);
    (sum2, carry_out)
}

fn field_mul_debug_cpu(a: &FieldElement, b: &FieldElement) -> ([u64; 8], [u64; 32]) {
    let mut t = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u64;
        for j in 0..4 {
            let (acc, c) = mul_add_cpu(a.limbs()[i], b.limbs()[j], t[i + j], carry);
            t[i + j] = acc;
            carry = c;
        }
        let mut k = i + 4;
        while carry != 0 && k < 8 {
            let (acc, c) = t[k].overflowing_add(carry);
            t[k] = acc;
            carry = if c { 1 } else { 0 };
            k += 1;
        }
    }

    let t_mul = t;
    let mut t_reduce = [0u64; 32];
    for i in 0..4 {
        let m = (0u64).wrapping_sub(t[i]);
        let mut carry = 0u64;
        for j in 0..4 {
            let (acc, c) = mul_add_cpu(m, bonsai_trie_gpu::field::STARK_PRIME[j], t[i + j], carry);
            t[i + j] = acc;
            carry = c;
        }
        let mut k = i + 4;
        while carry != 0 && k < 8 {
            let (acc, c) = t[k].overflowing_add(carry);
            t[k] = acc;
            carry = if c { 1 } else { 0 };
            k += 1;
        }
        for j in 0..8 {
            t_reduce[i * 8 + j] = t[j];
        }
    }

    (t_mul, t_reduce)
}

#[test]
#[ignore]
fn test_field_mul_debug_kernel() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);
    let func = device.get_func(MODULE_NAME, "field_mul_debug").unwrap();

    let a = FieldElement::from_u64(1);
    let b = FieldElement::from_u64(1);
    let host_a = vec![CudaFieldElement { limbs: *a.limbs() }];
    let host_b = vec![CudaFieldElement { limbs: *b.limbs() }];

    let d_a = device.htod_copy(host_a).expect("copy a");
    let d_b = device.htod_copy(host_b).expect("copy b");
    let mut d_t_mul = device.alloc_zeros::<u64>(8).expect("alloc t_mul");
    let mut d_t_red = device.alloc_zeros::<u64>(32).expect("alloc t_red");

    unsafe {
        func.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 },
            (&d_a, &d_b, &mut d_t_mul, &mut d_t_red))
            .expect("launch");
    }
    device.synchronize().expect("sync");

    let t_mul_gpu = device.dtoh_sync_copy(&d_t_mul).expect("dtoh t_mul");
    let t_red_gpu = device.dtoh_sync_copy(&d_t_red).expect("dtoh t_red");

    let (t_mul_cpu, t_red_cpu) = field_mul_debug_cpu(&a, &b);

    for i in 0..8 {
        if t_mul_gpu[i] != t_mul_cpu[i] {
            eprintln!("t_mul mismatch at {i}: gpu=0x{:016x} cpu=0x{:016x}", t_mul_gpu[i], t_mul_cpu[i]);
        }
        assert_eq!(t_mul_gpu[i], t_mul_cpu[i], "t_mul mismatch at {i}");
    }
    for i in 0..32 {
        if t_red_gpu[i] != t_red_cpu[i] {
            eprintln!("t_red mismatch at {i}: gpu=0x{:016x} cpu=0x{:016x}", t_red_gpu[i], t_red_cpu[i]);
        }
        assert_eq!(t_red_gpu[i], t_red_cpu[i], "t_red mismatch at {i}");
    }
}

#[test]
#[ignore]
fn test_field_mul_debug_compare() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);
    let mul = device.get_func(MODULE_NAME, "field_mul_batch").unwrap();
    let dbg = device.get_func(MODULE_NAME, "field_mul_debug").unwrap();

    let a = FieldElement::from_u64(1);
    let b = FieldElement::from_u64(1);
    let host_a = vec![CudaFieldElement { limbs: *a.limbs() }];
    let host_b = vec![CudaFieldElement { limbs: *b.limbs() }];

    let d_a = device.htod_copy(host_a).expect("copy a");
    let d_b = device.htod_copy(host_b).expect("copy b");
    let mut d_out = device.alloc_zeros::<CudaFieldElement>(1).expect("alloc out");
    let mut d_t_mul = device.alloc_zeros::<u64>(8).expect("alloc t_mul");
    let mut d_t_red = device.alloc_zeros::<u64>(32).expect("alloc t_red");

    unsafe {
        mul.clone()
            .launch(launch_cfg(1), (&d_a, &d_b, &mut d_out, 1i32))
            .expect("mul");
        dbg.clone()
            .launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&d_a, &d_b, &mut d_t_mul, &mut d_t_red),
            )
            .expect("dbg");
    }
    device.synchronize().expect("sync");

    let out = device.dtoh_sync_copy(&d_out).expect("dtoh out");
    let t_red_gpu = device.dtoh_sync_copy(&d_t_red).expect("dtoh t_red");

    let last = 24usize;
    let v = FieldElement::from_limbs([
        t_red_gpu[last + 4],
        t_red_gpu[last + 5],
        t_red_gpu[last + 6],
        t_red_gpu[last + 7],
    ]);

    let prime = prime_biguint();
    let r = mont_r_biguint();
    let r_inv = r.modpow(&(prime.clone() - BigUint::from(2u64)), &prime);
    let prod = (fe_to_biguint(&a) * fe_to_biguint(&b)) % &prime;
    let expected = biguint_to_fe(&((prod * &r_inv) % &prime));

    let got = FieldElement::from_limbs(out[0].limbs);
    eprintln!("mul_batch: 0x{}", to_hex(&got));
    eprintln!("debug v  : 0x{}", to_hex(&v));
    eprintln!("expected : 0x{}", to_hex(&expected));

    assert_eq!(got, v, "batch output differs from debug reduction value");
    assert_eq!(got, expected, "batch output differs from expected");
}

#[test]
#[ignore]
fn test_field_to_mont_kernel() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);
    let func = device.get_func(MODULE_NAME, "field_to_mont_batch").unwrap();

    let prime = prime_biguint();
    let r = mont_r_biguint();

    let inputs = [
        FieldElement::zero(),
        FieldElement::one(),
        FieldElement::from_limbs([
            0xffffffffffffffff,
            0xffffffffffffffff,
            0xffffffffffffffff,
            0x07ffffffffffffff,
        ]),
    ];
    let expected: Vec<FieldElement> = inputs
        .iter()
        .map(|f| {
            let a = fe_to_biguint(f);
            let mont = (a * &r) % &prime;
            biguint_to_fe(&mont)
        })
        .collect();

    let host: Vec<CudaFieldElement> = inputs
        .iter()
        .map(|f| CudaFieldElement { limbs: *f.limbs() })
        .collect();
    let n = host.len();
    let d_in = device.htod_copy(host.clone()).expect("copy in");
    let mut d_out = device.alloc_zeros::<CudaFieldElement>(n).expect("alloc out");

    unsafe {
        func.launch(launch_cfg(n), (&d_in, &mut d_out, n as i32))
            .expect("launch");
    }
    device.synchronize().expect("sync");
    let out = device.dtoh_sync_copy(&d_out).expect("dtoh");

    for (idx, output) in out.iter().enumerate() {
        let got = FieldElement::from_limbs(output.limbs);
        if got != expected[idx] {
            eprintln!("to_mont mismatch at {idx}");
            eprintln!("  input:  0x{}", to_hex(&inputs[idx]));
            eprintln!("  expect: 0x{}", to_hex(&expected[idx]));
            eprintln!("  got:    0x{}", to_hex(&got));
        }
        assert_eq!(expected[idx], got, "to_mont mismatch at {idx}");
    }
}

#[test]
#[ignore]
fn test_field_from_mont_kernel() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);
    let func = device
        .get_func(MODULE_NAME, "field_from_mont_batch")
        .unwrap();

    let prime = prime_biguint();
    let r = mont_r_biguint();
    let r_inv = r.modpow(&(prime.clone() - BigUint::from(2u64)), &prime);

    let inputs_std = [
        FieldElement::zero(),
        FieldElement::one(),
        FieldElement::from_u64(123456789),
    ];
    let inputs_mont: Vec<FieldElement> = inputs_std
        .iter()
        .map(|f| {
            let a = fe_to_biguint(f);
            let mont = (a * &r) % &prime;
            biguint_to_fe(&mont)
        })
        .collect();
    let expected: Vec<FieldElement> = inputs_mont
        .iter()
        .map(|f| {
            let a = fe_to_biguint(f);
            let std = (a * &r_inv) % &prime;
            biguint_to_fe(&std)
        })
        .collect();

    let host: Vec<CudaFieldElement> = inputs_mont
        .iter()
        .map(|f| CudaFieldElement { limbs: *f.limbs() })
        .collect();
    let n = host.len();
    let d_in = device.htod_copy(host.clone()).expect("copy in");
    let mut d_out = device.alloc_zeros::<CudaFieldElement>(n).expect("alloc out");

    unsafe {
        func.launch(launch_cfg(n), (&d_in, &mut d_out, n as i32))
            .expect("launch");
    }
    device.synchronize().expect("sync");
    let out = device.dtoh_sync_copy(&d_out).expect("dtoh");

    for (idx, output) in out.iter().enumerate() {
        let got = FieldElement::from_limbs(output.limbs);
        if got != expected[idx] {
            eprintln!("from_mont mismatch at {idx}");
            eprintln!("  input:  0x{}", to_hex(&inputs_mont[idx]));
            eprintln!("  expect: 0x{}", to_hex(&expected[idx]));
            eprintln!("  got:    0x{}", to_hex(&got));
        }
        assert_eq!(expected[idx], got, "from_mont mismatch at {idx}");
    }
}

#[test]
#[ignore]
fn test_field_mul_roundtrip_kernel() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);

    let to_mont = device
        .get_func(MODULE_NAME, "field_to_mont_batch")
        .unwrap();
    let mul = device.get_func(MODULE_NAME, "field_mul_batch").unwrap();
    let from_mont = device
        .get_func(MODULE_NAME, "field_from_mont_batch")
        .unwrap();

    let prime = prime_biguint();
    let inputs_a = [
        FieldElement::from_u64(3),
        FieldElement::from_u64(7),
        FieldElement::from_u64(123456789),
    ];
    let inputs_b = [
        FieldElement::from_u64(11),
        FieldElement::from_u64(13),
        FieldElement::from_u64(987654321),
    ];

    let expected: Vec<FieldElement> = inputs_a
        .iter()
        .zip(inputs_b.iter())
        .map(|(a, b)| {
            let aa = fe_to_biguint(a);
            let bb = fe_to_biguint(b);
            let prod = (aa * bb) % &prime;
            biguint_to_fe(&prod)
        })
        .collect();

    let host_a: Vec<CudaFieldElement> = inputs_a
        .iter()
        .map(|f| CudaFieldElement { limbs: *f.limbs() })
        .collect();
    let host_b: Vec<CudaFieldElement> = inputs_b
        .iter()
        .map(|f| CudaFieldElement { limbs: *f.limbs() })
        .collect();
    let n = host_a.len();

    let d_a = device.htod_copy(host_a).expect("copy a");
    let d_b = device.htod_copy(host_b).expect("copy b");
    let mut d_a_m = device.alloc_zeros::<CudaFieldElement>(n).expect("alloc a_m");
    let mut d_b_m = device.alloc_zeros::<CudaFieldElement>(n).expect("alloc b_m");
    let mut d_prod_m = device.alloc_zeros::<CudaFieldElement>(n).expect("alloc prod_m");
    let mut d_prod = device.alloc_zeros::<CudaFieldElement>(n).expect("alloc prod");

    unsafe {
        to_mont
            .clone()
            .launch(launch_cfg(n), (&d_a, &mut d_a_m, n as i32))
            .expect("to_mont a");
        to_mont
            .clone()
            .launch(launch_cfg(n), (&d_b, &mut d_b_m, n as i32))
            .expect("to_mont b");
        mul.clone()
            .launch(launch_cfg(n), (&d_a_m, &d_b_m, &mut d_prod_m, n as i32))
            .expect("mul");
        from_mont
            .clone()
            .launch(launch_cfg(n), (&d_prod_m, &mut d_prod, n as i32))
            .expect("from_mont");
    }
    device.synchronize().expect("sync");

    let out = device.dtoh_sync_copy(&d_prod).expect("dtoh");
    let out_m = device.dtoh_sync_copy(&d_prod_m).expect("dtoh mont");
    let a_m = device.dtoh_sync_copy(&d_a_m).expect("dtoh a_m");
    let b_m = device.dtoh_sync_copy(&d_b_m).expect("dtoh b_m");

    for idx in 0..n {
        let got = FieldElement::from_limbs(out[idx].limbs);
        if got != expected[idx] {
            eprintln!("mul roundtrip mismatch at {idx}");
            eprintln!("  a: 0x{}", to_hex(&inputs_a[idx]));
            eprintln!("  b: 0x{}", to_hex(&inputs_b[idx]));
            eprintln!("  a_m: 0x{}", to_hex(&FieldElement::from_limbs(a_m[idx].limbs)));
            eprintln!("  b_m: 0x{}", to_hex(&FieldElement::from_limbs(b_m[idx].limbs)));
            eprintln!(
                "  prod_m: 0x{}",
                to_hex(&FieldElement::from_limbs(out_m[idx].limbs))
            );
            eprintln!("  expect: 0x{}", to_hex(&expected[idx]));
            eprintln!("  got:    0x{}", to_hex(&got));
        }
        assert_eq!(expected[idx], got, "mul roundtrip mismatch at {idx}");
    }
}

#[test]
#[ignore]
fn test_field_mul_standard_inputs() {
    let device = Arc::new(CudaDevice::new(0).expect("cuda device"));
    load_kernels(&device);
    let mul = device.get_func(MODULE_NAME, "field_mul_batch").unwrap();

    let prime = prime_biguint();
    let r = mont_r_biguint();
    let r_inv = r.modpow(&(prime.clone() - BigUint::from(2u64)), &prime);

    let inputs_a = [
        FieldElement::from_u64(1),
        FieldElement::from_u64(3),
        FieldElement::from_u64(123456789),
    ];
    let inputs_b = [
        FieldElement::from_u64(1),
        FieldElement::from_u64(7),
        FieldElement::from_u64(987654321),
    ];

    let expected: Vec<FieldElement> = inputs_a
        .iter()
        .zip(inputs_b.iter())
        .map(|(a, b)| {
            let aa = fe_to_biguint(a);
            let bb = fe_to_biguint(b);
            let prod = (aa * bb) % &prime;
            let mont = (prod * &r_inv) % &prime;
            biguint_to_fe(&mont)
        })
        .collect();

    let host_a: Vec<CudaFieldElement> = inputs_a
        .iter()
        .map(|f| CudaFieldElement { limbs: *f.limbs() })
        .collect();
    let host_b: Vec<CudaFieldElement> = inputs_b
        .iter()
        .map(|f| CudaFieldElement { limbs: *f.limbs() })
        .collect();
    let n = host_a.len();
    let d_a = device.htod_copy(host_a).expect("copy a");
    let d_b = device.htod_copy(host_b).expect("copy b");
    let mut d_out = device.alloc_zeros::<CudaFieldElement>(n).expect("alloc out");

    unsafe {
        mul.clone()
            .launch(launch_cfg(n), (&d_a, &d_b, &mut d_out, n as i32))
            .expect("mul");
    }
    device.synchronize().expect("sync");

    let out = device.dtoh_sync_copy(&d_out).expect("dtoh");
    for idx in 0..n {
        let got = FieldElement::from_limbs(out[idx].limbs);
        if got != expected[idx] {
            eprintln!("mul std mismatch at {idx}");
            eprintln!("  a: 0x{}", to_hex(&inputs_a[idx]));
            eprintln!("  b: 0x{}", to_hex(&inputs_b[idx]));
            eprintln!("  expect: 0x{}", to_hex(&expected[idx]));
            eprintln!("  got:    0x{}", to_hex(&got));
        }
        assert_eq!(expected[idx], got, "mul std mismatch at {idx}");
    }
}
