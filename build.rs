use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let src = PathBuf::from("cuda/pedersen.cu");
    let ptx = out_dir.join("pedersen.ptx");

    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());

    let mut cmd = Command::new(nvcc);
    cmd.arg("-ptx")
        .arg("-O3")
        .arg("-std=c++14")
        .arg(&src)
        .arg("-o")
        .arg(&ptx);

    if let Ok(arch) = env::var("CUDA_ARCH") {
        cmd.arg("-arch").arg(arch);
    }

    let status = cmd.status().expect("failed to invoke nvcc");
    if !status.success() {
        panic!("nvcc failed to compile cuda/pedersen.cu");
    }

    println!("cargo:rerun-if-changed=cuda/pedersen.cu");
}
