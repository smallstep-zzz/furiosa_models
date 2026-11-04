use crate::kernel::constant_add_kernel::{constant_add_kernel, A};
use furiosa_opt_std::prelude::*;
use std::slice;

const INVALID_ARGUMENT: i32 = -1;
const RUNTIME_ERROR: i32 = -2;
const PANIC: i32 = -3;

/// Rust vISA proof kernel for exactly 2048 little-endian i32 values.
/// The C ABI is for the project's Python ctypes bridge.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rngd_add_one_i32(input: *const i32, output: *mut i32, len: usize) -> i32 {
    if input.is_null() || output.is_null() || len != 2048 {
        return INVALID_ARGUMENT;
    }
    std::panic::catch_unwind(|| {
        let values = unsafe { slice::from_raw_parts(input, len) }.to_vec();
        let runtime = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
            Ok(runtime) => runtime,
            Err(error) => {
                eprintln!("Rust vISA tokio runtime init failed: {error}");
                return RUNTIME_ERROR;
            }
        };
        let result = runtime.block_on(async {
            let mut device = Device::new(constant_add_kernel.topology())
                .map_err(|error| format!("Device::new: {error:?}"))?;
            let host = HostTensor::<i32, m![A]>::from_vec(values);
            let input_hbm = host.to_hbm(&mut device.pdma).await.map_err(|error| format!("to_hbm: {error:?}"))?;
            let output_hbm = launch(constant_add_kernel, (&mut device, &input_hbm))
                .await
                .map_err(|error| format!("launch: {error:?}"))?;
            let output_host = output_hbm.to_host::<m![A]>(&mut device.pdma)
                .await
                .map_err(|error| format!("to_host: {error:?}"))?;
            Ok::<Vec<i32>, String>(output_host.into_vec())
        });
        match result {
            Ok(values) => {
                unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), output, len) };
                0
            }
            Err(error) => {
                eprintln!("Rust vISA runtime failed: {error}");
                RUNTIME_ERROR
            }
        }
    }).unwrap_or(PANIC)
}
