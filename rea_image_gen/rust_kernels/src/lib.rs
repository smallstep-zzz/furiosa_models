#![expect(clippy::type_complexity)]
#![feature(proc_macro_hygiene)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]

pub mod kernel;
pub mod python_ffi;
