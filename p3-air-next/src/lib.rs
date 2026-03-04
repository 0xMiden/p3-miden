//! APIs for AIRs, and generalizations like PAIRs.

#![no_std]

extern crate alloc;

mod air;
pub mod symbolic;

pub use air::*;
pub use symbolic::*;
