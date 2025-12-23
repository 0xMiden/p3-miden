#![no_std]

extern crate alloc;

mod air;
#[cfg(debug_assertions)]
mod check_constraints;
mod config;
mod logup;
mod manual_lookup_gadget;
mod periodic_tables;
mod proof;
mod prover;
mod util;
mod verifier;

pub use air::{MidenAir, MidenAirBuilder};
#[cfg(debug_assertions)]
pub use check_constraints::*;
pub use config::*;
pub use logup::*;
pub use manual_lookup_gadget::*;
pub use proof::*;
pub use prover::*;
pub use verifier::*;

// Re-export symbolic types from p3-uni-stark
pub use p3_uni_stark::{
    Entry, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable,
    get_log_quotient_degree_extension, get_max_constraint_degree_extension,
    get_all_symbolic_constraints,
};

// Re-export commonly used p3-air traits for convenience
pub use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues,
    ExtensionBuilder, FilteredAirBuilder, PairBuilder, PermutationAirBuilder, PeriodicAirBuilder,
};
pub use p3_field::{ExtensionField, Field};
pub use p3_matrix::dense::RowMajorMatrix;
