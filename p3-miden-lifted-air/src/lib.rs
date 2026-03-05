//! AIR traits for the Miden lifted STARK protocol.
//!
//! This crate provides:
//! - [`LiftedAir`]: Super-trait for AIR definitions (inherits upstream + adds aux trace support)
//! - [`LiftedAirBuilder`]: Super-trait for constraint builders (blanket impl over upstream traits)
//! - [`PeriodicAirBuilder`] / [`AirWithPeriodicColumns`]: Periodic column support (copied from
//!   upstream, not yet in published p3-air 0.4.2)
//! - [`auxiliary`]: Auxiliary trace types (builder, cross-AIR identity checking)
//! - [`symbolic`]: Symbolic constraint analysis (expression trees, degree computation)

#![no_std]

extern crate alloc;

mod air;
pub mod auxiliary;
mod builder;
mod extension;
mod instance;
mod periodic;
pub mod symbolic;

pub use air::LiftedAir;
pub use auxiliary::{AuxBuilder, ReducedAuxValues, ReductionError, VarLenPublicInputs};
pub use builder::LiftedAirBuilder;
pub use extension::{ExtensionBuilder, PermutationAirBuilder};
pub use instance::{AirInstance, AirWitness, ValidationError, validate_instances};
pub use periodic::{AirWithPeriodicColumns, PeriodicAirBuilder};

// Re-export upstream traits for convenience so users don't need to depend on p3-air directly.
pub use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues,
    FilteredAirBuilder, PairBuilder,
};

// Re-export key symbolic types at crate root for ergonomic imports.
pub use symbolic::{
    ConstraintLayout, Entry, SymbolicAirBuilder, SymbolicExpression, SymbolicVariable,
    get_constraint_layout,
};

// Re-export commonly used field/matrix types.
pub use p3_field::{ExtensionField, Field};
pub use p3_matrix::dense::RowMajorMatrix;
