//! AIR traits for the Miden lifted STARK protocol.
//!
//! This crate provides:
//! - [`LiftedAir`]: Super-trait for AIR definitions (inherits upstream + adds aux trace support)
//! - [`LiftedAirBuilder`]: Super-trait for constraint builders (blanket impl over upstream traits)
//! - [`AirWithPeriodicColumns`]: Periodic column data trait (column periods and evaluations)
//! - [`auxiliary`]: Auxiliary trace types (builder, cross-AIR identity checking)
//!
//! Core traits ([`AirBuilder`], [`ExtensionBuilder`], [`PermutationAirBuilder`],
//! [`PeriodicAirBuilder`]) and symbolic types are re-exported from the `p3-air-next` fork.

#![no_std]

extern crate alloc;

mod air;
pub mod auxiliary;
mod builder;
mod instance;
mod periodic;

pub use air::LiftedAir;
pub use auxiliary::{AuxBuilder, ReducedAuxValues, ReductionError, VarLenPublicInputs};
pub use builder::LiftedAirBuilder;
pub use instance::{AirInstance, AirWitness, ValidationError, validate_instances};
pub use periodic::AirWithPeriodicColumns;

// Re-export upstream traits for convenience so users don't need to depend on p3-air directly.
pub use p3_air::{
    Air, AirBuilder, AirBuilderWithContext, BaseAir, ExtensionBuilder, FilteredAirBuilder,
    PeriodicAirBuilder, PermutationAirBuilder,
};

// Re-export fork symbolic types at crate root for ergonomic imports.
pub use p3_air::symbolic::{
    BaseEntry, ExtEntry, SymbolicAirBuilder, SymbolicExpression, SymbolicExpressionExt,
    SymbolicVariable, SymbolicVariableExt, get_all_symbolic_constraints, get_max_constraint_degree,
    get_max_constraint_degree_extension, get_symbolic_constraints,
    get_symbolic_constraints_extension,
};

// Re-export commonly used field/matrix types.
pub use p3_field::{ExtensionField, Field};
pub use p3_matrix::dense::RowMajorMatrix;
