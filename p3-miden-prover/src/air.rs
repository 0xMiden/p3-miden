//! Miden AIR traits for defining constraint systems.
//!
//! This module provides unified super-traits for defining AIRs in the Miden VM ecosystem.
//! Instead of specifying all the individual trait bounds, users can use `MidenAir` and
//! `MidenAirBuilder` which bundle all necessary functionality.
//!
//! ## Traits
//!
//! - [`MidenAir`]: Blanket super-trait combining all AIR-side traits
//! - [`MidenAirBuilder`]: Blanket super-trait combining all builder-side traits
//!
//! ## Usage
//!
//! Users implement the upstream p3-air traits directly, and `MidenAir` is automatically
//! implemented via a blanket impl:
//!
//! ```rust,ignore
//! use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
//! use p3_miden_prover::MidenAirBuilder;
//!
//! struct MyAir;
//!
//! impl<F: Field> BaseAir<F> for MyAir {
//!     fn width(&self) -> usize { 2 }
//! }
//!
//! impl<F: Field> BaseAirWithPublicValues<F> for MyAir {}
//!
//! impl<AB: MidenAirBuilder> Air<AB> for MyAir {
//!     fn eval(&self, builder: &mut AB) {
//!         // constraints here
//!     }
//! }
//! ```

use p3_air::{
    Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, BaseAirWithPublicValues,
    ExtensionBuilder, PairBuilder, PeriodicAirBuilder, PermutationAirBuilder,
};
use p3_lookup::folder::{ProverConstraintFolderWithLookups, VerifierConstraintFolderWithLookups};
use p3_uni_stark::SymbolicAirBuilder;

use crate::{StarkGenericConfig, Val};

// ================================================================================================
// MidenAirBuilder
// ================================================================================================

/// Super-trait for all AIR builders in the Miden VM ecosystem.
///
/// This trait combines all upstream p3-air builder traits:
/// - [`AirBuilder`]: Core builder with main trace, assertions, and row selectors
/// - [`AirBuilderWithPublicValues`]: Access to public values
/// - [`PairBuilder`]: Access to preprocessed columns
/// - [`ExtensionBuilder`]: Extension field support
/// - [`PermutationAirBuilder`]: Permutation/LogUp support
/// - [`PeriodicAirBuilder`]: Periodic column support
///
/// Types implementing all the upstream traits automatically implement `MidenAirBuilder`
/// through a blanket implementation.
pub trait MidenAirBuilder:
    AirBuilder
    + AirBuilderWithPublicValues
    + PairBuilder
    + ExtensionBuilder
    + PermutationAirBuilder
    + PeriodicAirBuilder
{
}

impl<T> MidenAirBuilder for T where
    T: AirBuilder
        + AirBuilderWithPublicValues
        + PairBuilder
        + ExtensionBuilder
        + PermutationAirBuilder
        + PeriodicAirBuilder
{
}

// ================================================================================================
// MidenAir
// ================================================================================================

/// Super-trait for all AIR definitions in the Miden VM ecosystem.
///
/// This trait combines all upstream p3-air AIR-side traits:
/// - [`BaseAir`]: Trace width, preprocessed trace, and periodic columns
/// - [`BaseAirWithPublicValues`]: Number of public values
/// - [`Air`]: Constraint evaluation (for all standard p3 builder types)
/// - [`Sync`]: Required for parallel proving
///
/// Types implementing all the upstream traits automatically implement `MidenAir`
/// through a blanket implementation. Users should implement the individual traits
/// and get `MidenAir` for free.
///
/// # Example
///
/// ```rust,ignore
/// use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
/// use p3_miden_prover::{MidenAir, MidenAirBuilder};
/// use p3_field::Field;
///
/// struct FibonacciAir;
///
/// impl<F: Field> BaseAir<F> for FibonacciAir {
///     fn width(&self) -> usize { 2 }
/// }
///
/// impl<F: Field> BaseAirWithPublicValues<F> for FibonacciAir {
///     fn num_public_values(&self) -> usize { 3 }
/// }
///
/// impl<AB: MidenAirBuilder> Air<AB> for FibonacciAir {
///     fn eval(&self, builder: &mut AB) {
///         // Define constraints here
///     }
/// }
///
/// // FibonacciAir now automatically implements MidenAir<SC> for any SC
/// ```
pub trait MidenAir<SC: StarkGenericConfig>:
    BaseAir<Val<SC>>
    + BaseAirWithPublicValues<Val<SC>>
    + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
    + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
    + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>
    + Sync
{
}

impl<SC: StarkGenericConfig, T> MidenAir<SC> for T where
    T: BaseAir<Val<SC>>
        + BaseAirWithPublicValues<Val<SC>>
        + Air<SymbolicAirBuilder<Val<SC>, SC::Challenge>>
        + for<'a> Air<ProverConstraintFolderWithLookups<'a, SC>>
        + for<'a> Air<VerifierConstraintFolderWithLookups<'a, SC>>
        + Sync
{
}
