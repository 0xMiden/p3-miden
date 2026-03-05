//! The `LiftedAirBuilder` super-trait for constraint evaluation builders.

use p3_air::{AirBuilder, AirBuilderWithPublicValues};

use crate::{ExtensionBuilder, PeriodicAirBuilder, PermutationAirBuilder};

/// Super-trait bundling all builder capabilities needed by the lifted STARK system.
///
/// Instead of re-declaring methods from upstream traits, this inherits from all of them.
/// The only Miden-specific addition is [`LiftedAirBuilder::aux_values`], which provides
/// prover-supplied aux values for use in [`eval`](crate::LiftedAir::eval) constraints.
///
/// Note: cross-AIR bus identity checking is handled separately by
/// [`LiftedAir::reduced_aux_values`](crate::LiftedAir::reduced_aux_values), which
/// operates on concrete field values outside the constraint builder.
pub trait LiftedAirBuilder:
    AirBuilder
    + AirBuilderWithPublicValues
    + ExtensionBuilder
    + PermutationAirBuilder
    + PeriodicAirBuilder
{
    /// Prover-supplied aux values carried in the proof (extension field).
    ///
    /// These are the values returned by [`AuxBuilder::build_aux_trace`](crate::AuxBuilder::build_aux_trace).
    /// How they relate to the aux trace is AIR-defined — a common pattern is to
    /// constrain them to equal the aux trace's last row, but this is not required.
    fn aux_values(&self) -> &[Self::VarEF] {
        &[]
    }
}
