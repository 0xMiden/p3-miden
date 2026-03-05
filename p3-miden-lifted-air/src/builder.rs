//! The `LiftedAirBuilder` super-trait for constraint evaluation builders.

use crate::{AirBuilder, ExtensionBuilder, PeriodicAirBuilder, PermutationAirBuilder};

/// Super-trait bundling all builder capabilities needed by the lifted STARK system.
///
/// Inherits from upstream traits and adds no additional methods.
/// Prover-supplied aux values are exposed via [`PermutationAirBuilder::permutation_values`].
///
/// Note: cross-AIR bus identity checking is handled separately by
/// [`LiftedAir::reduced_aux_values`](crate::LiftedAir::reduced_aux_values), which
/// operates on concrete field values outside the constraint builder.
pub trait LiftedAirBuilder:
    AirBuilder + ExtensionBuilder + PermutationAirBuilder + PeriodicAirBuilder
{
}

// Blanket impl: any type satisfying the super-trait bounds is a LiftedAirBuilder.
impl<T> LiftedAirBuilder for T where
    T: AirBuilder + ExtensionBuilder + PermutationAirBuilder + PeriodicAirBuilder
{
}
