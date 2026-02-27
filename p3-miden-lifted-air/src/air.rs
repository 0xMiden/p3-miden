//! The `LiftedAir` super-trait for AIR definitions in the lifted STARK system.

use p3_air::{BaseAir, BaseAirWithPublicValues};
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_util::log2_ceil_usize;

use crate::aux::{ReducedAuxValues, ReductionError, VarLenPublicInputs};
use crate::{AirWithPeriodicColumns, LiftedAirBuilder, SymbolicAirBuilder};

/// Super-trait for AIR definitions used by the lifted STARK prover/verifier.
///
/// Inherits from upstream traits for width, public values, and periodic columns.
/// Adds Miden-specific auxiliary trace support.
///
/// # Type Parameters
/// - `F`: Base field
/// - `EF`: Extension field (for aux trace challenges and aux values)
pub trait LiftedAir<F: Field, EF>:
    Sync + BaseAir<F> + BaseAirWithPublicValues<F> + AirWithPeriodicColumns<F>
{
    /// Number of extension-field challenges required for the auxiliary trace.
    fn num_randomness(&self) -> usize {
        0
    }

    /// Number of extension-field columns in the auxiliary trace.
    fn aux_width(&self) -> usize {
        0
    }

    /// Number of extension-field aux values committed to the Fiat-Shamir transcript.
    ///
    /// These are the scalars returned by [`AuxBuilder::build_aux_trace`](crate::AuxBuilder::build_aux_trace)
    /// alongside the aux trace matrix. Their count may differ from [`aux_width`](Self::aux_width)
    /// (the number of aux trace columns).
    fn num_aux_values(&self) -> usize {
        0
    }

    /// Number of variable-length public inputs this AIR expects.
    ///
    /// Each input is a slice of base-field elements that
    /// [`reduced_aux_values`](Self::reduced_aux_values) reduces to a single value.
    /// Callers must provide exactly this many slices.
    fn num_var_len_public_inputs(&self) -> usize {
        0
    }

    /// Reduce this AIR's aux values to a [`ReducedAuxValues`] contribution.
    ///
    /// Called by the verifier (with concrete field values, not symbolic expressions)
    /// to compute each AIR's contribution to the global cross-AIR bus identity check.
    /// The verifier accumulates contributions across all AIRs and checks that the
    /// combined result is identity (prod=1, sum=0).
    ///
    /// # Arguments
    /// - `aux_values`: prover-supplied aux values (from the proof)
    /// - `challenges`: extension-field challenges (same as used for aux trace building)
    /// - `public_values`: this AIR's public values (base field)
    /// - `var_len_public_inputs`: reducible inputs for the cross-AIR identity check
    ///
    /// # Errors
    ///
    /// Returns a [`ReductionError`] if the inputs are invalid (e.g. wrong
    /// count or wrong length of `var_len_public_inputs`).
    ///
    /// Default: returns identity (correct for AIRs without buses).
    fn reduced_aux_values(
        &self,
        _aux_values: &[EF],
        _challenges: &[EF],
        _public_values: &[F],
        _var_len_public_inputs: VarLenPublicInputs<'_, F>,
    ) -> Result<ReducedAuxValues<EF>, ReductionError>
    where
        EF: ExtensionField<F>,
    {
        Ok(ReducedAuxValues::identity())
    }

    /// Evaluate all AIR constraints using the provided builder.
    fn eval<AB: LiftedAirBuilder<F = F>>(&self, builder: &mut AB);

    /// Log₂ of the number of quotient chunks, inferred from symbolic constraint analysis.
    ///
    /// Evaluates the AIR on a [`SymbolicAirBuilder`](crate::SymbolicAirBuilder) to determine
    /// the maximum constraint degree M, then returns `log2_ceil(M - 1)` (padded so M ≥ 2).
    ///
    /// Uses `SymbolicAirBuilder<F>` (i.e. `EF = F`) which is sufficient for degree
    /// computation since extension-field operations have the same degree structure.
    ///
    /// # Why `M − 1` chunks?
    ///
    /// Let N be the trace height (so trace columns are polynomials of degree < N).
    /// Symbolic evaluation assigns each constraint a *degree multiple* M, meaning the
    /// resulting numerator polynomial C(X) has degree bounded by roughly M·(N − 1).
    ///
    /// In a STARK, the constraint numerator is divisible by the trace vanishing
    /// polynomial `Z_H(X) = Xᴺ − 1`, so the quotient polynomial
    /// `Q(X) = C(X) / Z_H(X)` has
    ///
    /// `deg(Q) ≤ deg(C) − N ≤ M·(N − 1) − N < (M − 1)·N`.
    ///
    /// We commit to Q(X) by splitting it into D chunks of degree < N. The bound above
    /// shows that D = M − 1 chunks suffice; we then round D up to a power of two and
    /// return `log2(D)`.
    ///
    /// We clamp M ≥ 2 so that D ≥ 1. If M = 1 then `deg(C) < N`, and divisibility by
    /// `Z_H` would force C(X) to be the zero polynomial (i.e. the constraint carries no
    /// information about the trace).
    fn log_quotient_degree(&self) -> usize
    where
        Self: Sized,
    {
        let preprocessed_width = self.preprocessed_trace().map_or(0, |t| t.width());
        let mut builder = SymbolicAirBuilder::<F>::new(
            preprocessed_width,
            self.width(),
            self.num_public_values(),
            self.aux_width(),
            self.num_aux_values(),
            self.num_randomness(),
            self.periodic_columns().len(),
        );
        self.eval(&mut builder);

        let base_degree = builder
            .base_constraints()
            .iter()
            .map(|c| c.degree_multiple())
            .max()
            .unwrap_or(0);
        let ext_degree = builder
            .extension_constraints()
            .iter()
            .map(|c| c.degree_multiple())
            .max()
            .unwrap_or(0);
        let constraint_degree = base_degree.max(ext_degree).max(2);

        log2_ceil_usize(constraint_degree - 1)
    }

    /// Number of quotient chunks: `2^log_quotient_degree()`.
    fn constraint_degree(&self) -> usize
    where
        Self: Sized,
    {
        1 << self.log_quotient_degree()
    }
}
