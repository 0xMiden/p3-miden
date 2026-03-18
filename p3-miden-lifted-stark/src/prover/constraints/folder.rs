//! SIMD-optimized constraint folder for prover evaluation.
//!
//! [`ProverConstraintFolder`] accumulates base and extension constraints on-the-fly during
//! `air.eval()`, folding each constraint with its alpha power directly into a running sum.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{Algebra, BasedVectorSpace, ExtensionField, Field, PackedField};
use p3_miden_lifted_air::{
    AirBuilder, EmptyWindow, ExtensionBuilder, PeriodicAirBuilder, PermutationAirBuilder, RowWindow,
};

use crate::selectors::Selectors;

/// Packed constraint folder for SIMD-optimized prover evaluation.
///
/// Uses packed types to evaluate constraints on multiple domain points simultaneously:
/// - `P`: Packed base field (e.g., `PackedBabyBear`)
/// - `PE`: Packed extension field - must be `Algebra<EF> + Algebra<P> + BasedVectorSpace<P>`
///
/// Accumulates constraints on-the-fly during `air.eval()` by folding each constraint
/// with its pre-computed alpha power directly into running accumulators (`base_acc` for
/// base-field constraints, `ext_acc` for extension-field constraints).
///
/// # Type Parameters
/// - `F`: Base field scalar
/// - `EF`: Extension field scalar
/// - `P`: Packed base field (with `P::Scalar = F`)
/// - `PE`: Packed extension field (must implement appropriate algebra traits)
pub(crate) struct ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    /// Main trace two-row window (packed base field)
    pub main: RowWindow<'a, P>,
    /// Aux/permutation trace two-row window (packed extension field)
    pub aux: RowWindow<'a, PE>,
    /// Randomness for aux trace (packed extension field)
    pub packed_randomness: &'a [PE],
    /// Public values (base field scalars)
    pub public_values: &'a [F],
    /// Periodic column values (packed base field)
    pub periodic_values: &'a [P],
    /// Permutation values (packed extension field)
    pub permutation_values: &'a [PE],
    /// Constraint selectors (packed base field)
    pub selectors: Selectors<P>,
    /// Base-field alpha powers, reordered to match base constraint emission order.
    /// `base_alpha_powers[d][j]` = d-th basis coefficient of alpha power for j-th base constraint.
    pub base_alpha_powers: &'a [Vec<F>],
    /// Extension-field alpha powers, reordered to match ext constraint emission order.
    pub ext_alpha_powers: &'a [EF],
    /// Running accumulator for base-field constraints (folded into PE via alpha powers).
    pub base_acc: PE,
    /// Running accumulator for extension-field constraints (folded via alpha powers).
    pub ext_acc: PE,
    /// Index of the next base constraint to be emitted.
    pub base_constraint_index: usize,
    /// Index of the next extension constraint to be emitted.
    pub ext_constraint_index: usize,
    /// Total expected constraint count (debug-only bookkeeping).
    pub constraint_count: usize,
    pub _phantom: PhantomData<EF>,
}

impl<'a, F, EF, P, PE> ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    /// Return the accumulated constraint folding result.
    ///
    /// Constraints were folded on-the-fly during `air.eval()`: each `assert_zero` /
    /// `assert_zero_ext` call multiplied by the corresponding alpha power and
    /// accumulated into `base_acc` (base-field constraints) or `ext_acc` (extension-
    /// field constraints).
    #[inline]
    pub fn finalize_constraints(self) -> PE {
        debug_assert_eq!(
            self.base_constraint_index + self.ext_constraint_index,
            self.constraint_count
        );
        debug_assert_eq!(
            self.base_constraint_index,
            self.base_alpha_powers.first().map_or(0, Vec::len)
        );
        debug_assert_eq!(self.ext_constraint_index, self.ext_alpha_powers.len());

        self.base_acc + self.ext_acc
    }
}

impl<'a, F, EF, P, PE> AirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type F = F;
    type Expr = P;
    type Var = P;
    type PreprocessedWindow = EmptyWindow<P>;
    type MainWindow = RowWindow<'a, P>;
    type PublicVar = F;

    #[inline]
    fn main(&self) -> Self::MainWindow {
        self.main
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        EmptyWindow::empty_ref()
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        self.selectors.is_first_row
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        self.selectors.is_last_row
    }

    #[inline]
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.selectors.is_transition
        } else {
            panic!("only window size 2 supported")
        }
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let val: P = x.into();
        let idx = self.base_constraint_index;
        let delta = PE::from_basis_coefficients_fn(|d| val * self.base_alpha_powers[d][idx]);
        self.base_acc += delta;
        self.base_constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        for x in array {
            self.assert_zero(x);
        }
    }

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<'a, F, EF, P, PE> ExtensionBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type EF = EF;
    type ExprEF = PE;
    type VarEF = PE;

    #[inline]
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        let val: PE = x.into();
        self.ext_acc += val * self.ext_alpha_powers[self.ext_constraint_index];
        self.ext_constraint_index += 1;
    }
}

impl<'a, F, EF, P, PE> PermutationAirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type MP = RowWindow<'a, PE>;
    type RandomVar = PE;
    type PermutationVar = PE;

    #[inline]
    fn permutation(&self) -> Self::MP {
        self.aux
    }

    #[inline]
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.packed_randomness
    }

    #[inline]
    fn permutation_values(&self) -> &[Self::PermutationVar] {
        self.permutation_values
    }
}

impl<'a, F, EF, P, PE> PeriodicAirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type PeriodicVar = P;

    #[inline]
    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.periodic_values
    }
}
