//! Constraint evaluation for the prover.
//!
//! This module provides:
//! - [`evaluate_constraints_into`]: SIMD-parallel constraint evaluation on the quotient domain
//! - [`ProverConstraintFolder`]: SIMD-optimized folder for constraint evaluation

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedField, PackedFieldExtension,
    PackedValue, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::Matrix;
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_maybe_rayon::prelude::*;
use p3_miden_lifted_air::{
    AirBuilder, AirBuilderWithPublicValues, ConstraintLayout, ExtensionBuilder, LiftedAir,
    LiftedAirBuilder, PairBuilder, PeriodicAirBuilder, PermutationAirBuilder,
};
use p3_miden_lifted_stark::{LiftedCoset, Selectors};

use crate::periodic::PeriodicLde;

// ============================================================================
// Constraint Evaluation
// ============================================================================

/// Type alias for packed base field from F.
type PackedVal<F> = <F as Field>::Packing;

/// Type alias for packed extension field from EF.
type PackedExt<F, EF> = <EF as ExtensionField<F>>::ExtensionPacking;

/// Batch size for constraint linear-combination chunks in [`finalize_constraints`].
const CONSTRAINT_BATCH: usize = 8;

/// Batched linear combination of packed extension field values with EF coefficients.
///
/// Extension-field analogue of [`PackedField::packed_linear_combination`]. Processes
/// `coeffs` and `values` in chunks of [`CONSTRAINT_BATCH`], then handles the remainder.
#[inline]
fn batched_ext_linear_combination<PE, EF>(coeffs: &[EF], values: &[PE]) -> PE
where
    EF: Field,
    PE: PrimeCharacteristicRing + Algebra<EF> + Copy,
{
    debug_assert_eq!(coeffs.len(), values.len());
    let len = coeffs.len();
    let mut acc = PE::ZERO;
    let mut start = 0;
    while start + CONSTRAINT_BATCH <= len {
        let batch: [PE; CONSTRAINT_BATCH] =
            core::array::from_fn(|i| values[start + i] * coeffs[start + i]);
        acc += PE::sum_array::<CONSTRAINT_BATCH>(&batch);
        start += CONSTRAINT_BATCH;
    }
    for (&coeff, &val) in coeffs[start..].iter().zip(&values[start..]) {
        acc += val * coeff;
    }
    acc
}

/// Batched linear combination of packed base field values with F coefficients.
///
/// Wraps [`PackedField::packed_linear_combination`] with batched chunking
/// and remainder handling, mirroring [`batched_ext_linear_combination`].
#[inline]
fn batched_base_linear_combination<P: PackedField>(coeffs: &[P::Scalar], values: &[P]) -> P {
    debug_assert_eq!(coeffs.len(), values.len());
    let len = coeffs.len();
    let mut acc = P::ZERO;
    let mut start = 0;
    while start + CONSTRAINT_BATCH <= len {
        acc += P::packed_linear_combination::<CONSTRAINT_BATCH>(
            &coeffs[start..start + CONSTRAINT_BATCH],
            &values[start..start + CONSTRAINT_BATCH],
        );
        start += CONSTRAINT_BATCH;
    }
    for (&coeff, &val) in coeffs[start..].iter().zip(&values[start..]) {
        acc += val * coeff;
    }
    acc
}

/// Evaluate constraints on the quotient domain, adding results into `output`.
///
/// Here `gJ` is the quotient evaluation coset of size `N * D`, the subset of the
/// committed LDE coset `gK` (size `N * B`) that contains just enough points to
/// evaluate the quotient point-wise. For each point on `gJ`, we evaluate all AIR
/// constraints, fold them with powers of `alpha`, and add the resulting numerator value:
///
/// `output[i] += folded_constraints(xᵢ)`.
///
/// The caller is responsible for preparing `output` before calling this function
/// (e.g. cyclically extending and scaling by beta for multi-trace accumulation).
/// Input matrices must be in natural order on gJ.
///
/// Uses SIMD-packed parallel iteration via rayon for optimal performance:
/// - Processes `WIDTH` points simultaneously using packed field types
/// - Main trace stays in base field, only aux trace uses extension field
/// - Constraints are collected then finalized in batches via decomposed alpha powers
///
/// Why we fold with `alpha`: the prover does not want to carry K separate constraint
/// polynomials through the rest of the protocol. A random linear combination
///
/// `C_fold(x) = Σₖ α^{K−1−k}·Cₖ(x)`
///
/// collapses them into one numerator polynomial while preserving soundness (a non-zero
/// constraint survives with high probability).
///
/// Why we only evaluate on `gJ`: `gJ` (size `N * D`) is a subset of the committed LDE
/// coset `gK` (size `N * B`). For `B >= D`, these `N * D` points are sufficient for
/// the quotient-degree bounds used by the protocol; division by the vanishing polynomial
/// happens later.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_constraints_into<F, EF, A, M>(
    output: &mut [EF],
    air: &A,
    main_on_gj: &M,
    aux_on_gj: &M,
    coset: &LiftedCoset,
    alpha: EF,
    randomness: &[EF],
    public_values: &[F],
    periodic_lde: &PeriodicLde<F>,
    layout: &ConstraintLayout,
    aux_values: &[EF],
) where
    F: TwoAdicField + PrimeCharacteristicRing,
    EF: ExtensionField<F>,
    PackedExt<F, EF>: Algebra<EF> + Algebra<PackedVal<F>> + BasedVectorSpace<PackedVal<F>>,
    A: LiftedAir<F, EF>,
    M: Matrix<F> + Sync,
{
    type P<F> = PackedVal<F>;
    type PE<F, EF> = PackedExt<F, EF>;

    let gj_height = coset.lde_height();
    assert_eq!(output.len(), gj_height);
    let constraint_degree = coset.blowup();
    let width = P::<F>::WIDTH;

    assert!(
        gj_height.is_multiple_of(width),
        "quotient height must be divisible by packing width"
    );

    // Precompute selectors via coset method
    let sels = coset.selectors::<F>();

    // ─── Decompose alpha powers by constraint layout ───
    let aux_ef_width = air.aux_width();
    let constraint_count = layout.total_constraints();
    let base_count = layout.base_indices.len();
    let ext_count = layout.ext_indices.len();
    let (base_alpha_powers, ext_alpha_powers) = layout.decompose_alpha(alpha);

    // Main trace width
    let main_width = main_on_gj.width();

    // Pack randomness for aux trace
    let packed_randomness: Vec<PE<F, EF>> = randomness.iter().copied().map(Into::into).collect();

    // Pack aux values
    let packed_aux_values: Vec<PE<F, EF>> = aux_values.iter().copied().map(Into::into).collect();

    // Parallel iteration over quotient domain points, step by WIDTH.
    // Write directly into output slice via par_chunks_mut.
    output
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(r, chunk)| {
            let i_start = r * width;

            // Extract packed selectors from precomputed vectors
            let selectors = sels.packed_at::<P<F>>(i_start);

            // Get main trace as packed row pair (stays in base field)
            let main_packed: Vec<P<F>> =
                main_on_gj.vertically_packed_row_pair(i_start, constraint_degree);
            let main = RowMajorMatrix::new(main_packed, main_width);

            // Get aux trace as packed row pair and convert to packed extension field
            let aux_base_packed: Vec<P<F>> =
                aux_on_gj.vertically_packed_row_pair(i_start, constraint_degree);

            // Convert from packed base field to packed extension field
            // Each EF element is formed from DIMENSION consecutive base field elements
            let aux_packed: Vec<PE<F, EF>> = (0..aux_ef_width * 2)
                .map(|i| {
                    PE::<F, EF>::from_basis_coefficients_fn(|j| {
                        aux_base_packed[i * EF::DIMENSION + j]
                    })
                })
                .collect();
            let aux = RowMajorMatrix::new(aux_packed, aux_ef_width);

            // Get packed periodic values
            let periodic_values: Vec<P<F>> = periodic_lde.packed_values_at(i_start).collect();

            // Build packed folder and evaluate constraints
            let mut folder: ProverConstraintFolder<'_, F, EF, P<F>, PE<F, EF>> =
                ProverConstraintFolder {
                    main: main.as_view(),
                    aux: aux.as_view(),
                    packed_randomness: &packed_randomness,
                    public_values,
                    periodic_values: &periodic_values,
                    aux_values: &packed_aux_values,
                    selectors,
                    base_alpha_powers: &base_alpha_powers,
                    ext_alpha_powers: &ext_alpha_powers,
                    constraint_index: 0,
                    constraint_count,
                    base_constraints: Vec::with_capacity(base_count),
                    ext_constraints: Vec::with_capacity(ext_count),
                    _phantom: PhantomData,
                };

            air.eval(&mut folder);
            let folded = folder.finalize_constraints();

            // Unpack folded result and add scalars directly into the output chunk.
            for (slot, val) in chunk.iter_mut().zip(PE::<F, EF>::to_ext_iter([folded])) {
                *slot += val;
            }
        });
}

// ============================================================================
// Prover Constraint Folder (SIMD-optimized)
// ============================================================================

/// Packed constraint folder for SIMD-optimized prover evaluation.
///
/// Uses packed types to evaluate constraints on multiple domain points simultaneously:
/// - `P`: Packed base field (e.g., `PackedBabyBear`)
/// - `PE`: Packed extension field - must be `Algebra<EF> + Algebra<P> + BasedVectorSpace<P>`
///
/// Collects constraints during `air.eval()` into separate base/ext vectors, then
/// combines them in [`Self::finalize_constraints`] using decomposed alpha powers and
/// `packed_linear_combination` for efficient SIMD accumulation.
///
/// # Type Parameters
/// - `F`: Base field scalar
/// - `EF`: Extension field scalar
/// - `P`: Packed base field (with `P::Scalar = F`)
/// - `PE`: Packed extension field (must implement appropriate algebra traits)
pub struct ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    /// Main trace matrix view (packed base field)
    pub main: RowMajorMatrixView<'a, P>,
    /// Aux/permutation trace matrix view (packed extension field)
    pub aux: RowMajorMatrixView<'a, PE>,
    /// Randomness for aux trace (packed extension field)
    pub packed_randomness: &'a [PE],
    /// Public values (base field scalars)
    pub public_values: &'a [F],
    /// Periodic column values (packed base field)
    pub periodic_values: &'a [P],
    /// Aux values (packed extension field)
    pub aux_values: &'a [PE],
    /// Constraint selectors (packed base field)
    pub selectors: Selectors<P>,
    /// Base-field alpha powers, reordered to match base constraint emission order.
    /// `base_alpha_powers[d][j]` = d-th basis coefficient of alpha power for j-th base constraint.
    pub base_alpha_powers: &'a [Vec<F>],
    /// Extension-field alpha powers, reordered to match ext constraint emission order.
    pub ext_alpha_powers: &'a [EF],
    /// Current constraint index (debug-only bookkeeping)
    pub constraint_index: usize,
    /// Total expected constraint count (debug-only bookkeeping)
    pub constraint_count: usize,
    /// Collected base-field constraints for this row
    pub base_constraints: Vec<P>,
    /// Collected extension-field constraints for this row
    pub ext_constraints: Vec<PE>,
    pub _phantom: PhantomData<EF>,
}

impl<'a, F, EF, P, PE> ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    /// Combine all collected constraints with their pre-computed alpha powers.
    ///
    /// Base constraints use `batched_base_linear_combination` per basis dimension,
    /// decomposing the extension-field multiply into D base-field SIMD dot products.
    /// Extension constraints use `batched_ext_linear_combination` with scalar EF
    /// coefficients. Both process in chunks of `CONSTRAINT_BATCH`.
    ///
    /// We keep base and extension constraints separate because the base constraints can
    /// stay in the base field and use packed SIMD arithmetic. Decomposing EF powers of
    /// `alpha` into base-field coordinates turns the base-field fold into a small number
    /// of packed dot-products, avoiding repeated cross-field promotions.
    #[inline]
    pub fn finalize_constraints(self) -> PE {
        debug_assert_eq!(self.constraint_index, self.constraint_count);
        debug_assert_eq!(
            self.base_constraints.len(),
            self.base_alpha_powers.first().map_or(0, Vec::len)
        );
        debug_assert_eq!(self.ext_constraints.len(), self.ext_alpha_powers.len());

        // Base constraints: D independent base-field dot products
        let base = &self.base_constraints;
        let base_powers = self.base_alpha_powers;
        let acc = PE::from_basis_coefficients_fn(|d| {
            batched_base_linear_combination(&base_powers[d], base)
        });

        // Extension constraints: EF-coefficient dot product
        acc + batched_ext_linear_combination(self.ext_alpha_powers, &self.ext_constraints)
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
    type M = RowMajorMatrixView<'a, P>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
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
        self.base_constraints.push(x.into());
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let expr_array = array.map(Into::into);
        self.base_constraints.extend(expr_array);
        self.constraint_index += N;
    }
}

impl<'a, F, EF, P, PE> AirBuilderWithPublicValues for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type PublicVar = F;

    #[inline]
    fn public_values(&self) -> &[Self::PublicVar] {
        self.public_values
    }
}

impl<'a, F, EF, P, PE> PairBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    fn preprocessed(&self) -> Self::M {
        panic!("preprocessed trace not supported in this prototype")
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
        self.ext_constraints.push(x.into());
        self.constraint_index += 1;
    }
}

impl<'a, F, EF, P, PE> PermutationAirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    type MP = RowMajorMatrixView<'a, PE>;
    type RandomVar = PE;

    #[inline]
    fn permutation(&self) -> Self::MP {
        self.aux
    }

    #[inline]
    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.packed_randomness
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

impl<'a, F, EF, P, PE> LiftedAirBuilder for ProverConstraintFolder<'a, F, EF, P, PE>
where
    F: Field,
    EF: ExtensionField<F>,
    P: PackedField<Scalar = F>,
    PE: Algebra<EF> + Algebra<P> + BasedVectorSpace<P> + Copy + Send + Sync,
{
    fn aux_values(&self) -> &[Self::VarEF] {
        self.aux_values
    }
}
