//! Constraint evaluation for the prover.
//!
//! - `evaluate_constraints_into`: SIMD-parallel constraint evaluation on the quotient domain
//! - `folder`: SIMD-optimized constraint folder and finalization
//! - `layout`: Constraint layout discovery (base vs extension) and alpha decomposition

mod folder;
mod layout;

pub(crate) use folder::ProverConstraintFolder;
pub(crate) use layout::{ConstraintLayout, get_constraint_layout};

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, PackedFieldExtension, PackedValue,
    TwoAdicField,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_miden_lifted_air::LiftedAir;
use p3_miden_lifted_stark::LiftedCoset;

use crate::periodic::PeriodicLde;

/// Type alias for packed base field from F.
type PackedVal<F> = <F as Field>::Packing;

/// Type alias for packed extension field from EF.
type PackedExt<F, EF> = <EF as ExtensionField<F>>::ExtensionPacking;

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
pub(crate) fn evaluate_constraints_into<F, EF, A, M>(
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
    F: TwoAdicField,
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
