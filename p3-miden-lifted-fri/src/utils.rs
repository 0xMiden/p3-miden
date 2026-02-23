use alloc::vec::Vec;
use core::array;
use core::ops::{Add, Mul};

use p3_field::coset::TwoAdicMultiplicativeCoset;
use p3_field::{
    ExtensionField, Field, FieldArray, PackedFieldExtension, PackedValue, PrimeCharacteristicRing,
    TwoAdicField,
};
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::reverse_slice_index_bits;

// ============================================================================
// Extension trait for PackedFieldExtension methods not in upstream
// ============================================================================

/// Horner fold with an explicit accumulator.
///
/// Computes `acc·xⁿ + v₀·xⁿ⁻¹ + v₁·xⁿ⁻² + ... + vₙ₋₁·x⁰` where n = len(vals).
/// Equivalently: `((acc·x + v₀)·x + v₁)·x + ... + vₙ₋₁`.
/// The first element gets the highest power of `x`.
///
/// For polynomial evaluation `p(x) = Σᵢ cᵢ·xⁱ`, pass coefficients in
/// descending degree order `[cₙ, ..., c₁, c₀]`.
#[inline]
pub(crate) fn horner_acc<Acc, Val, X, I>(acc: Acc, x: X, vals: I) -> Acc
where
    I: IntoIterator<Item = Val>,
    Acc: Mul<X, Output = Acc> + Add<Val, Output = Acc>,
    X: Clone,
{
    vals.into_iter().fold(acc, |acc, val| acc * x.clone() + val)
}

/// Horner fold starting from zero.
///
/// See [`horner_acc`] for the evaluation convention.
#[inline]
pub(crate) fn horner<Acc, Val, X, I>(x: X, vals: I) -> Acc
where
    I: IntoIterator<Item = Val>,
    Acc: Default + Mul<X, Output = Acc> + Add<Val, Output = Acc>,
    X: Clone,
{
    horner_acc(Acc::default(), x, vals)
}

/// Extension trait adding `pack_ext_columns` and `to_ext_slice` methods.
///
/// These methods enable efficient SIMD operations on arrays of extension field elements
/// by providing column-wise packing and unpacking utilities.
pub(crate) trait PackedFieldExtensionExt<
    BaseField: Field,
    ExtField: ExtensionField<BaseField, ExtensionPacking = Self>,
>: PackedFieldExtension<BaseField, ExtField>
{
    /// Pack N columns from WIDTH rows into N packed extension field elements.
    ///
    /// Input: `rows[lane][col]` - WIDTH rows, each with N extension field elements.
    /// Output: `result[col]` - N packed values, where each packs WIDTH lanes.
    fn pack_ext_columns<const N: usize>(rows: &[[ExtField; N]]) -> [Self; N] {
        let width = BaseField::Packing::WIDTH;
        debug_assert_eq!(rows.len(), width);
        array::from_fn(|col| {
            let col_elems: Vec<ExtField> = (0..width).map(|lane| rows[lane][col]).collect();
            Self::from_ext_slice(&col_elems)
        })
    }

    /// Extract all lanes to an output slice.
    fn to_ext_slice(&self, out: &mut [ExtField]) {
        let width = BaseField::Packing::WIDTH;
        for (lane, slot) in out.iter_mut().enumerate().take(width) {
            *slot = self.extract(lane);
        }
    }
}

impl<
    BaseField: Field,
    ExtField: ExtensionField<BaseField, ExtensionPacking = P>,
    P: PackedFieldExtension<BaseField, ExtField>,
> PackedFieldExtensionExt<BaseField, ExtField> for P
{
}

// ============================================================================
// Extension trait for Matrix batch operations
// ============================================================================

/// Extension trait adding batched column dot product to `p3_matrix::Matrix`.
///
/// This adds the `columnwise_dot_product_batched` method that computes N dot products
/// simultaneously, improving cache utilization by loading each matrix row once.
pub(crate) trait MatrixExt<T: Send + Sync + Clone>: Matrix<T> {
    /// Compute Mᵀ · [v₀, v₁, ..., vₙ₋₁] for N weight vectors simultaneously.
    ///
    /// Computes `result[col][j] = Σᵣ M[r, col] · vⱼ[r]` for r in `0..vs.len()`,
    /// for all columns and all j ∈ [0, N). Requires `vs.len() <= self.height()`.
    fn columnwise_dot_product_batched<EF, const N: usize>(
        &self,
        vs: &[FieldArray<EF, N>],
    ) -> Vec<FieldArray<EF, N>>
    where
        T: Field,
        EF: ExtensionField<T>;
}

impl<T: Send + Sync + Clone, M: Matrix<T>> MatrixExt<T> for M {
    fn columnwise_dot_product_batched<EF, const N: usize>(
        &self,
        vs: &[FieldArray<EF, N>],
    ) -> Vec<FieldArray<EF, N>>
    where
        T: Field,
        EF: ExtensionField<T>,
    {
        debug_assert!(
            vs.len() <= self.height(),
            "more weight vectors ({}) than matrix rows ({})",
            vs.len(),
            self.height()
        );
        let packed_width = self.width().div_ceil(T::Packing::WIDTH);

        let packed_results: Vec<EF::ExtensionPacking> = self
            .par_padded_horizontally_packed_rows::<T::Packing>()
            .zip(vs)
            .par_fold_reduce(
                || EF::ExtensionPacking::zero_vec(packed_width * N),
                |mut acc, (packed_row, scales)| {
                    let (acc_chunks, _) = acc.as_chunks_mut::<N>();
                    // Broadcast each scalar scale to all SIMD lanes
                    let packed_scales: [EF::ExtensionPacking; N] =
                        scales.map_into_array(EF::ExtensionPacking::from);

                    // acc[c][j] += scales[j] · row[c] for column batch c, point j
                    acc_chunks
                        .iter_mut()
                        .zip(packed_row)
                        .for_each(|(acc_c, row_c)| {
                            for j in 0..N {
                                acc_c[j] += packed_scales[j] * row_c;
                            }
                        });
                    acc
                },
                |mut acc_l, acc_r| {
                    acc_l.iter_mut().zip(&acc_r).for_each(|(lj, rj)| *lj += *rj);
                    acc_l
                },
            );

        // Unpack: chunk[j].lane(i) → result[c·W + i][j] for column batch c
        packed_results
            .chunks(N)
            .flat_map(|chunk| {
                (0..T::Packing::WIDTH)
                    .map(move |lane| FieldArray::from_fn(|j| chunk[j].extract(lane)))
            })
            .take(self.width())
            .collect()
    }
}

/// Coset points `gK` in bit-reversed order.
///
/// Note: the coset shift `g` is fixed to `F::GENERATOR` by convention in this PCS.
///
/// Bit-reversal gives two properties essential for lifting:
/// - **Adjacent negation**: `gK[2i+1] = -gK[2i]`, so both square to the same value
/// - **Prefix nesting**: `gK[0..n/r]` equals the r-th power coset `(gK)ʳ`
///
/// Together these enable iterative weight folding in barycentric evaluation.
///
/// # Panics
/// Panics if the two-adic coset construction fails (e.g., `log_n` exceeds the field's
/// two-adicity), since this unwraps `TwoAdicMultiplicativeCoset::new`.
pub fn bit_reversed_coset_points<F: TwoAdicField>(log_n: usize) -> Vec<F> {
    let coset = TwoAdicMultiplicativeCoset::new(F::GENERATOR, log_n).unwrap();
    let mut pts: Vec<F> = coset.iter().collect();
    reverse_slice_index_bits(&mut pts);
    pts
}
