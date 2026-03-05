//! Constraint layout: maps global constraint indices to base/ext streams.
//!
//! Also provides [`ConstraintLayoutBuilder`], a lightweight AIR builder that discovers
//! constraint types without building symbolic expression trees.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lifted_air::{
    AirBuilder, ExtensionBuilder, LiftedAir, PeriodicAirBuilder, PermutationAirBuilder,
};
use tracing::instrument;

/// Maps between global constraint indices and the separated base/ext streams.
///
/// When alpha powers are pre-computed in global order `[α^{N−1}, …, α⁰]`,
/// the layout tells us which powers correspond to base-field constraints (for
/// `packed_linear_combination`) and which to extension-field constraints.
#[derive(Debug, Default)]
pub(crate) struct ConstraintLayout {
    /// Global indices of base-field constraints, in emission order.
    pub base_indices: Vec<usize>,
    /// Global indices of extension-field constraints, in emission order.
    pub ext_indices: Vec<usize>,
}

impl ConstraintLayout {
    /// Total number of constraints (base + extension).
    pub fn total_constraints(&self) -> usize {
        self.base_indices.len() + self.ext_indices.len()
    }

    /// Decompose `α` into reordered powers for base and extension constraints.
    ///
    /// Returns `(base_alpha_powers, ext_alpha_powers)` where:
    /// - `base_alpha_powers[d][j]` = d-th basis coefficient of the alpha power for
    ///   the j-th base constraint (transposed + reordered for `packed_linear_combination`)
    /// - `ext_alpha_powers[j]` = full EF alpha power for the j-th extension constraint
    ///
    /// Constraints are emitted in one global order and folded into a single random
    /// linear combination using powers of `α`:
    ///
    /// `C_fold(X) = Σ_{i=0..K−1} α^{K−1−i} · Cᵢ(X)`.
    ///
    /// We use descending powers because the verifier evaluates the fold at a single
    /// point via Horner (streaming): `acc = acc·α + Cᵢ`.
    ///
    /// The prover accumulates base-field constraints with packed (SIMD) arithmetic for
    /// throughput, while extension constraints must stay in the extension field. This
    /// method splits the precomputed powers accordingly, and also transposes EF powers
    /// into their base-field coordinates so the base-field path can use
    /// `packed_linear_combination` without repeated cross-field conversions.
    pub fn decompose_alpha<F: Field, EF: ExtensionField<F>>(
        &self,
        alpha: EF,
    ) -> (Vec<Vec<F>>, Vec<EF>) {
        let total = self.total_constraints();

        // alpha_powers[i] = α^{total − 1 − i}, so constraint i gets
        // weight α^{total − 1 − i} in the linear combination.
        let mut alpha_powers: Vec<EF> = alpha.powers().take(total).collect();
        alpha_powers.reverse();

        // Base: transpose EF -> [F; D] and reorder by base_indices in one pass
        let base_alpha_powers: Vec<Vec<F>> = (0..EF::DIMENSION)
            .map(|d| {
                self.base_indices
                    .iter()
                    .map(|&idx| alpha_powers[idx].as_basis_coefficients_slice()[d])
                    .collect()
            })
            .collect();

        // Ext: pick full EF powers by ext_indices
        let ext_alpha_powers: Vec<EF> = self
            .ext_indices
            .iter()
            .map(|&idx| alpha_powers[idx])
            .collect();

        (base_alpha_powers, ext_alpha_powers)
    }
}

// ============================================================================
// Constraint Layout Builder (lightweight, no symbolic expressions)
// ============================================================================

/// Evaluate the AIR on a lightweight builder and return the constraint layout.
///
/// Runs `air.eval()` on a [`ConstraintLayoutBuilder`] that uses concrete field zeros
/// for all variables. This discovers which constraints are base-field vs extension-field
/// without building symbolic expression trees — only the emission order matters.
#[instrument(name = "compute constraint layout", skip_all, level = "debug")]
pub(crate) fn get_constraint_layout<F, EF, A>(air: &A, num_public_values: usize) -> ConstraintLayout
where
    F: Field,
    EF: ExtensionField<F>,
    A: LiftedAir<F, EF>,
{
    let preprocessed_width = air.preprocessed_trace().map_or(0, |t| t.width());
    let mut builder = ConstraintLayoutBuilder::<F>::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        air.aux_width(),
        air.num_aux_values(),
        air.num_randomness(),
        air.periodic_columns().len(),
    );
    air.eval(&mut builder);
    builder.into_layout()
}

/// Lightweight AIR builder that only tracks constraint types (base vs extension).
///
/// Uses concrete field zeros for all variables — no symbolic expression trees, no degree
/// tracking, no `Arc` allocations. Builds a [`ConstraintLayout`] directly by recording
/// which `assert_*` method is called for each constraint.
struct ConstraintLayoutBuilder<F: Field> {
    preprocessed: RowMajorMatrix<F>,
    main: RowMajorMatrix<F>,
    public_values: Vec<F>,
    periodic_values: Vec<F>,
    permutation: RowMajorMatrix<F>,
    permutation_challenges: Vec<F>,
    permutation_values: Vec<F>,
    layout: ConstraintLayout,
    constraint_count: usize,
}

impl<F: Field> ConstraintLayoutBuilder<F> {
    fn new(
        preprocessed_width: usize,
        width: usize,
        num_public_values: usize,
        permutation_width: usize,
        num_aux_values: usize,
        num_permutation_challenges: usize,
        num_periodic_columns: usize,
    ) -> Self {
        Self {
            preprocessed: RowMajorMatrix::new(
                vec![F::ZERO; 2 * preprocessed_width],
                preprocessed_width,
            ),
            main: RowMajorMatrix::new(vec![F::ZERO; 2 * width], width),
            public_values: vec![F::ZERO; num_public_values],
            periodic_values: vec![F::ZERO; num_periodic_columns],
            permutation: RowMajorMatrix::new(
                vec![F::ZERO; 2 * permutation_width],
                permutation_width,
            ),
            permutation_challenges: vec![F::ZERO; num_permutation_challenges],
            permutation_values: vec![F::ZERO; num_aux_values],
            layout: ConstraintLayout::default(),
            constraint_count: 0,
        }
    }

    fn into_layout(self) -> ConstraintLayout {
        self.layout
    }
}

impl<F: Field> AirBuilder for ConstraintLayoutBuilder<F> {
    type F = F;
    type Expr = F;
    type Var = F;
    type M = RowMajorMatrix<F>;
    type PublicVar = F;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn preprocessed(&self) -> Option<Self::M> {
        Some(self.preprocessed.clone())
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }

    fn is_first_row(&self) -> Self::Expr {
        F::ZERO
    }

    fn is_last_row(&self) -> Self::Expr {
        F::ZERO
    }

    fn is_transition_window(&self, _size: usize) -> Self::Expr {
        F::ZERO
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, _x: I) {
        self.layout.base_indices.push(self.constraint_count);
        self.constraint_count += 1;
    }
}

impl<F: Field> ExtensionBuilder for ConstraintLayoutBuilder<F> {
    type EF = F;
    type ExprEF = F;
    type VarEF = F;

    fn assert_zero_ext<I>(&mut self, _x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.layout.ext_indices.push(self.constraint_count);
        self.constraint_count += 1;
    }
}

impl<F: Field> PermutationAirBuilder for ConstraintLayoutBuilder<F> {
    type MP = RowMajorMatrix<F>;
    type RandomVar = F;
    type PermutationVal = F;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.permutation_challenges
    }

    fn permutation_values(&self) -> &[Self::PermutationVal] {
        &self.permutation_values
    }
}

impl<F: Field> PeriodicAirBuilder for ConstraintLayoutBuilder<F> {
    type PeriodicVar = F;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &self.periodic_values
    }
}
