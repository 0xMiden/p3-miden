//! Symbolic AIR builder for constraint analysis.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::{
    Air, AirBuilder, AirBuilderWithPublicValues, Entry, ExtensionBuilder, LiftedAir,
    LiftedAirBuilder, PairBuilder, PeriodicAirBuilder, PermutationAirBuilder, SymbolicExpression,
    SymbolicVariable,
};

/// Compute the maximum constraint degree of base field constraints.
#[instrument(skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    num_periodic_columns: usize,
) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_max_constraint_degree_extension(
        air,
        preprocessed_width,
        num_public_values,
        0,
        0,
        0,
        num_periodic_columns,
    )
}

/// Compute the maximum constraint degree across both base and extension field constraints.
#[instrument(
    name = "infer base and extension constraint degree",
    skip_all,
    level = "debug"
)]
pub fn get_max_constraint_degree_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_aux_values: usize,
    num_permutation_challenges: usize,
    num_periodic_columns: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let (base_constraints, extension_constraints) = get_all_symbolic_constraints(
        air,
        preprocessed_width,
        num_public_values,
        permutation_width,
        num_aux_values,
        num_permutation_challenges,
        num_periodic_columns,
    );

    let base_degree = base_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);

    let extension_degree = extension_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);
    base_degree.max(extension_degree)
}

/// Evaluate the AIR symbolically and return the base field constraint expressions.
#[instrument(
    name = "evaluate base constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints<F, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    num_periodic_columns: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        0,
        0,
        0,
        num_periodic_columns,
    );
    air.eval(&mut builder);
    builder.base_constraints()
}

/// Evaluate the AIR symbolically and return the extension field constraint expressions.
#[instrument(
    name = "evaluate extension constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_aux_values: usize,
    num_permutation_challenges: usize,
    num_periodic_columns: usize,
) -> Vec<SymbolicExpression<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        permutation_width,
        num_aux_values,
        num_permutation_challenges,
        num_periodic_columns,
    );
    air.eval(&mut builder);
    builder.extension_constraints()
}

/// Evaluate the AIR symbolically and return both base and extension field constraint expressions.
#[instrument(
    name = "evaluate all constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_all_symbolic_constraints<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    num_public_values: usize,
    permutation_width: usize,
    num_aux_values: usize,
    num_permutation_challenges: usize,
    num_periodic_columns: usize,
) -> (Vec<SymbolicExpression<F>>, Vec<SymbolicExpression<EF>>)
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        permutation_width,
        num_aux_values,
        num_permutation_challenges,
        num_periodic_columns,
    );
    air.eval(&mut builder);
    (builder.base_constraints(), builder.extension_constraints())
}

// ============================================================================
// Constraint Layout
// ============================================================================

/// Tracks whether a constraint was emitted via `assert_zero` (base) or `assert_zero_ext` (ext).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConstraintType {
    Base,
    Ext,
}

/// Maps between global constraint indices and the separated base/ext streams.
///
/// When alpha powers are pre-computed in global order `[α^{N−1}, …, α⁰]`,
/// the layout tells us which powers correspond to base-field constraints (for
/// `packed_linear_combination`) and which to extension-field constraints.
#[derive(Debug, Default)]
pub struct ConstraintLayout {
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

/// Evaluate the AIR symbolically and return the constraint layout.
///
/// This runs `air.eval()` on a [`SymbolicAirBuilder`] to discover which constraints
/// are base-field vs extension-field, and their global ordering. The layout is used
/// by the prover to reorder decomposed alpha powers for efficient accumulation.
///
/// Uses `SymbolicAirBuilder<F>` (i.e. `EF = F`) which is sufficient for layout
/// computation since only constraint structure matters, not actual field arithmetic.
///
/// `num_public_values` is passed explicitly because
/// `BaseAirWithPublicValues::num_public_values` defaults to 0 and many AIRs do not
/// override it.
#[instrument(name = "compute constraint layout", skip_all, level = "debug")]
pub fn get_constraint_layout<F, EF, A>(air: &A, num_public_values: usize) -> ConstraintLayout
where
    F: Field,
    EF: ExtensionField<F>,
    A: LiftedAir<F, EF>,
{
    let preprocessed_width = air.preprocessed_trace().map_or(0, |t| t.width());
    let mut builder = SymbolicAirBuilder::<F>::new(
        preprocessed_width,
        air.width(),
        num_public_values,
        air.aux_width(),
        air.num_aux_values(),
        air.num_randomness(),
        air.periodic_columns().len(),
    );
    air.eval(&mut builder);
    builder.constraint_layout()
}

// ============================================================================
// Symbolic AIR Builder
// ============================================================================

/// An [`AirBuilder`] for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field, EF: ExtensionField<F> = F> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    periodic: Vec<SymbolicVariable<F>>,
    base_constraints: Vec<SymbolicExpression<F>>,
    permutation: RowMajorMatrix<SymbolicVariable<EF>>,
    permutation_challenges: Vec<SymbolicVariable<EF>>,
    aux_values: Vec<SymbolicVariable<EF>>,
    extension_constraints: Vec<SymbolicExpression<EF>>,
    constraint_types: Vec<ConstraintType>,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicAirBuilder<F, EF> {
    /// Create a new `SymbolicAirBuilder` with the given dimensions.
    pub fn new(
        preprocessed_width: usize,
        width: usize,
        num_public_values: usize,
        permutation_width: usize,
        num_aux_values: usize,
        num_permutation_challenges: usize,
        num_periodic_columns: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width)
                    .map(move |index| SymbolicVariable::new(Entry::Preprocessed { offset }, index))
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width).map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(Entry::Public, index))
            .collect();
        let periodic = (0..num_periodic_columns)
            .map(|index| SymbolicVariable::new(Entry::Periodic, index))
            .collect();
        let perm_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..permutation_width)
                    .map(move |index| SymbolicVariable::new(Entry::Permutation { offset }, index))
            })
            .collect();
        let permutation = RowMajorMatrix::new(perm_values, permutation_width);
        let permutation_challenges = (0..num_permutation_challenges)
            .map(|index| SymbolicVariable::new(Entry::Challenge, index))
            .collect();
        let aux_values = (0..num_aux_values)
            .map(|index| SymbolicVariable::new(Entry::AuxValue, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            periodic,
            base_constraints: vec![],
            permutation,
            permutation_challenges,
            aux_values,
            extension_constraints: vec![],
            constraint_types: vec![],
        }
    }

    /// Return the constraint layout mapping global indices to base/ext streams.
    pub fn constraint_layout(&self) -> ConstraintLayout {
        let mut base_indices = Vec::new();
        let mut ext_indices = Vec::new();
        for (idx, kind) in self.constraint_types.iter().enumerate() {
            match kind {
                ConstraintType::Base => base_indices.push(idx),
                ConstraintType::Ext => ext_indices.push(idx),
            }
        }
        ConstraintLayout {
            base_indices,
            ext_indices,
        }
    }

    /// Return the collected extension field constraints.
    pub fn extension_constraints(&self) -> Vec<SymbolicExpression<EF>> {
        self.extension_constraints.clone()
    }

    /// Return the collected base field constraints.
    pub fn base_constraints(&self) -> Vec<SymbolicExpression<F>> {
        self.base_constraints.clone()
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for SymbolicAirBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;

    fn main(&self) -> Self::M {
        self.main.clone()
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("SymbolicAirBuilder only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
        self.constraint_types.push(ConstraintType::Base);
    }
}

impl<F: Field, EF: ExtensionField<F>> PairBuilder for SymbolicAirBuilder<F, EF> {
    fn preprocessed(&self) -> Self::M {
        self.preprocessed.clone()
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilderWithPublicValues for SymbolicAirBuilder<F, EF> {
    type PublicVar = SymbolicVariable<F>;
    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpression<EF>: Algebra<SymbolicExpression<F>>,
{
    type EF = EF;
    type ExprEF = SymbolicExpression<EF>;
    type VarEF = SymbolicVariable<EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.extension_constraints.push(x.into());
        self.constraint_types.push(ConstraintType::Ext);
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpression<EF>: Algebra<SymbolicExpression<F>>,
{
    type MP = RowMajorMatrix<Self::VarEF>;

    type RandomVar = SymbolicVariable<EF>;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.permutation_challenges
    }
}

impl<F: Field, EF: ExtensionField<F>> PeriodicAirBuilder for SymbolicAirBuilder<F, EF> {
    type PeriodicVar = SymbolicVariable<F>;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &self.periodic
    }
}

impl<F: Field, EF: ExtensionField<F>> LiftedAirBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpression<EF>: Algebra<SymbolicExpression<F>>,
{
    fn aux_values(&self) -> &[Self::VarEF] {
        &self.aux_values
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_matrix::Matrix;

    use super::*;
    use crate::{AirWithPeriodicColumns, BaseAir, PairBuilder, PeriodicAirBuilder};

    type EF = BinomialExtensionField<BabyBear, 4>;

    // ==================== Configurable Mock AIR ====================
    //
    // A flexible test AIR that can be configured to generate constraints
    // with specific degree and structure. This allows testing constraint
    // degree computation for various AIR patterns with a single implementation.

    /// Which variable type to use as the base of the constraint expression.
    /// Different variable types have different degree contributions:
    /// - Main/Preprocessed/Periodic/Permutation: degree 1 (trace polynomials)
    /// - Public/Challenge: degree 0 (constants from verifier's perspective)
    #[derive(Clone, Copy, Default)]
    enum VariableKind {
        #[default]
        Main,
        Preprocessed,
        Public,
        Periodic,
        Permutation, // extension field variable
        Challenge,   // extension field variable
    }

    /// Condition wrapper for constraints.
    #[derive(Clone, Copy, Default)]
    enum Condition {
        #[default]
        None,
        Transition,
        FirstRow,
        LastRow,
    }

    /// Returns the standard periodic columns: 2 columns with periods 2 and 4.
    fn mock_periodic_columns() -> Vec<Vec<BabyBear>> {
        vec![
            vec![BabyBear::new(1), BabyBear::new(2)], // period 2
            vec![
                BabyBear::new(10),
                BabyBear::new(20),
                BabyBear::new(30),
                BabyBear::new(40),
            ], // period 4
        ]
    }

    /// A configurable AIR for testing constraint degree computation.
    ///
    /// Generates a constraint of the form: `condition * variable^exponent`
    /// The actual degree depends on:
    /// - Variable kind (trace columns = degree 1, public/challenge = degree 0)
    /// - Exponent (number of multiplications)
    /// - Condition (adds degree 1 if present)
    struct MockAir {
        /// Which variable type to exponentiate
        variable: VariableKind,
        /// Number of times to multiply the variable (1 = linear, 2 = quadratic)
        exponent: usize,
        /// Optional condition wrapper (adds degree 1)
        condition: Condition,
        /// Periodic columns (always initialized to standard test data)
        periodic_columns: Vec<Vec<BabyBear>>,
    }

    impl Default for MockAir {
        fn default() -> Self {
            Self {
                variable: VariableKind::default(),
                exponent: usize::default(),
                condition: Condition::default(),
                periodic_columns: mock_periodic_columns(),
            }
        }
    }

    impl MockAir {
        /// Compute expected constraint degree based on configuration.
        fn expected_degree(&self) -> usize {
            // Variable degree: trace columns have degree 1, public/challenge have degree 0
            let var_degree = match self.variable {
                VariableKind::Main
                | VariableKind::Preprocessed
                | VariableKind::Periodic
                | VariableKind::Permutation => 1,
                VariableKind::Public | VariableKind::Challenge => 0,
            };

            let base_degree = var_degree * self.exponent;

            // Condition adds degree (IsFirstRow and IsLastRow have degree 1, IsTransition has degree 0)
            match self.condition {
                Condition::None | Condition::Transition => base_degree,
                Condition::FirstRow | Condition::LastRow => base_degree + 1,
            }
        }
    }

    impl BaseAir<BabyBear> for MockAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl AirWithPeriodicColumns<BabyBear> for MockAir {
        fn periodic_columns(&self) -> &[Vec<BabyBear>] {
            &self.periodic_columns
        }
    }

    impl Air<SymbolicAirBuilder<BabyBear, EF>> for MockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<BabyBear, EF>) {
            // Build the constraint expression based on variable kind
            match self.variable {
                VariableKind::Main => {
                    let main = builder.main();
                    let local = main.row_slice(0).expect("matrix has rows");
                    let mut expr: SymbolicExpression<BabyBear> = local[0].into();
                    for _ in 1..self.exponent {
                        expr *= local[0];
                    }
                    self.assert_with_condition(builder, expr);
                }
                VariableKind::Preprocessed => {
                    let prep = builder.preprocessed();
                    let local = prep.row_slice(0).expect("matrix has rows");
                    let mut expr: SymbolicExpression<BabyBear> = local[0].into();
                    for _ in 1..self.exponent {
                        expr *= local[0];
                    }
                    self.assert_with_condition(builder, expr);
                }
                VariableKind::Public => {
                    let public = builder.public_values();
                    let mut expr: SymbolicExpression<BabyBear> = public[0].into();
                    for _ in 1..self.exponent {
                        expr *= public[0];
                    }
                    self.assert_with_condition(builder, expr);
                }
                VariableKind::Periodic => {
                    let periodic = builder.periodic_values();
                    let mut expr: SymbolicExpression<BabyBear> = periodic[0].into();
                    for _ in 1..self.exponent {
                        expr *= periodic[0];
                    }
                    self.assert_with_condition(builder, expr);
                }
                VariableKind::Permutation => {
                    let perm = builder.permutation();
                    let local = perm.row_slice(0).expect("matrix has rows");
                    let mut expr: SymbolicExpression<EF> = local[0].into();
                    for _ in 1..self.exponent {
                        expr *= local[0];
                    }
                    self.assert_ext_with_condition(builder, expr);
                }
                VariableKind::Challenge => {
                    let challenges = builder.permutation_randomness();
                    let mut expr: SymbolicExpression<EF> = challenges[0].into();
                    for _ in 1..self.exponent {
                        expr *= challenges[0];
                    }
                    self.assert_ext_with_condition(builder, expr);
                }
            }
        }
    }

    impl MockAir {
        /// Assert a base field expression with optional condition.
        fn assert_with_condition(
            &self,
            builder: &mut SymbolicAirBuilder<BabyBear, EF>,
            expr: SymbolicExpression<BabyBear>,
        ) {
            match self.condition {
                Condition::None => builder.assert_zero(expr),
                Condition::Transition => builder.when_transition().assert_zero(expr),
                Condition::FirstRow => builder.when_first_row().assert_zero(expr),
                Condition::LastRow => builder.when_last_row().assert_zero(expr),
            }
        }

        /// Assert an extension field expression with optional condition.
        fn assert_ext_with_condition(
            &self,
            builder: &mut SymbolicAirBuilder<BabyBear, EF>,
            expr: SymbolicExpression<EF>,
        ) {
            match self.condition {
                Condition::None => builder.assert_zero_ext(expr),
                Condition::Transition => {
                    let cond: SymbolicExpression<EF> = builder.is_transition().into();
                    builder.assert_zero_ext(cond * expr);
                }
                Condition::FirstRow => {
                    let cond: SymbolicExpression<EF> = builder.is_first_row().into();
                    builder.assert_zero_ext(cond * expr);
                }
                Condition::LastRow => {
                    let cond: SymbolicExpression<EF> = builder.is_last_row().into();
                    builder.assert_zero_ext(cond * expr);
                }
            }
        }
    }

    // ==================== Constraint Degree Tests ====================
    //
    // These tests verify that constraint degree is computed correctly for
    // all variable types. The degree determines quotient polynomial chunking
    // in the STARK protocol.

    /// Helper to compute constraint degree for a MockAir.
    /// Always assumes 1 of each variable type is available.
    fn compute_degree(air: &MockAir) -> usize {
        get_max_constraint_degree_extension::<BabyBear, EF, _>(
            air, 1, // preprocessed_width
            1, // num_public_values
            1, // permutation_width
            0, // num_aux_values
            1, // num_challenges
            1, // num_periodic_columns
        )
    }

    #[test]
    fn test_variable_degree_by_kind() {
        // Test cases: (variable_kind, exponent, expected_degree)
        // Trace columns (Main, Preprocessed, Periodic, Permutation) have degree 1
        // Constants (Public, Challenge) have degree 0
        let cases = [
            (VariableKind::Main, 3, 3),         // main^3 = degree 3
            (VariableKind::Preprocessed, 2, 2), // preprocessed^2 = degree 2
            (VariableKind::Periodic, 2, 2),     // periodic^2 = degree 2
            (VariableKind::Permutation, 2, 2),  // permutation^2 = degree 2 (extension field)
            (VariableKind::Public, 5, 0),       // public^5 = degree 0 (constants)
            (VariableKind::Challenge, 3, 0),    // challenge^3 = degree 0 (constants)
        ];

        for (variable, exponent, expected) in cases {
            let air = MockAir {
                variable,
                exponent,
                ..Default::default()
            };
            let degree = compute_degree(&air);
            assert_eq!(degree, air.expected_degree());
            assert_eq!(degree, expected);
        }
    }

    #[test]
    fn test_condition_adds_degree() {
        // FirstRow and LastRow add degree 1, Transition adds degree 0
        let cases = [
            (Condition::None, 2),       // main^2 = degree 2
            (Condition::Transition, 2), // is_transition (0) * main^2 = degree 2
            (Condition::FirstRow, 3),   // is_first_row (1) * main^2 = degree 3
            (Condition::LastRow, 3),    // is_last_row (1) * main^2 = degree 3
        ];

        for (condition, expected) in cases {
            let air = MockAir {
                variable: VariableKind::Main,
                exponent: 2,
                condition,
                ..Default::default()
            };
            let degree = compute_degree(&air);
            assert_eq!(degree, air.expected_degree());
            assert_eq!(degree, expected);
        }
    }

    #[test]
    fn test_extension_field_with_condition() {
        // Extension field variables (Permutation) with conditions
        // Using FirstRow (degree 1) to test condition contribution
        let air = MockAir {
            variable: VariableKind::Permutation,
            exponent: 2,
            condition: Condition::FirstRow,
            ..Default::default()
        };
        let degree = compute_degree(&air);
        assert_eq!(degree, air.expected_degree());
        assert_eq!(degree, 3); // is_first_row (1) * permutation^2 (2) = degree 3
    }

    // ==================== Symbolic Constraint Capture Tests ====================

    #[test]
    fn test_periodic_constraint_captured() {
        // Verify that a constraint using periodic values is actually recorded
        // in the symbolic output (not just degree computation).
        let air = MockAir {
            variable: VariableKind::Periodic,
            exponent: 1,
            condition: Condition::None,
            ..Default::default()
        };
        let (base_constraints, _) = get_all_symbolic_constraints::<BabyBear, EF, _>(
            &air,
            1,
            1,
            1,
            0,
            1,
            air.num_periodic_columns(),
        );
        assert_eq!(
            base_constraints.len(),
            1,
            "periodic constraint should be captured"
        );
        assert!(
            matches!(&base_constraints[0], SymbolicExpression::Variable(v) if v.entry == Entry::Periodic),
            "constraint should reference a periodic variable"
        );
    }

    // ==================== AirWithPeriodicColumns Trait Tests ====================

    #[test]
    fn test_periodic_columns_matrix_different_periods() {
        // MockAir::default() has 2 columns with periods 2 and 4
        let air = MockAir::default();

        assert_eq!(air.num_periodic_columns(), 2);
        assert_eq!(air.get_column_period(0), Some(2));
        assert_eq!(air.get_column_period(1), Some(4));
        assert_eq!(air.get_max_column_period(), Some(4));

        let matrix = air.periodic_columns_matrix().expect("should have matrix");
        assert_eq!(matrix.height(), 4, "should extend to max period");
        assert_eq!(matrix.width(), 2);

        // Column 0 repeats: [1, 2, 1, 2], Column 1: [10, 20, 30, 40]
        let expected = [
            BabyBear::new(1),
            BabyBear::new(10),
            BabyBear::new(2),
            BabyBear::new(20),
            BabyBear::new(1),
            BabyBear::new(30),
            BabyBear::new(2),
            BabyBear::new(40),
        ];
        assert_eq!(matrix.values, expected);
    }
}
