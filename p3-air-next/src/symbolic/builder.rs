use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::symbolic::SymbolicExpr;
use crate::symbolic::expression::BaseLeaf;
use crate::symbolic::expression_ext::SymbolicExpressionExt;
use crate::symbolic::variable::{BaseEntry, ExtEntry, SymbolicVariableExt};
use crate::{
    Air, AirBuilder, ExtensionBuilder, PeriodicAirBuilder, PermutationAirBuilder,
    SymbolicExpression, SymbolicVariable,
};

#[instrument(skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(air: &A, preprocessed_width: usize) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_max_constraint_degree_extension(air, preprocessed_width, 0, 0, 0)
}

#[instrument(
    name = "infer base and extension constraint degree",
    skip_all,
    level = "debug"
)]
pub fn get_max_constraint_degree_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
    num_permutation_values: usize,
) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let (base_constraints, extension_constraints) = get_all_symbolic_constraints(
        air,
        preprocessed_width,
        permutation_width,
        num_permutation_challenges,
        num_permutation_values,
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

#[instrument(
    name = "evaluate base constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints<F, A>(
    air: &A,
    preprocessed_width: usize,
) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        air.num_public_values(),
        0,
        0,
        0,
        0,
    );
    air.eval(&mut builder);
    builder.base_constraints()
}

#[instrument(
    name = "evaluate extension constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints_extension<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
    num_permutation_values: usize,
) -> Vec<SymbolicExpressionExt<F, EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        air.num_public_values(),
        permutation_width,
        num_permutation_challenges,
        num_permutation_values,
        0,
    );
    air.eval(&mut builder);
    builder.extension_constraints()
}

#[instrument(
    name = "evaluate all constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_all_symbolic_constraints<F, EF, A>(
    air: &A,
    preprocessed_width: usize,
    permutation_width: usize,
    num_permutation_challenges: usize,
    num_permutation_values: usize,
) -> (
    Vec<SymbolicExpression<F>>,
    Vec<SymbolicExpressionExt<F, EF>>,
)
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let mut builder = SymbolicAirBuilder::new(
        preprocessed_width,
        air.width(),
        air.num_public_values(),
        permutation_width,
        num_permutation_challenges,
        num_permutation_values,
        0,
    );
    air.eval(&mut builder);
    (builder.base_constraints(), builder.extension_constraints())
}

/// An [`AirBuilder`] for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field, EF: ExtensionField<F> = F> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    periodic: Vec<SymbolicVariable<F>>,
    base_constraints: Vec<SymbolicExpression<F>>,
    permutation: RowMajorMatrix<SymbolicVariableExt<F, EF>>,
    permutation_challenges: Vec<SymbolicVariableExt<F, EF>>,
    permutation_values: Vec<SymbolicVariableExt<F, EF>>,
    extension_constraints: Vec<SymbolicExpressionExt<F, EF>>,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicAirBuilder<F, EF> {
    pub fn new(
        preprocessed_width: usize,
        width: usize,
        num_public_values: usize,
        permutation_width: usize,
        num_permutation_challenges: usize,
        num_permutation_values: usize,
        num_periodic_columns: usize,
    ) -> Self {
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width).map(move |index| {
                    SymbolicVariable::new(BaseEntry::Preprocessed { offset }, index)
                })
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..width)
                    .map(move |index| SymbolicVariable::new(BaseEntry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(BaseEntry::Public, index))
            .collect();
        let periodic = (0..num_periodic_columns)
            .map(|index| SymbolicVariable::new(BaseEntry::Periodic, index))
            .collect();
        let perm_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..permutation_width).map(move |index| {
                    SymbolicVariableExt::new(ExtEntry::Permutation { offset }, index)
                })
            })
            .collect();
        let permutation = RowMajorMatrix::new(perm_values, permutation_width);
        let permutation_challenges = (0..num_permutation_challenges)
            .map(|index| SymbolicVariableExt::new(ExtEntry::Challenge, index))
            .collect();
        let permutation_values = (0..num_permutation_values)
            .map(|index| SymbolicVariableExt::new(ExtEntry::PermutationValue, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, width),
            public_values,
            periodic,
            base_constraints: vec![],
            permutation,
            permutation_challenges,
            permutation_values,
            extension_constraints: vec![],
        }
    }

    pub fn extension_constraints(&self) -> Vec<SymbolicExpressionExt<F, EF>> {
        self.extension_constraints.clone()
    }

    pub fn base_constraints(&self) -> Vec<SymbolicExpression<F>> {
        self.base_constraints.clone()
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for SymbolicAirBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type M = RowMajorMatrix<Self::Var>;
    type PublicVar = SymbolicVariable<F>;

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
        SymbolicExpr::Leaf(BaseLeaf::IsFirstRow)
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpr::Leaf(BaseLeaf::IsLastRow)
    }

    /// # Panics
    /// This function panics if `size` is not `2`.
    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpr::Leaf(BaseLeaf::IsTransition)
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type EF = EF;
    type ExprEF = SymbolicExpressionExt<F, EF>;
    type VarEF = SymbolicVariableExt<F, EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.extension_constraints.push(x.into());
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type MP = RowMajorMatrix<Self::VarEF>;

    type RandomVar = SymbolicVariableExt<F, EF>;

    type PermutationVal = SymbolicVariableExt<F, EF>;

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

impl<F: Field, EF: ExtensionField<F>> PeriodicAirBuilder for SymbolicAirBuilder<F, EF> {
    type PeriodicVar = SymbolicVariable<F>;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &self.periodic
    }
}
