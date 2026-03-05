//! Local copies of [`ExtensionBuilder`] and [`PermutationAirBuilder`] with a stricter
//! `Algebra<Expr>` bound.
//!
//! The upstream `p3_air::ExtensionBuilder` requires `ExprEF: From<Self::Expr>`, but we
//! need `ExprEF: Algebra<Self::Expr>` for full arithmetic interop between base and
//! extension expression types.
//!
//! Since `p3_air::PermutationAirBuilder` inherits from `p3_air::ExtensionBuilder`, we
//! must provide our own copy to keep the supertrait chain internally consistent.
//!
//! Both traits mirror the upstream API exactly; only the `ExprEF` bound differs.
//!
//! Temporary: will be removed once upstream p3-air adopts the `Algebra` bound.

use p3_air::{AirBuilder, FilteredAirBuilder};
use p3_field::{ExtensionField, Field};
use p3_matrix::Matrix;

use p3_field::{Algebra, PrimeCharacteristicRing};

// ─── ExtensionBuilder ────────────────────────────────────────────────

/// Extension builder with `ExprEF: Algebra<Expr>` (upstream only requires `From<Expr>`).
///
/// The stronger `Algebra<Expr>` bound provides `Add`, `Sub`, `Mul`, and `From` between
/// base and extension expressions — everything needed for natural mixed-field constraints.
pub trait ExtensionBuilder: AirBuilder<F: Field> {
    /// Extension field type.
    type EF: ExtensionField<Self::F>;

    /// Expression type over extension field elements.
    ///
    /// `Algebra<Self::Expr>` enables mixed base/extension arithmetic in constraints.
    /// `Algebra<Self::EF>` enables multiplying expressions by extension-field constants.
    type ExprEF: Algebra<Self::Expr> + Algebra<Self::EF>;

    /// Variable type over extension field elements.
    type VarEF: Into<Self::ExprEF> + Copy + Send + Sync;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>;

    /// Assert that two extension field expressions are equal.
    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        self.assert_zero_ext(x.into() - y.into());
    }

    /// Assert that an extension field expression is one.
    fn assert_one_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_eq_ext(x, Self::ExprEF::ONE);
    }
}

// ─── PermutationAirBuilder ───────────────────────────────────────────

/// Permutation air builder inheriting from our local [`ExtensionBuilder`].
///
/// Identical to `p3_air::PermutationAirBuilder` except it extends our `ExtensionBuilder`
/// (with `Algebra<Expr>` bound) instead of upstream's.
pub trait PermutationAirBuilder: ExtensionBuilder {
    /// Matrix type over extension field variables representing a permutation.
    type MP: Matrix<Self::VarEF>;

    /// Randomness variable type used in permutation commitments.
    type RandomVar: Into<Self::ExprEF> + Copy + Send + Sync;

    /// Get the permutation trace matrix.
    fn permutation(&self) -> Self::MP;

    /// Get the randomness values for the permutation.
    fn permutation_randomness(&self) -> &[Self::RandomVar];
}

// ─── FilteredAirBuilder impls ────────────────────────────────────────

impl<AB: ExtensionBuilder> ExtensionBuilder for FilteredAirBuilder<'_, AB> {
    type EF = AB::EF;
    type ExprEF = AB::ExprEF;
    type VarEF = AB::VarEF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.inner.assert_zero_ext(x.into() * self.condition());
    }
}

impl<AB: PermutationAirBuilder> PermutationAirBuilder for FilteredAirBuilder<'_, AB> {
    type MP = AB::MP;
    type RandomVar = AB::RandomVar;

    fn permutation(&self) -> Self::MP {
        self.inner.permutation()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.inner.permutation_randomness()
    }
}
