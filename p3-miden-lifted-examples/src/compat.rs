//! Compatibility adapter: fork [`AirBuilder`] → crates.io [`p3_air::AirBuilder`].
//!
//! The fork's `AirBuilder` (from `p3-air-next`, re-exported by `p3-miden-lifted-air`)
//! is a strict superset of the crates.io 0.4.2 `AirBuilder`. This adapter wraps a fork
//! builder to satisfy the crates.io trait bound, enabling delegation to upstream AIR types
//! (e.g. `Blake3Air`, `KeccakAir`) that implement `p3_air::Air<AB: p3_air::AirBuilder>`.

use p3_miden_lifted_air::AirBuilder;

/// Wraps a fork [`AirBuilder`] to satisfy the crates.io [`p3_air::AirBuilder`] bound.
pub(crate) struct UpstreamCompat<'a, AB>(pub &'a mut AB);

impl<AB: AirBuilder> p3_air::AirBuilder for UpstreamCompat<'_, AB> {
    type F = AB::F;
    type Expr = AB::Expr;
    type Var = AB::Var;
    type M = AB::M;

    fn main(&self) -> Self::M {
        self.0.main()
    }

    fn is_first_row(&self) -> Self::Expr {
        self.0.is_first_row()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.0.is_last_row()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        self.0.is_transition_window(size)
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.0.assert_zero(x)
    }
}
