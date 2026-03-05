//! Periodic column traits (copied from upstream `0xMiden/Plonky3` branch `periodic-air-builder`,
//! not yet available in published p3-air 0.4.2).

use alloc::vec::Vec;

use p3_air::{AirBuilder, BaseAir, FilteredAirBuilder};
use p3_matrix::dense::RowMajorMatrix;

/// An extension of `BaseAir` that includes support for periodic columns.
///
/// Periodic columns are columns whose values repeat with a fixed period that divides the
/// trace length. They are derived from public parameters and are never committed as part
/// of the trace — instead, both prover and verifier compute them from the data provided here.
pub trait AirWithPeriodicColumns<F>: BaseAir<F> {
    /// Return the periodic table data: a list of columns, each a `Vec<F>` of evaluations.
    ///
    /// Each inner `Vec<F>` represents one periodic column. Its length is the period of
    /// that column, and the entries are the evaluations over a subgroup of that order.
    fn periodic_columns(&self) -> &[Vec<F>];

    /// Return the number of periodic columns.
    fn num_periodic_columns(&self) -> usize {
        self.periodic_columns().len()
    }

    /// Return the period of the column at index `col_idx`, if it exists.
    fn get_column_period(&self, col_idx: usize) -> Option<usize> {
        self.periodic_columns().get(col_idx).map(|col| col.len())
    }

    /// Return the maximum period among all periodic columns, or `None` if there are none.
    fn get_max_column_period(&self) -> Option<usize> {
        self.periodic_columns().iter().map(|col| col.len()).max()
    }

    /// Return a matrix with all periodic columns extended to a common height.
    ///
    /// Columns with smaller periods are repeated cyclically to fill the extended domain.
    /// Returns `None` if there are no periodic columns.
    fn periodic_columns_matrix(&self) -> Option<RowMajorMatrix<F>>
    where
        F: Clone + Send + Sync,
    {
        let cols = self.periodic_columns();
        if cols.is_empty() {
            return None;
        }

        let max_period = self.get_max_column_period()?;
        let num_cols = cols.len();

        let mut values = Vec::with_capacity(max_period * num_cols);
        for row in 0..max_period {
            for col in cols {
                let period = col.len();
                values.push(col[row % period].clone());
            }
        }

        Some(RowMajorMatrix::new(values, num_cols))
    }
}

/// Trait for builders supporting periodic columns.
///
/// Periodic columns are columns whose values repeat with a period p dividing the trace
/// length. At row i, the value of periodic column j equals `periodic_table[j][i mod p]`.
pub trait PeriodicAirBuilder: AirBuilder {
    /// Variable type for periodic column values.
    type PeriodicVar: Into<Self::Expr> + Copy;

    /// Return the evaluations of periodic columns at the current row.
    fn periodic_values(&self) -> &[Self::PeriodicVar];
}

impl<AB: PeriodicAirBuilder> PeriodicAirBuilder for FilteredAirBuilder<'_, AB> {
    type PeriodicVar = AB::PeriodicVar;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        self.inner.periodic_values()
    }
}
