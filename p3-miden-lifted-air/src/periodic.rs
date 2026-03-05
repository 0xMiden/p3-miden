//! Periodic column data trait — extends `BaseAir` with periodic table metadata.
//!
//! [`PeriodicAirBuilder`](p3_air::PeriodicAirBuilder) (the builder trait for evaluating
//! periodic column values at each row) lives in `p3-air-next`. This module only provides
//! [`AirWithPeriodicColumns`], which carries the *data* (column periods and evaluations).

use alloc::vec::Vec;

use p3_air::BaseAir;
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
