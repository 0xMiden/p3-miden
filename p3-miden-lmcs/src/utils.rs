//! Utility functions for LMCS operations.

use alloc::vec::Vec;
use core::array;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use serde::{Deserialize, Serialize};

// ============================================================================
// RowList
// ============================================================================

/// Flat storage of variable-width rows.
///
/// In a STARK proof, each row typically holds one committed matrix's evaluations at a
/// leaf index queried by the verifier as part of the low-degree test (LDT). Matrices
/// have different widths because they encode different sets of constraint polynomials
/// (e.g., main trace vs auxiliary trace).
///
/// Stores all elements contiguously in a single `Vec<T>`, with a separate `Vec<usize>`
/// tracking the width of each row. This avoids N+1 heap allocations compared to
/// `Vec<Vec<T>>` and enables efficient flat iteration.
///
/// Invariant: `widths.iter().sum::<usize>() == elems.len()`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: Deserialize<'de>"))]
pub struct RowList<T> {
    elems: Vec<T>,
    widths: Vec<usize>,
}

impl<T> RowList<T> {
    /// Create a `RowList` from raw elements and widths.
    ///
    /// # Panics
    ///
    /// Panics if `widths.iter().sum() != elems.len()`.
    pub fn new(elems: Vec<T>, widths: &[usize]) -> Self {
        let expected: usize = widths.iter().sum();
        assert_eq!(
            elems.len(),
            expected,
            "RowList invariant violated: {} elems but widths sum to {}",
            elems.len(),
            expected,
        );
        Self {
            elems,
            widths: widths.to_vec(),
        }
    }

    /// Build a `RowList` from an iterator of row-like items.
    ///
    /// Accepts anything that derefs to `[T]`: owned `Vec<T>`, `&Vec<T>`, `&[T]`, etc.
    pub fn from_rows<R: AsRef<[T]>>(rows: impl IntoIterator<Item = R>) -> Self
    where
        T: Clone,
    {
        let mut elems = Vec::new();
        let mut widths = Vec::new();
        for row in rows {
            let row = row.as_ref();
            widths.push(row.len());
            elems.extend_from_slice(row);
        }
        Self { elems, widths }
    }

    /// Contiguous element slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.elems
    }

    /// Iterate over all elements by value.
    #[inline]
    pub fn iter_values(&self) -> impl Iterator<Item = T> + '_
    where
        T: Copy,
    {
        self.elems.iter().copied()
    }

    /// Number of rows.
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.widths.len()
    }

    /// Iterate over rows as slices.
    pub fn iter_rows(&self) -> impl Iterator<Item = &[T]> {
        let mut offset = 0;
        self.widths.iter().map(move |&w| {
            let row = &self.elems[offset..offset + w];
            offset += w;
            row
        })
    }
}

impl<T: Copy + Default> RowList<T> {
    /// Iterate over all elements with each row zero-padded to a multiple of `alignment`.
    ///
    /// Alignment matches the cryptographic sponge's absorption rate. Both prover and
    /// verifier must hash identical padded data for the Merkle commitment to verify,
    /// so OOD evaluations sent over the transcript use the same padding convention.
    ///
    /// Yields the original row elements followed by implicit zeros, without allocating
    /// a padded copy.
    pub fn iter_aligned(&self, alignment: usize) -> impl Iterator<Item = T> + '_ {
        self.iter_rows().flat_map(move |row| {
            let padding = aligned_len(row.len(), alignment) - row.len();
            row.iter()
                .copied()
                .chain(core::iter::repeat_n(T::default(), padding))
        })
    }
}

impl<T: Default + Clone> RowList<T> {
    /// Build a `RowList` from an iterator of row-like items, padding each to `alignment`.
    pub fn from_rows_aligned<R: AsRef<[T]>>(
        rows: impl IntoIterator<Item = R>,
        alignment: usize,
    ) -> Self {
        let mut elems = Vec::new();
        let mut widths = Vec::new();
        for row in rows {
            let row = row.as_ref();
            let padded_len = aligned_len(row.len(), alignment);
            widths.push(padded_len);
            elems.extend_from_slice(row);
            elems.resize(elems.len() + (padded_len - row.len()), T::default());
        }
        Self { elems, widths }
    }
}

/// Extension trait for `PackedValue` providing columnar pack/unpack operations.
///
/// These methods perform transpose operations on packed data, useful for
/// SIMD-parallelized Merkle tree construction.
pub trait PackedValueExt: PackedValue {
    /// Pack columns from `WIDTH` rows of scalar values.
    ///
    /// Given `WIDTH` rows of `N` scalar values, extract each column and pack it
    /// into a single packed value. This performs a transpose operation.
    #[inline]
    #[must_use]
    fn pack_columns<const N: usize>(rows: &[[Self::Value; N]]) -> [Self; N] {
        assert_eq!(rows.len(), Self::WIDTH);
        array::from_fn(|col| Self::from_fn(|lane| rows[lane][col]))
    }
}

// Blanket implementation for all PackedValue types
impl<T: PackedValue> PackedValueExt for T {}

/// Compute the aligned length for `len` given an alignment.
#[inline]
pub const fn aligned_len(len: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        len
    } else {
        len.next_multiple_of(alignment)
    }
}

/// Return widths aligned to `alignment`.
pub fn aligned_widths(widths: impl IntoIterator<Item = usize>, alignment: usize) -> Vec<usize> {
    widths
        .into_iter()
        .map(|w| aligned_len(w, alignment))
        .collect()
}

/// Pad a row with `Default::default()` so its length is a multiple of `alignment`.
///
/// This is a formatting convention for transcript hints; LMCS does not enforce that
/// padded values are zero unless the caller checks them.
pub fn pad_row_to_alignment<F: Default>(mut row: Vec<F>, alignment: usize) -> Vec<F> {
    debug_assert!(alignment > 0, "alignment must be non-zero");
    let padded_len = aligned_len(row.len(), alignment);
    row.resize_with(padded_len, || F::default());
    row
}

/// Upsample matrix to exactly `target_height` rows via nearest-neighbor repetition.
///
/// Each original row is repeated `target_height / height` times.
/// Requires `target_height >= height` and both be powers of two.
///
/// This is the explicit form of the "lifting" operation used in LMCS, where smaller
/// matrices are virtually extended to match the height of the tallest matrix.
pub fn upsample_matrix<F: Clone + Send + Sync>(
    matrix: &impl Matrix<F>,
    target_height: usize,
) -> RowMajorMatrix<F> {
    let height = matrix.height();
    debug_assert!(target_height >= height);
    debug_assert!(height.is_power_of_two() && target_height.is_power_of_two());

    let repeat_factor = target_height / height;
    let width = matrix.width();

    let mut values = Vec::with_capacity(target_height * width);
    for row in matrix.rows() {
        let row_vec: Vec<F> = row.collect();
        for _ in 0..repeat_factor {
            values.extend(row_vec.iter().cloned());
        }
    }

    RowMajorMatrix::new(values, width)
}
