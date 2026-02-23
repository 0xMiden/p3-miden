//! Matrix generation utilities for tests and benchmarks.

use alloc::vec::Vec;

use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;

use crate::fixtures::BENCH_SEED;

// =============================================================================
// Benchmark matrix generation
// =============================================================================

/// Generate benchmark matrices from relative specs.
///
/// Creates matrices with heights relative to `max_height = 1 << log_max_height`.
/// Each spec `(offset, width)` creates a matrix with:
/// - height = `max_height >> offset`
/// - width = `width`
///
/// Matrices in each group are sorted by ascending height.
pub fn generate_matrices_from_specs<F: Field>(
    specs: &[&[(usize, usize)]],
    log_max_height: usize,
) -> Vec<Vec<RowMajorMatrix<F>>>
where
    StandardUniform: Distribution<F>,
{
    let rng = &mut SmallRng::seed_from_u64(BENCH_SEED);
    let max_height = 1usize << log_max_height;

    specs
        .iter()
        .map(|group_specs| {
            let mut matrices: Vec<RowMajorMatrix<F>> = group_specs
                .iter()
                .map(|&(offset, width)| {
                    let height = max_height >> offset;
                    RowMajorMatrix::rand(rng, height, width)
                })
                .collect();
            // Sort by ascending height (required by LMCS)
            matrices.sort_by_key(|m| m.height());
            matrices
        })
        .collect()
}

/// Generate a single flat matrix for FRI fold benchmarks.
pub fn generate_flat_matrix<F: Field>(log_height: usize, width: usize) -> RowMajorMatrix<F>
where
    StandardUniform: Distribution<F>,
{
    let rng = &mut SmallRng::seed_from_u64(BENCH_SEED);
    RowMajorMatrix::rand(rng, 1 << log_height, width)
}

/// Calculate total elements across all matrices.
pub fn total_elements<F: Clone + Send + Sync>(matrix_groups: &[Vec<RowMajorMatrix<F>>]) -> u64 {
    matrix_groups
        .iter()
        .flat_map(|g| g.iter())
        .map(|m| {
            let dims = m.dimensions();
            (dims.height * dims.width) as u64
        })
        .sum()
}

/// Calculate total elements for a flat matrix list.
pub fn total_elements_flat<F: Clone + Send + Sync>(matrices: &[RowMajorMatrix<F>]) -> u64 {
    matrices
        .iter()
        .map(|m| {
            let dims = m.dimensions();
            (dims.height * dims.width) as u64
        })
        .sum()
}

// =============================================================================
// Test matrix utilities
// =============================================================================

/// Generate a matrix of LDE evaluations for random low-degree polynomials.
///
/// Each column is a polynomial of degree `poly_degree`, evaluated on the coset gK
/// in bit-reversed order, where g = `shift` and K is a subgroup of order `lde_size`.
///
/// The coset evaluation is computed by scaling coefficients: for f(X) = Σ c_j X^j,
/// the coset evaluations f(gX) = Σ (c_j g^j) X^j are obtained by DFT of scaled coefficients.
pub fn random_lde_matrix<F, V>(
    rng: &mut SmallRng,
    log_poly_degree: usize,
    log_blowup: usize,
    num_columns: usize,
    shift: F,
) -> RowMajorMatrix<V>
where
    F: TwoAdicField,
    V: BasedVectorSpace<F> + Clone + Send + Sync + Default,
    StandardUniform: Distribution<V>,
{
    let poly_degree = 1 << log_poly_degree;
    let dft = Radix2DFTSmallBatch::<F>::default();

    let evals = RowMajorMatrix::rand(rng, poly_degree, num_columns);
    let lde = dft.coset_lde_algebra_batch(evals, log_blowup, shift);
    lde.bit_reverse_rows().to_row_major_matrix()
}

/// Concatenate matrices horizontally, padding each to a multiple of `R`.
///
/// All matrices are lifted to the maximum height first.
pub fn concatenate_matrices<F: Field + PrimeCharacteristicRing, const R: usize>(
    matrices: &[RowMajorMatrix<F>],
) -> RowMajorMatrix<F> {
    let max_height = matrices.last().unwrap().height();
    let width: usize = matrices.iter().map(|m| aligned_len(m.width(), R)).sum();

    let concatenated_data: Vec<_> = (0..max_height)
        .flat_map(|idx| {
            matrices.iter().flat_map(move |m| {
                let mut row = m.row_slice(idx).unwrap().to_vec();
                let padded_width = aligned_len(row.len(), R);
                row.resize(padded_width, F::ZERO);
                row
            })
        })
        .collect();
    RowMajorMatrix::new(concatenated_data, width)
}

#[inline]
const fn aligned_len(len: usize, alignment: usize) -> usize {
    if alignment <= 1 {
        len
    } else {
        len.next_multiple_of(alignment)
    }
}
