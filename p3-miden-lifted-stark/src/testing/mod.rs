//! Unified testing infrastructure for the lifted STARK crate.
//!
//! Provides three complete configuration variants, each containing everything
//! needed to test at any level (LMCS, PCS, or full STARK):
//!
//! - `configs::goldilocks_poseidon2`
//! - `configs::goldilocks_keccak`
//! - `configs::goldilocks_blake3_192`
//!
//! Also provides shared fixtures, matrix generation utilities, and test helpers.

pub mod airs;
#[cfg(feature = "std")]
pub mod bench_configs;
pub mod configs;
pub mod params;
#[cfg(feature = "std")]
pub mod stats;

#[cfg(test)]
mod test_aux_shape;
#[cfg(test)]
mod test_bus;
#[cfg(test)]
mod test_multi_aux_alignment;
#[cfg(test)]
mod test_tiny_air;

// Re-export commonly used params at the module level for convenience.
use alloc::{vec, vec::Vec};

use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
pub use params::{
    BENCH_PCS_PARAMS, FRI_FOLD_ARITY_2, FRI_FOLD_ARITY_4, FRI_FOLD_ARITY_8, LOG_HEIGHTS,
    PARALLEL_STR, QC_CONSTRAINT_DEGREE, QC_PCS_PARAMS, RELATIVE_SPECS, TEST_SEED,
};
use rand::{
    RngExt, SeedableRng,
    distr::{Distribution, StandardUniform},
    rngs::SmallRng,
};

use crate::lmcs::utils::aligned_len;

// =============================================================================
// Matrix generation
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
    log_max_height: u8,
) -> Vec<Vec<RowMajorMatrix<F>>>
where
    StandardUniform: Distribution<F>,
{
    let rng = &mut SmallRng::seed_from_u64(TEST_SEED);
    let max_height = 1 << log_max_height as usize;

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

/// Calculate total elements across all matrices.
pub fn total_elements<F: Field>(matrix_groups: &[Vec<RowMajorMatrix<F>>]) -> u64 {
    matrix_groups
        .iter()
        .flat_map(|g| g.iter())
        .map(|m| {
            let dims = m.dimensions();
            (dims.height * dims.width) as u64
        })
        .sum()
}

// =============================================================================
// LMCS test helpers
// =============================================================================

/// Sample `count` random indices in `[0, upper)`.
pub fn sample_indices<R: rand::Rng>(rng: &mut R, upper: usize, count: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        indices.push(rng.random_range(0..upper));
    }
    indices
}

/// Common matrix group scenarios for testing lifting with varying heights.
///
/// Each scenario is a list of (height, width) pairs, sorted by ascending height.
/// The `rate` parameter controls the RATE-based width scenarios.
///
/// # Parameters
/// - `pack_width`: The SIMD packing width (e.g., `P::WIDTH` for packed field)
/// - `rate`: The sponge rate for width alignment scenarios
pub fn matrix_scenarios<P: PackedValue>(rate: usize) -> Vec<Vec<(usize, usize)>> {
    let pack_width = P::WIDTH.max(2);
    vec![
        // Single matrices
        vec![(1, 1)],
        vec![(1, rate - 1)],
        // Multiple heights (must be ascending)
        vec![(2, 3), (4, 5), (8, rate)],
        vec![(1, 5), (1, 3), (2, 7), (4, 1), (8, rate + 1)],
        // Packing boundary tests
        vec![
            (pack_width / 2, rate - 1),
            (pack_width, rate),
            (pack_width * 2, rate + 3),
        ],
        vec![(pack_width, rate + 5), (pack_width * 2, 25)],
        vec![
            (1, rate * 2),
            (pack_width / 2, rate * 2 - 1),
            (pack_width, rate * 2),
            (pack_width * 2, rate * 3 - 2),
        ],
        // Same-height matrices
        vec![(4, rate - 1), (4, rate), (8, rate + 3), (8, rate * 2)],
        // Single tall matrix
        vec![(pack_width * 2, rate - 1)],
    ]
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

/// Upsample matrix to exactly `target_height` rows via nearest-neighbor repetition.
///
/// Each original row is repeated `target_height / height` times.
/// Requires `target_height >= height` and both be powers of two.
///
/// This is the explicit form of the "lifting" operation used in LMCS, where smaller
/// matrices are virtually extended to match the height of the tallest matrix.
pub fn upsample_matrix<F: Clone + Send + Sync>(
    matrix: &impl p3_matrix::Matrix<F>,
    target_height: usize,
) -> RowMajorMatrix<F> {
    let height = matrix.height();
    assert!(target_height >= height);
    assert!(height.is_power_of_two() && target_height.is_power_of_two());

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

// =============================================================================
// define_test_config! macro
// =============================================================================

/// Generates LMCS type aliases and channel helper functions for a test config.
///
/// Requires these items in scope from the base config module:
/// `Felt`, `Sponge`, `Compress`, `Challenger`, `test_challenger`
///
/// Also requires `Lmcs` to be defined as a type alias in the invoking module.
macro_rules! define_lmcs_test_helpers {
    () => {
        use $crate::lmcs::Lmcs as LmcsTrait;

        pub type TestTree = <Lmcs as LmcsTrait>::Tree<p3_matrix::dense::RowMajorMatrix<Felt>>;
        pub type TestCommitment = <Lmcs as LmcsTrait>::Commitment;
        pub type TestTranscriptData = p3_miden_transcript::TranscriptData<Felt, TestCommitment>;
        pub type TestDigest = <Challenger as p3_challenger::CanFinalizeDigest>::Digest;
        pub type TestProverChannel =
            p3_miden_transcript::ProverTranscript<Felt, TestCommitment, Challenger>;
        pub type TestVerifierChannel<'a> =
            p3_miden_transcript::VerifierTranscript<'a, Felt, TestCommitment, Challenger>;

        pub fn prover_channel() -> TestProverChannel {
            p3_miden_transcript::ProverTranscript::new(test_challenger())
        }

        pub fn prover_channel_with_commitment(commitment: &TestCommitment) -> TestProverChannel {
            let mut challenger = test_challenger();
            p3_challenger::CanObserve::observe(&mut challenger, commitment.clone());
            p3_miden_transcript::ProverTranscript::new(challenger)
        }

        pub fn verifier_channel(data: &TestTranscriptData) -> TestVerifierChannel<'_> {
            p3_miden_transcript::VerifierTranscript::from_data(test_challenger(), data)
        }

        pub fn verifier_channel_with_commitment<'a>(
            data: &'a TestTranscriptData,
            commitment: &TestCommitment,
        ) -> TestVerifierChannel<'a> {
            let mut challenger = test_challenger();
            p3_challenger::CanObserve::observe(&mut challenger, commitment.clone());
            p3_miden_transcript::VerifierTranscript::from_data(challenger, data)
        }
    };
}

pub(crate) use define_lmcs_test_helpers;

// =============================================================================
// Internal re-exports for benchmarks
// =============================================================================
pub use crate::pcs::{
    deep::interpolate::PointQuotients, fri::fold::FriFold, prover::open_with_channel,
    utils::bit_reversed_coset_points,
};
pub use crate::prover::quotient::commit_quotient;
