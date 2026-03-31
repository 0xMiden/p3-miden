//! Unified testing infrastructure for the lifted STARK crate.
//!
//! Provides three complete configuration variants, each containing everything
//! needed to test at any level (LMCS, PCS, or full STARK):
//!
//! - [`configs::goldilocks_poseidon2`]
//! - [`configs::goldilocks_keccak`]
//! - [`configs::goldilocks_blake3_192`]
//!
//! Also provides shared fixtures, matrix generation utilities, and test helpers.

pub mod airs;
#[cfg(feature = "std")]
pub mod bench_configs;
pub mod configs;
#[cfg(feature = "std")]
pub mod stats;

use alloc::{vec, vec::Vec};

use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use rand::{
    RngExt, SeedableRng,
    distr::{Distribution, StandardUniform},
    rngs::SmallRng,
};

use crate::lmcs::utils::aligned_len;

// =============================================================================
// Fixtures
// =============================================================================

/// Standard seed for reproducible tests/benchmarks.
pub const TEST_SEED: u64 = 2025;

/// Alias for benchmark seed (same value as TEST_SEED).
pub const BENCH_SEED: u64 = TEST_SEED;

/// Standard log heights for benchmarking: 2^16, 2^18, 2^20 leaves.
pub const LOG_HEIGHTS: &[u8] = &[16, 18, 20];

/// Standard relative specs for benchmark matrix groups.
///
/// Each inner slice is a separate commitment group.
/// Tuple format: `(offset_from_max, width)` where `log_height = log_max_height - offset`.
///
/// This gives realistic matrix configurations similar to STARK traces:
/// - Group 0: Main trace columns at various heights
/// - Group 1: Auxiliary/permutation columns
/// - Group 2: Quotient polynomial chunks
pub const RELATIVE_SPECS: &[&[(usize, usize)]] = &[
    &[(4, 10), (2, 100), (0, 50)],
    &[(4, 8), (2, 20), (0, 20)],
    &[(0, 16)],
];

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
    let rng = &mut SmallRng::seed_from_u64(BENCH_SEED);
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

// =============================================================================
// define_test_config! macro
// =============================================================================

/// Generates LMCS type aliases and channel helper functions for a test config.
///
/// Requires these items in scope from the base config module:
/// `F`, `Sponge`, `Compress`, `Challenger`, `test_challenger`
///
/// Also requires `Lmcs` to be defined as a type alias in the invoking module.
macro_rules! define_lmcs_test_helpers {
    () => {
        use $crate::lmcs::Lmcs as LmcsTrait;

        pub type TestTree = <Lmcs as LmcsTrait>::Tree<p3_matrix::dense::RowMajorMatrix<F>>;
        pub type TestCommitment = <Lmcs as LmcsTrait>::Commitment;
        pub type TestTranscriptData = p3_miden_transcript::TranscriptData<F, TestCommitment>;
        pub type TestDigest = <Challenger as p3_challenger::CanFinalizeDigest>::Digest;
        pub type TestProverChannel =
            p3_miden_transcript::ProverTranscript<F, TestCommitment, Challenger>;
        pub type TestVerifierChannel<'a> =
            p3_miden_transcript::VerifierTranscript<'a, F, TestCommitment, Challenger>;

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
// PCS re-exports for benchmarks
// =============================================================================
/// PCS prover entry point (re-exported for benchmarks).
pub use crate::pcs::prover::open_with_channel;

/// PCS utilities for benchmarks.
pub mod pcs_utils {
    pub use crate::pcs::{
        deep::interpolate::PointQuotients, fri::FriFold, utils::bit_reversed_coset_points,
    };
}
