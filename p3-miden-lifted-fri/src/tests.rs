//! Common test fixtures and end-to-end tests for the lifted FRI PCS.
//!
//! Re-exports test fixtures from `p3_miden_dev_utils` for use in tests.

use alloc::vec;
use alloc::vec::Vec;

use p3_challenger::CanObserve;
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::{Lmcs, LmcsConfig, LmcsTree};
use p3_miden_transcript::{ProverTranscript, TranscriptData, VerifierChannel, VerifierTranscript};
use p3_util::log2_strict_usize;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use crate::deep::DeepParams;
use crate::fri::{FriFold, FriParams};
use crate::{PcsParams, prover::open_with_channel, verifier::verify_with_channel};

pub use p3_miden_dev_utils::configs::baby_bear_poseidon2::*;
pub use p3_miden_dev_utils::matrix::random_lde_matrix;

pub type BaseLmcs = LmcsConfig<P, P, Sponge, Compress, WIDTH, DIGEST>;
pub type TestTree = <BaseLmcs as Lmcs>::Tree<RowMajorMatrix<F>>;
pub type TestCommitment = <BaseLmcs as Lmcs>::Commitment;
pub type TestTranscriptData = TranscriptData<F, TestCommitment>;
pub type TestProverChannel = ProverTranscript<F, TestCommitment, Challenger>;
pub type TestVerifierChannel<'a> = VerifierTranscript<'a, F, TestCommitment, Challenger>;

pub fn test_lmcs() -> BaseLmcs {
    let (_, sponge, compress) = test_components();
    LmcsConfig::new(sponge, compress)
}

pub fn prover_channel() -> TestProverChannel {
    ProverTranscript::new(test_challenger())
}

pub fn prover_channel_with_commitment(commitment: &TestCommitment) -> TestProverChannel {
    let mut challenger = test_challenger();
    challenger.observe(*commitment);
    ProverTranscript::new(challenger)
}

pub fn verifier_channel(data: &'_ TestTranscriptData) -> TestVerifierChannel<'_> {
    VerifierTranscript::from_data(test_challenger(), data)
}

pub fn verifier_channel_with_commitment<'a>(
    data: &'a TestTranscriptData,
    commitment: &TestCommitment,
) -> TestVerifierChannel<'a> {
    let mut challenger = test_challenger();
    challenger.observe(*commitment);
    VerifierTranscript::from_data(challenger, data)
}

pub fn sample_indices<R: Rng>(rng: &mut R, upper: usize, count: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(count);
    for _ in 0..count {
        indices.push(rng.random_range(0..upper));
    }
    indices
}

fn test_params() -> PcsParams {
    let fri = FriParams {
        log_blowup: 2,
        fold: FriFold::ARITY_2,
        log_final_degree: 2,
        folding_pow_bits: 1,
    };
    let deep = DeepParams { deep_pow_bits: 1 };
    PcsParams {
        deep,
        fri,
        num_queries: 5,
        query_pow_bits: 1,
    }
}

// ============================================================================
// End-to-end tests
// ============================================================================

use crate::verifier::PcsError;

/// Run the full prover+verifier roundtrip for the given trees and params.
/// On success, also checks that the transcript is fully consumed.
fn run_pcs_case(params: &PcsParams, trees: Vec<TestTree>, seed: u64) -> Result<(), PcsError> {
    let rng = &mut SmallRng::seed_from_u64(seed);
    let lmcs = test_lmcs();

    let lde_height = trees[0].leaves().last().map(|m| m.height()).unwrap_or(0);
    let log_lde_height = log2_strict_usize(lde_height);
    let eval_points: [EF; 2] = [rng.sample(StandardUniform), rng.sample(StandardUniform)];

    let commitments: Vec<_> = trees.iter().map(|t| (t.root(), t.widths())).collect();
    let trace_trees: Vec<&_> = trees.iter().collect();

    // Prover: observe all commitments before opening.
    let mut challenger = test_challenger();
    for (c, _) in &commitments {
        challenger.observe(*c);
    }
    let mut prover_channel = ProverTranscript::new(challenger);

    open_with_channel::<F, EF, _, _, _, 2>(
        params,
        &lmcs,
        log_lde_height,
        eval_points,
        &trace_trees,
        &mut prover_channel,
    );
    let transcript = prover_channel.into_data();

    // Verifier: observe commitments in the same order.
    let mut challenger = test_challenger();
    for (c, _) in &commitments {
        challenger.observe(*c);
    }
    let mut verifier_channel = VerifierTranscript::from_data(challenger, &transcript);

    let result = verify_with_channel::<F, EF, _, _, 2>(
        params,
        &lmcs,
        &commitments,
        log_lde_height,
        eval_points,
        &mut verifier_channel,
    );

    if result.is_ok() {
        assert!(
            verifier_channel.is_empty(),
            "transcript should be fully consumed"
        );
    }
    result.map(|_| ())
}

#[test]
fn test_pcs_cases() {
    let lmcs = test_lmcs();
    let params = test_params();

    // Case 1: single matrix, single tree.
    let rng = &mut SmallRng::seed_from_u64(42);
    let matrix = random_lde_matrix(rng, 6, params.fri.log_blowup, 3, F::GENERATOR);
    let tree = lmcs.build_aligned_tree(vec![matrix]);
    run_pcs_case(&params, vec![tree], 100).expect("single-tree roundtrip");

    // Case 2: two separate trees with different column counts.
    let rng = &mut SmallRng::seed_from_u64(24);
    let mat_a = random_lde_matrix(rng, 6, params.fri.log_blowup, 2, F::GENERATOR);
    let mat_b = random_lde_matrix(rng, 6, params.fri.log_blowup, 4, F::GENERATOR);
    let tree_a = lmcs.build_aligned_tree(vec![mat_a]);
    let tree_b = lmcs.build_aligned_tree(vec![mat_b]);
    run_pcs_case(&params, vec![tree_a, tree_b], 200).expect("multi-tree roundtrip");

    // Case 3: mixed heights in one commitment group (LMCS upsampling).
    let rng = &mut SmallRng::seed_from_u64(99);
    let short = random_lde_matrix(rng, 4, params.fri.log_blowup, 2, F::GENERATOR);
    let tall = random_lde_matrix(rng, 6, params.fri.log_blowup, 3, F::GENERATOR);
    let tree = lmcs.build_aligned_tree(vec![short, tall]);
    run_pcs_case(&params, vec![tree], 300).expect("mixed-height roundtrip");

    // Case 4: random (non-low-degree) data — FRI should reject.
    let rng = &mut SmallRng::seed_from_u64(77);
    let reject_params = PcsParams {
        deep: DeepParams { deep_pow_bits: 1 },
        fri: FriParams {
            log_blowup: 1,
            fold: FriFold::ARITY_2,
            log_final_degree: 2,
            folding_pow_bits: 1,
        },
        num_queries: 20,
        query_pow_bits: 1,
    };
    let height = 1 << 8;
    let matrix = RowMajorMatrix::<F>::rand(rng, height, 3);
    let tree = lmcs.build_aligned_tree(vec![matrix]);
    assert!(
        run_pcs_case(&reject_params, vec![tree], 400).is_err(),
        "should reject high-degree polynomial"
    );
}
