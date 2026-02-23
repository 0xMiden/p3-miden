//! Integration tests for FRI protocol commit/verify cycles.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use p3_challenger::CanObserve;
use p3_dft::{Radix2DFTSmallBatch, TwoAdicSubgroupDft};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_transcript::{VerifierChannel, VerifierTranscript};
use p3_util::{log2_strict_usize, reverse_bits_len};
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::prover::FriPolys;
use super::verifier::FriOracle;
use super::*;
use crate::tests::{
    BaseLmcs, Challenger, EF, F, TestTranscriptData, prover_channel, random_lde_matrix,
    sample_indices, test_challenger, test_lmcs, verifier_channel,
};

// ============================================================================
// Integration tests
// ============================================================================

struct FriRoundtripCase {
    name: &'static str,
    log_poly_degree: usize,
    log_blowup: usize,
    log_final_degree: usize,
    fold: FriFold,
    folding_pow_bits: usize,
    num_queries: usize,
}

const FRI_ROUNDTRIP_CASES: &[FriRoundtripCase] = &[
    FriRoundtripCase {
        name: "arity-2",
        log_poly_degree: 10,
        log_blowup: 2,
        log_final_degree: 2,
        fold: FriFold::ARITY_2,
        folding_pow_bits: 1,
        num_queries: 3,
    },
    FriRoundtripCase {
        name: "arity-4",
        log_poly_degree: 10,
        log_blowup: 2,
        log_final_degree: 2,
        fold: FriFold::ARITY_4,
        folding_pow_bits: 1,
        num_queries: 3,
    },
    FriRoundtripCase {
        name: "arity-8",
        log_poly_degree: 10,
        log_blowup: 2,
        log_final_degree: 2,
        fold: FriFold::ARITY_8,
        folding_pow_bits: 1,
        num_queries: 3,
    },
    FriRoundtripCase {
        name: "blowup-0",
        log_poly_degree: 8,
        log_blowup: 0,
        log_final_degree: 3,
        fold: FriFold::ARITY_2,
        folding_pow_bits: 0,
        num_queries: 2,
    },
];

/// Build initial_evals map from tree indices and bit-reversed evaluation array.
///
/// `evals` is in bit-reversed order: `evals[tree_idx]` = f(g·ω^{bitrev(tree_idx)}).
/// `tree_indices` are the bit-reversed tree positions.
/// Returns a map keyed by tree index.
fn build_initial_evals(evals: &[EF], tree_indices: &BTreeSet<usize>) -> BTreeMap<usize, EF> {
    tree_indices
        .iter()
        .map(|&tree_idx| (tree_idx, evals[tree_idx]))
        .collect()
}

fn prove_queries(
    params: &FriParams,
    lmcs: &BaseLmcs,
    evals: Vec<EF>,
    tree_indices: &BTreeSet<usize>,
) -> TestTranscriptData {
    let mut prover_channel = prover_channel();
    let fri_polys = FriPolys::<F, EF, _>::new(params, lmcs, evals, &mut prover_channel);
    fri_polys.prove_queries(params, tree_indices, &mut prover_channel);
    prover_channel.into_data()
}

fn verify_queries(
    params: &FriParams,
    lmcs: &BaseLmcs,
    transcript: &TestTranscriptData,
    lde_size: usize,
    initial_evals: &BTreeMap<usize, EF>,
    challenger: Option<Challenger>,
) -> Result<(), FriError> {
    let mut channel = match challenger {
        Some(challenger) => VerifierTranscript::from_data(challenger, transcript),
        None => verifier_channel(transcript),
    };
    let log_domain_size = log2_strict_usize(lde_size);
    let oracle = FriOracle::new(params, log_domain_size, &mut channel)?;
    let result = oracle.test_low_degree(lmcs, params, initial_evals.clone(), &mut channel);
    if result.is_ok() {
        assert!(channel.is_empty(), "transcript should be fully consumed");
    }
    result
}

fn run_roundtrip_case(case: &FriRoundtripCase, seed: u64) -> Result<(), FriError> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let lmcs = test_lmcs();

    let params = FriParams {
        log_blowup: case.log_blowup,
        fold: case.fold,
        log_final_degree: case.log_final_degree,
        folding_pow_bits: case.folding_pow_bits,
    };

    let evals =
        random_lde_matrix::<F, EF>(&mut rng, case.log_poly_degree, case.log_blowup, 1, F::ONE)
            .values;
    let lde_size = evals.len();
    let log_domain_size = log2_strict_usize(lde_size);
    // Sample exponents and convert to tree indices (bit-reversed)
    let tree_indices: BTreeSet<usize> = sample_indices(&mut rng, lde_size, case.num_queries)
        .into_iter()
        .map(|exp| reverse_bits_len(exp, log_domain_size))
        .collect();
    let initial_evals = build_initial_evals(&evals, &tree_indices);

    let transcript = prove_queries(&params, &lmcs, evals, &tree_indices);
    verify_queries(&params, &lmcs, &transcript, lde_size, &initial_evals, None)
}

/// Table-driven roundtrip cases that must verify successfully.
#[test]
fn test_fri_roundtrip_cases() {
    for (case_idx, case) in FRI_ROUNDTRIP_CASES.iter().enumerate() {
        let seed = 42 + case_idx as u64;
        let result = run_roundtrip_case(case, seed);
        assert!(
            result.is_ok(),
            "case {} failed with {:?}",
            case.name,
            result
        );
    }
}

/// Test that verification fails with wrong initial evaluation.
#[test]
fn test_fri_verify_wrong_eval() {
    let mut rng = SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    let log_poly_degree = 8;
    let log_blowup = 2;
    let log_final_degree = 2;

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        folding_pow_bits: 1,
    };

    let evals = random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();
    let log_domain_size = log2_strict_usize(lde_size);
    let tree_indices: BTreeSet<usize> = sample_indices(&mut rng, lde_size, 2)
        .into_iter()
        .map(|exp| reverse_bits_len(exp, log_domain_size))
        .collect();
    let mut initial_evals = build_initial_evals(&evals, &tree_indices);

    // Tamper with the first evaluation
    let first_idx = *tree_indices.first().unwrap();
    let correct_eval = initial_evals[&first_idx];
    let mut wrong_eval: EF = rng.sample(StandardUniform);
    while wrong_eval == correct_eval {
        wrong_eval = rng.sample(StandardUniform);
    }
    initial_evals.insert(first_idx, wrong_eval);

    let transcript = prove_queries(&params, &lmcs, evals, &tree_indices);
    let result = verify_queries(&params, &lmcs, &transcript, lde_size, &initial_evals, None);

    assert!(
        matches!(result, Err(FriError::EvaluationMismatch { .. })),
        "expected EvaluationMismatch error, got {:?}",
        result
    );
}

/// Test that verification fails with mismatched proof data (wrong betas scenario).
///
/// When the verifier's challenger state differs from the prover's (e.g., due to
/// different commitments being observed), the derived betas will be wrong,
/// causing verification to fail.
#[test]
fn test_fri_verify_wrong_beta() {
    let mut rng = SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    let log_poly_degree = 8;
    let log_blowup = 2;
    let log_final_degree = 2;

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        folding_pow_bits: 0, // No grinding to simplify test
    };

    // Create two independent provers with different evaluations.
    let evals1 =
        random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let evals2 =
        random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals1.len();
    let log_domain_size = log2_strict_usize(lde_size);

    // Prover 1: generate FRI transcript (grinds per-round internally).
    let tree_indices: BTreeSet<usize> = sample_indices(&mut rng, lde_size, 2)
        .into_iter()
        .map(|exp| reverse_bits_len(exp, log_domain_size))
        .collect();
    let initial_evals = build_initial_evals(&evals1, &tree_indices);
    let transcript = prove_queries(&params, &lmcs, evals1, &tree_indices);

    // Prover 2: generate different transcript (different commitments = different betas).
    let mut prover2_channel = prover_channel();
    let _ = FriPolys::<F, EF, _>::new(&params, &lmcs, evals2, &mut prover2_channel);
    let other_commitment = prover2_channel
        .into_data()
        .commitments()
        .first()
        .cloned()
        .expect("prover2 should produce commitments");

    // Verifier: use prover1's transcript but alter challenger state beforehand.
    let mut wrong_challenger = test_challenger();
    wrong_challenger.observe(other_commitment);
    let result = verify_queries(
        &params,
        &lmcs,
        &transcript,
        lde_size,
        &initial_evals,
        Some(wrong_challenger),
    );

    // Should fail because wrong betas produce wrong folding results
    assert!(
        matches!(
            result,
            Err(FriError::EvaluationMismatch { .. }) | Err(FriError::FinalPolyMismatch { .. })
        ),
        "expected EvaluationMismatch or FinalPolyMismatch error, got {:?}",
        result
    );
}

/// Zero-round FRI: when the evaluation domain is at or below the final polynomial degree.
#[test]
fn test_fri_zero_rounds_final_poly_only() {
    let mut rng = SmallRng::seed_from_u64(123);
    let lmcs = test_lmcs();

    let log_poly_degree = 4;
    let log_blowup = 0;
    let log_final_degree = log_poly_degree; // final degree >= domain size => zero rounds

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        folding_pow_bits: 0,
    };

    let evals = random_lde_matrix::<F, EF>(&mut rng, log_poly_degree, log_blowup, 1, F::ONE).values;
    let lde_size = evals.len();
    let log_domain_size = log2_strict_usize(lde_size);
    let tree_indices: BTreeSet<usize> = sample_indices(&mut rng, lde_size, 2)
        .into_iter()
        .map(|exp| reverse_bits_len(exp, log_domain_size))
        .collect();
    let initial_evals = build_initial_evals(&evals, &tree_indices);
    let transcript = prove_queries(&params, &lmcs, evals, &tree_indices);

    let mut channel = verifier_channel(&transcript);
    let fri_transcript: FriTranscript<F, EF, _> =
        FriTranscript::from_verifier_channel(&params, log_domain_size, &mut channel)
            .expect("transcript parsing should succeed");

    assert!(
        fri_transcript.rounds.is_empty(),
        "expected zero folding rounds"
    );
    assert_eq!(
        fri_transcript.final_poly.len(),
        lde_size,
        "final polynomial should match domain size"
    );

    verify_queries(&params, &lmcs, &transcript, lde_size, &initial_evals, None)
        .expect("zero-round FRI should verify");
}

/// Test that the final polynomial is correctly computed by evaluating it
/// at points in the final domain and comparing with folded values.
#[test]
fn test_final_polynomial_correctness() {
    let mut rng = SmallRng::seed_from_u64(123);
    let lmcs = test_lmcs();

    let log_poly_degree = 6;
    let log_blowup = 2;
    let log_final_degree = 3;

    let params = FriParams {
        log_blowup,
        fold: FriFold::ARITY_2,
        log_final_degree,
        folding_pow_bits: 0, // No grinding for this test
    };

    let poly_degree = 1 << log_poly_degree;
    let final_degree = 1 << log_final_degree;
    let rounds = log_poly_degree - log_final_degree;
    let stride = 1 << rounds;

    let g_coeffs: Vec<EF> = (0..final_degree)
        .map(|_| rng.sample(StandardUniform))
        .collect();
    let mut f_coeffs = vec![EF::ZERO; poly_degree];
    for (i, coeff) in g_coeffs.iter().enumerate() {
        f_coeffs[i * stride] = *coeff;
    }

    let coeffs_matrix = RowMajorMatrix::new(f_coeffs, 1);
    let dft = Radix2DFTSmallBatch::<F>::default();
    // DFT output is already in standard order for Radix2DFTSmallBatch.
    let evals_h = dft.coset_dft_algebra_batch(coeffs_matrix, F::ONE);
    let lde = dft.coset_lde_algebra_batch(evals_h, log_blowup, F::ONE);
    let evals = lde.bit_reverse_rows().to_row_major_matrix().values;

    let log_domain_size = log_poly_degree + log_blowup;

    let mut prover_channel = prover_channel();
    let _fri_polys = FriPolys::<F, EF, _>::new(&params, &lmcs, evals.clone(), &mut prover_channel);
    let transcript = prover_channel.into_data();

    let mut v_channel = verifier_channel(&transcript);
    let fri_transcript: FriTranscript<F, EF, _> =
        FriTranscript::from_verifier_channel(&params, log_domain_size, &mut v_channel)
            .expect("transcript parsing should succeed");

    assert_eq!(
        fri_transcript.final_poly.len(),
        g_coeffs.len(),
        "Final polynomial should have {} coefficients",
        g_coeffs.len()
    );
    let mut g_coeffs_rev = g_coeffs;
    g_coeffs_rev.reverse();
    assert_eq!(
        fri_transcript.final_poly, g_coeffs_rev,
        "Final polynomial coefficients should be in descending degree order"
    );
}
