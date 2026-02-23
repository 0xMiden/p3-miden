//! End-to-end tests for DEEP quotient prover/verifier agreement.

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lmcs::{Lmcs, LmcsTree};
use p3_util::reverse_bits_len;
use rand::distr::StandardUniform;
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::DeepParams;
use super::prover::DeepPoly;
use super::verifier::DeepOracle;
use crate::tests::{
    EF, F, prover_channel_with_commitment, test_lmcs, verifier_channel_with_commitment,
};

/// End-to-end: prover's `DeepPoly.open()` must match verifier's channel-based openings.
#[test]
fn deep_quotient_end_to_end() {
    let rng = &mut SmallRng::seed_from_u64(42);
    let lmcs = test_lmcs();

    // Parameters
    let log_blowup: usize = 2;
    let log_lde_height = 10;
    let lde_height = 1 << log_lde_height;

    let params = DeepParams { deep_pow_bits: 1 };
    // Two random opening points
    let z1: EF = rng.sample(StandardUniform);
    let z2: EF = rng.sample(StandardUniform);

    // Create matrices of varying heights (ascending order required)
    // specs: (log_scaling, width) where height = n >> log_scaling
    let specs: Vec<(usize, usize)> = vec![(2, 2), (1, 3), (0, 4)]; // heights: n/4, n/2, n
    let matrices: Vec<RowMajorMatrix<F>> = specs
        .iter()
        .map(|&(log_scaling, width)| {
            let height = lde_height >> log_scaling;
            RowMajorMatrix::rand(rng, height, width)
        })
        .collect();

    // Step 1: Commit matrices via LMCS (aligned for trace commitments)
    let tree = lmcs.build_aligned_tree(matrices);
    let commitment = tree.root();
    let widths = tree.widths();

    // Step 3: Prover constructs DeepPoly (handles observe, grind, sample internally)
    let mut prover_channel = prover_channel_with_commitment(&commitment);
    let trace_trees: &[&_] = &[&tree];
    let deep_poly = DeepPoly::from_trees::<crate::tests::BaseLmcs, _, 2, _>(
        &params,
        trace_trees,
        [z1, z2],
        log_blowup,
        &mut prover_channel,
    );
    // Sample tree indices (bit-reversed exponents). Tree stores in bit-reversed order.
    let tree_indices: BTreeSet<usize> = [0, 1, lde_height / 4, lde_height / 2, lde_height - 1]
        .into_iter()
        .map(|exp| reverse_bits_len(exp, log_lde_height))
        .collect();
    tree.prove_batch(tree_indices.iter().copied(), &mut prover_channel);
    let transcript = prover_channel.into_data();

    // Create commitments slice for multi-commitment API (single commitment in this case)
    let commitments = vec![(commitment, widths)];

    // Step 4: Verifier constructs DeepOracle with same transcript state
    let mut verifier_channel = verifier_channel_with_commitment(&transcript, &commitment);
    let (deep_oracle, _evals) = DeepOracle::new(
        &params,
        &[z1, z2],
        commitments,
        log_lde_height,
        &mut verifier_channel,
    )
    .expect("DeepOracle construction should succeed");

    // Step 5: Verify at multiple query tree indices (proofs are read from transcript)
    let verifier_evals = deep_oracle
        .open_batch(&lmcs, &tree_indices, &mut verifier_channel)
        .expect("Merkle verification should pass");

    for &tree_idx in &tree_indices {
        // Prover's deep_evals are in bit-reversed order: deep_evals[tree_idx] = Q(g·ω^{bitrev(tree_idx)})
        let prover_eval = deep_poly.deep_evals[tree_idx];
        let verifier_eval = verifier_evals[&tree_idx];
        assert_eq!(
            prover_eval, verifier_eval,
            "Prover and verifier disagree at tree index {tree_idx}"
        );
    }
}
