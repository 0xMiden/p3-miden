//! Criterion benchmarks for PCS open aligned with Winterfell parameters (p3-miden-fri).

use std::sync::Arc;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
use p3_matrix::Matrix;
use p3_miden_dev_utils::configs::goldilocks_poseidon2 as gl;
use p3_miden_dev_utils::{BenchScenario, GoldilocksPoseidon2, generate_matrices_from_specs};
use p3_miden_fri::{FriParameters, TwoAdicFriPcs};
mod bench_config;
use bench_config::{env_log_trace_heights, env_usize, log2_pow2};

// =============================================================================
// Winterfell-aligned constants
// =============================================================================

/// Log blowup factor for FRI (blowup = 8).
const LOG_BLOWUP: usize = 3;
/// Number of FRI queries.
const NUM_QUERIES: usize = 32;
/// Log degree of final polynomial.
const LOG_FINAL_DEGREE: usize = 5;
/// Query PoW bits (Winterfell grinding factor).
const QUERY_POW_BITS: usize = 16;
/// Folding factor log2 (folding = 8).
const LOG_FOLDING_FACTOR: usize = 3;

/// Matrix groups aligned with Winterfell mock: 72 main, 8 aux, 1 quotient.
const WINTERFELL_SPECS: &[&[(usize, usize)]] = &[
    &[(0, 72)], // main trace columns
    &[(0, 8)],  // auxiliary columns
    &[(0, 1)],  // quotient / composition columns
];

type Dft = Radix2DitParallel<gl::F>;
type BaseMmcs = gl::BaseMmcs;
type ChallengeMmcs = ExtensionMmcs<gl::F, gl::EF, BaseMmcs>;
type Pcs = TwoAdicFriPcs<gl::F, Dft, BaseMmcs, ChallengeMmcs>;
type Challenger = gl::Challenger;
type Commitment = <Pcs as PcsTrait<gl::EF, Challenger>>::Commitment;
type ProverData = <Pcs as PcsTrait<gl::EF, Challenger>>::ProverData;

struct OpenSetup {
    commits_and_data: Vec<(Commitment, ProverData, usize)>,
}

fn pcs_open_setup(pcs: &Pcs, log_trace_height: usize) -> OpenSetup {
    // Generate base matrices aligned with Winterfell widths.
    let matrix_groups = generate_matrices_from_specs(WINTERFELL_SPECS, log_trace_height);

    // Commit each group separately to match Winterfell's 3 roots.
    let mut commits_and_data = Vec::with_capacity(matrix_groups.len());
    for matrices in matrix_groups.into_iter() {
        let num_matrices = matrices.len();
        let domains_and_evals = matrices.into_iter().map(|m| {
            let domain = <Pcs as PcsTrait<gl::EF, Challenger>>::natural_domain_for_degree(
                pcs,
                m.height(),
            );
            (domain, m)
        });
        let (commitment, prover_data) =
            <Pcs as PcsTrait<gl::EF, Challenger>>::commit(pcs, domains_and_evals);
        commits_and_data.push((commitment, prover_data, num_matrices));
    }

    OpenSetup { commits_and_data }
}

fn pcs_open_only(pcs: &Pcs, setup: &OpenSetup) {
    // Sample OOD points from challenger after observing commitments.
    let mut challenger = gl::test_challenger();
    for (commitment, _, _) in &setup.commits_and_data {
        challenger.observe(*commitment);
    }
    let z1: gl::EF = challenger.sample_algebra_element();
    let z2: gl::EF = challenger.sample_algebra_element();

    // Prepare opening points for each matrix in each commitment.
    let commitment_data_with_opening_points: Vec<_> = setup
        .commits_and_data
        .iter()
        .map(|(_, prover_data, num_matrices)| {
            let points = vec![vec![z1, z2]; *num_matrices];
            (prover_data, points)
        })
        .collect();

    let _ = <Pcs as PcsTrait<gl::EF, Challenger>>::open(
        pcs,
        commitment_data_with_opening_points,
        &mut challenger,
    );
}

fn pcs_criterion(c: &mut Criterion) {
    let mut group = c.benchmark_group("p3_miden_fri_pcs");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let dft = Dft::default();
    let base_mmcs = GoldilocksPoseidon2::mmcs();
    let challenge_mmcs = ExtensionMmcs::new(base_mmcs.clone());

    let log_blowup = env_usize("P3_LOG_BLOWUP", LOG_BLOWUP);
    let num_queries = env_usize("P3_NUM_QUERIES", NUM_QUERIES);
    let log_final_degree = env_usize("P3_LOG_FINAL_DEGREE", LOG_FINAL_DEGREE);
    let query_pow_bits = env_usize("P3_QUERY_POW_BITS", QUERY_POW_BITS);
    let folding = env_usize("P3_FOLDING", 1usize << LOG_FOLDING_FACTOR);
    let log_folding_factor = log2_pow2(folding, "P3_FOLDING");

    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: log_final_degree,
        num_queries,
        proof_of_work_bits: query_pow_bits,
        mmcs: challenge_mmcs,
        log_folding_factor,
    };

    let pcs = Arc::new(Pcs::new(dft, base_mmcs, fri_params));

    for &log_trace_height in env_log_trace_heights().iter() {
        let lde_size = 1usize << (log_trace_height + log_blowup);
        let setup = pcs_open_setup(pcs.as_ref(), log_trace_height);
        let pcs = pcs.clone();
        group.bench_function(BenchmarkId::from_parameter(lde_size), move |bench| {
            bench.iter(|| pcs_open_only(pcs.as_ref(), &setup));
        });
    }

    group.finish();
}

criterion_group!(pcs_group, pcs_criterion);
criterion_main!(pcs_group);
