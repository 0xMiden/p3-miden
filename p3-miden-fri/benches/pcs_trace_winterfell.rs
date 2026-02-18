//! Traced PCS run aligned with Winterfell parameters (p3-miden-fri).
//!
//! Runs TwoAdicFriPcs open (Goldilocks + Poseidon2, folding=8, blowup=8)
//! with a tracing subscriber that prints hierarchical span timings.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo +nightly bench -p p3-miden-fri --bench pcs_trace_winterfell
//! ```

use std::time::Instant;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{ExtensionMmcs, Pcs as PcsTrait};
use p3_dft::Radix2DitParallel;
use p3_matrix::Matrix;
use p3_miden_dev_utils::configs::goldilocks_poseidon2 as gl;
use p3_miden_dev_utils::{BenchScenario, GoldilocksPoseidon2, generate_matrices_from_specs};
use p3_miden_fri::{FriParameters, TwoAdicFriPcs};
mod bench_config;
use bench_config::{env_log_trace_heights, env_usize, log2_pow2};
use tracing::{info_span};
use tracing_forest::ForestLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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

fn main() {
    // Initialize tracing subscriber.
    // Use RUST_LOG to control verbosity. Set P3_LOG_FORMAT=forest for tree output.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let registry = tracing_subscriber::registry::Registry::default().with(filter);

    if std::env::var("P3_LOG_FORMAT").as_deref() == Ok("forest") {
        registry.with(ForestLayer::default()).init();
    } else {
        let format = tracing_subscriber::fmt::layer()
            .with_level(false)
            .with_target(false)
            .with_thread_names(false)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .with_ansi(false)
            .compact();
        registry.with(format).init();
    }

    type F = gl::F;
    type EF = gl::EF;
    type BaseMmcs = gl::BaseMmcs;
    type ChallengeMmcs = ExtensionMmcs<F, EF, BaseMmcs>;
    type Dft = Radix2DitParallel<F>;
    type Pcs = TwoAdicFriPcs<F, Dft, BaseMmcs, ChallengeMmcs>;
    type Challenger = gl::Challenger;

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

    let pcs = Pcs::new(dft, base_mmcs, fri_params);

    for &log_trace_height in env_log_trace_heights().iter() {
        let lde_size = 1usize << (log_trace_height + log_blowup);
        eprintln!("\n{}", "=".repeat(60));
        eprintln!(
            "=== Goldilocks p3-miden-fri  log_trace_height={log_trace_height}  (lde={lde_size}) fold={folding} ==="
        );
        eprintln!("{}\n", "=".repeat(60));

        // Generate base matrices aligned with Winterfell widths.
        let matrix_groups = generate_matrices_from_specs(WINTERFELL_SPECS, log_trace_height);

        // Commit each group separately to match Winterfell's 3 roots.
        let mut commits_and_data = Vec::with_capacity(matrix_groups.len());
        for (group_idx, matrices) in matrix_groups.iter().enumerate() {
            let total_cols: usize = matrices.iter().map(|m| m.width()).sum();
            let (commitment, prover_data) = match group_idx {
                0 => info_span!(
                    "commit_main",
                    group_idx,
                    num_matrices = matrices.len(),
                    total_cols
                )
                .in_scope(|| {
                    let domains_and_evals = matrices.iter().map(|m| {
                        let domain =
                            <Pcs as PcsTrait<EF, Challenger>>::natural_domain_for_degree(
                                &pcs,
                                m.height(),
                            );
                        (domain, m.clone())
                    });
                    <Pcs as PcsTrait<EF, Challenger>>::commit(&pcs, domains_and_evals)
                }),
                1 => info_span!(
                    "commit_aux",
                    group_idx,
                    num_matrices = matrices.len(),
                    total_cols
                )
                .in_scope(|| {
                    let domains_and_evals = matrices.iter().map(|m| {
                        let domain =
                            <Pcs as PcsTrait<EF, Challenger>>::natural_domain_for_degree(
                                &pcs,
                                m.height(),
                            );
                        (domain, m.clone())
                    });
                    <Pcs as PcsTrait<EF, Challenger>>::commit(&pcs, domains_and_evals)
                }),
                2 => info_span!(
                    "commit_quotient",
                    group_idx,
                    num_matrices = matrices.len(),
                    total_cols
                )
                .in_scope(|| {
                    let domains_and_evals = matrices.iter().map(|m| {
                        let domain =
                            <Pcs as PcsTrait<EF, Challenger>>::natural_domain_for_degree(
                                &pcs,
                                m.height(),
                            );
                        (domain, m.clone())
                    });
                    <Pcs as PcsTrait<EF, Challenger>>::commit(&pcs, domains_and_evals)
                }),
                _ => info_span!(
                    "commit_group",
                    group_idx,
                    num_matrices = matrices.len(),
                    total_cols
                )
                .in_scope(|| {
                    let domains_and_evals = matrices.iter().map(|m| {
                        let domain =
                            <Pcs as PcsTrait<EF, Challenger>>::natural_domain_for_degree(
                                &pcs,
                                m.height(),
                            );
                        (domain, m.clone())
                    });
                    <Pcs as PcsTrait<EF, Challenger>>::commit(&pcs, domains_and_evals)
                }),
            };

            commits_and_data.push((commitment, prover_data, matrices.len()));
        }

        // Sample OOD points from challenger after observing commitments.
        let mut challenger = gl::test_challenger();
        for (commitment, _, _) in &commits_and_data {
            challenger.observe(*commitment);
        }
        let z1: EF = challenger.sample_algebra_element();
        let z2: EF = challenger.sample_algebra_element();

        // Prepare opening points for each matrix in each commitment.
        let commitment_data_with_opening_points: Vec<_> = commits_and_data
            .iter()
            .map(|(_, prover_data, num_matrices)| {
                let points = vec![vec![z1, z2]; *num_matrices];
                (prover_data, points)
            })
            .collect();

        let start = Instant::now();
        let _open_span = info_span!("opening_total", lde_size).entered();
        let (_opened, _proof) =
            <Pcs as PcsTrait<EF, Challenger>>::open(
                &pcs,
                commitment_data_with_opening_points,
                &mut challenger,
            );
        drop(_open_span);
        let elapsed = start.elapsed();

        eprintln!(">>> Total open: {elapsed:.3?}\n");
    }
}
