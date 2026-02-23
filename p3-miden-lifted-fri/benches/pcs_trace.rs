//! Traced PCS run for profiling with `tracing-subscriber`.
//!
//! Runs the lifted PCS open (Goldilocks + Poseidon2, arity-4) at log heights 16, 18, 20
//! with a tracing subscriber that prints hierarchical span timings.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-miden-lifted-fri --bench pcs_trace --features parallel
//! ```

use std::time::Instant;

use p3_challenger::{CanObserve, FieldChallenger};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::goldilocks_poseidon2 as gl;
use p3_miden_dev_utils::{LOG_HEIGHTS, RELATIVE_SPECS, generate_matrices_from_specs};
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lifted_fri::{PcsParams, prover as lifted_prover};
use p3_miden_lmcs::{Lmcs, LmcsConfig, LmcsTree};
use p3_miden_transcript::ProverTranscript;
use p3_util::log2_strict_usize;
use tracing_subscriber::EnvFilter;

type F = gl::F;
type EF = gl::EF;
type GoldilocksLmcs =
    LmcsConfig<gl::P, gl::P, gl::Sponge, gl::Compress, { gl::WIDTH }, { gl::DIGEST }>;

fn main() {
    // Initialize tracing subscriber.
    // Use RUST_LOG to control verbosity, e.g. RUST_LOG=debug for debug_span! events.
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("debug")),
        )
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
        .init();

    let dft = Radix2DitParallel::<F>::default();
    let shift = F::GENERATOR;

    let params = PcsParams {
        deep: DeepParams { deep_pow_bits: 0 },
        fri: FriParams {
            log_blowup: 2,
            fold: FriFold::ARITY_4,
            log_final_degree: 8,
            folding_pow_bits: 0,
        },
        num_queries: 30,
        query_pow_bits: 0,
    };

    for &log_lde_height in LOG_HEIGHTS {
        let size = 1usize << log_lde_height;
        eprintln!("\n{}", "=".repeat(60));
        eprintln!("=== Goldilocks lifted/arity4  log_height={log_lde_height}  (n={size}) ===");
        eprintln!("{}\n", "=".repeat(60));

        let matrix_groups: Vec<Vec<RowMajorMatrix<F>>> =
            generate_matrices_from_specs(RELATIVE_SPECS, log_lde_height);

        let lmcs = {
            let (_, sponge, compress) = gl::test_components();
            GoldilocksLmcs::new(sponge, compress)
        };

        // Compute LDE matrices and build LMCS tree
        let mut all_lde_matrices: Vec<RowMajorMatrix<F>> = matrix_groups
            .iter()
            .flat_map(|matrices| {
                matrices.iter().map(|m| {
                    let lde = dft.coset_lde_batch(m.clone(), 2, shift);
                    lde.bit_reverse_rows().to_row_major_matrix()
                })
            })
            .collect();
        all_lde_matrices.sort_by_key(|m| m.height());

        let tree = lmcs.build_aligned_tree(all_lde_matrices);
        let commitment = tree.root();
        let log_lde_height = log2_strict_usize(tree.height());

        let mut challenger = gl::test_challenger();
        challenger.observe(commitment);
        let z1: EF = challenger.sample_algebra_element();
        let z2: EF = challenger.sample_algebra_element();
        let mut channel = ProverTranscript::new(challenger);

        let trace_trees: &[&_] = &[&tree];

        let start = Instant::now();
        lifted_prover::open_with_channel::<F, EF, _, _, _, 2>(
            &params,
            &lmcs,
            log_lde_height,
            [z1, z2],
            trace_trees,
            &mut channel,
        );
        let elapsed = start.elapsed();

        eprintln!(">>> Total open_with_channel: {elapsed:.3?}\n");
    }
}
