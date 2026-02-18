//! Criterion benchmarks for PCS open aligned with Winterfell parameters (lifted FRI).

use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_challenger::{CanObserve, FieldChallenger};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::bitrev::BitReversibleMatrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::goldilocks_poseidon2 as gl;
use p3_miden_dev_utils::generate_matrices_from_specs;
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lifted_fri::{PcsParams, prover as lifted_prover};
use p3_miden_lmcs::{Lmcs, LmcsConfig, LmcsTree};
use p3_miden_transcript::ProverTranscript;
use p3_util::log2_strict_usize;
mod bench_config;
use bench_config::{env_log_trace_heights, env_usize};

type F = gl::F;
type EF = gl::EF;
type GoldilocksLmcs =
    LmcsConfig<gl::P, gl::P, gl::Sponge, gl::Compress, { gl::WIDTH }, { gl::DIGEST }>;
type Tree = <GoldilocksLmcs as Lmcs>::Tree<RowMajorMatrix<F>>;

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
fn fri_fold_from_usize(fold: usize) -> FriFold {
    match fold {
        2 => FriFold::ARITY_2,
        4 => FriFold::ARITY_4,
        8 => FriFold::ARITY_8,
        _ => panic!("P3_FOLDING must be one of: 2, 4, 8"),
    }
}

/// Matrix groups aligned with Winterfell mock: 72 main, 8 aux, 1 quotient.
const WINTERFELL_SPECS: &[&[(usize, usize)]] = &[
    &[(0, 72)], // main trace columns
    &[(0, 8)],  // auxiliary columns
    &[(0, 1)],  // quotient / composition columns
];

struct OpenSetup {
    trees: Vec<Tree>,
    log_lde_height: usize,
}

fn pcs_open_setup(lmcs: &GoldilocksLmcs, log_trace_height: usize, log_blowup: usize) -> OpenSetup {
    let dft = Radix2DitParallel::<F>::default();
    let shift = F::GENERATOR;

    // Generate base matrices aligned with Winterfell widths.
    let matrix_groups: Vec<Vec<RowMajorMatrix<F>>> =
        generate_matrices_from_specs(WINTERFELL_SPECS, log_trace_height);

    // Compute LDE matrices and build one LMCS tree per commitment group.
    let trees: Vec<Tree> = matrix_groups
        .iter()
        .map(|matrices| {
            let mut lde_matrices: Vec<RowMajorMatrix<F>> = matrices
                .iter()
                .map(|m| {
                    let lde = dft.coset_lde_batch(m.clone(), log_blowup, shift);
                    lde.bit_reverse_rows().to_row_major_matrix()
                })
                .collect();
            lde_matrices.sort_by_key(|m| m.height());
            lmcs.build_aligned_tree(lde_matrices)
        })
        .collect();

    let first_height = trees
        .first()
        .expect("at least one trace tree required")
        .height();
    let log_lde_height = log2_strict_usize(first_height);

    OpenSetup {
        trees,
        log_lde_height,
    }
}

fn pcs_open_only(params: &PcsParams, lmcs: &GoldilocksLmcs, setup: &OpenSetup) {
    let mut challenger = gl::test_challenger();
    for tree in &setup.trees {
        challenger.observe(tree.root());
    }
    let z1: EF = challenger.sample_algebra_element();
    let z2: EF = challenger.sample_algebra_element();
    let mut channel = ProverTranscript::new(challenger);

    let trace_trees: Vec<&_> = setup.trees.iter().collect();
    lifted_prover::open_with_channel::<F, EF, _, _, _, 2>(
        params,
        lmcs,
        setup.log_lde_height,
        [z1, z2],
        &trace_trees,
        &mut channel,
    );
}

fn pcs_criterion(c: &mut Criterion) {
    let mut group = c.benchmark_group("p3_miden_lifted_fri_pcs");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let log_blowup = env_usize("P3_LOG_BLOWUP", LOG_BLOWUP);
    let num_queries = env_usize("P3_NUM_QUERIES", NUM_QUERIES);
    let log_final_degree = env_usize("P3_LOG_FINAL_DEGREE", LOG_FINAL_DEGREE);
    let query_pow_bits = env_usize("P3_QUERY_POW_BITS", QUERY_POW_BITS);
    let folding = env_usize("P3_FOLDING", 8);

    let params = PcsParams {
        deep: DeepParams { deep_pow_bits: 0 },
        fri: FriParams {
            log_blowup,
            fold: fri_fold_from_usize(folding),
            log_final_degree,
            folding_pow_bits: 0,
        },
        num_queries,
        query_pow_bits,
    };

    let lmcs = {
        let (_, sponge, compress) = gl::test_components();
        GoldilocksLmcs::new(sponge, compress)
    };

    for &log_trace_height in env_log_trace_heights().iter() {
        let lde_size = 1usize << (log_trace_height + log_blowup);
        let setup = pcs_open_setup(&lmcs, log_trace_height, log_blowup);
        let lmcs = lmcs.clone();
        group.bench_function(BenchmarkId::from_parameter(lde_size), move |bench| {
            bench.iter(|| pcs_open_only(&params, &lmcs, &setup));
        });
    }

    group.finish();
}

criterion_group!(pcs_group, pcs_criterion);
criterion_main!(pcs_group);
