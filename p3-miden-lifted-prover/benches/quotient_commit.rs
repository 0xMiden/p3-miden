//! Microbenchmark: lifted `commit_quotient`.
//!
//! Commits random quotient evaluations (N*D extension field values) to a Merkle
//! tree. This isolates the decomposition + LDE + commit pipeline from the rest
//! of proving.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench -p p3-miden-lifted-prover --bench quotient_commit
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_dev_utils::criterion_config_long;
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lifted_prover::quotient::commit_quotient;
use p3_miden_lifted_stark::LiftedCoset;
use p3_miden_lmcs::LmcsConfig;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// =============================================================================
// Types
// =============================================================================

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;
type Dft = p3_dft::Radix2DitParallel<Val>;

// Lifted types
type LiftedLmcs = LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;
type LiftedConfig =
    p3_miden_lifted_verifier::GenericStarkConfig<Val, Challenge, LiftedLmcs, Dft, Challenger>;
type Challenger = DuplexChallenger<Val, bb::Perm, { bb::WIDTH }, { bb::RATE }>;

// =============================================================================
// Constants
// =============================================================================

const LOG_BLOWUP: usize = 1;
const D: usize = 2; // constraint degree (KeccakAir)

// =============================================================================
// Setup helpers
// =============================================================================

fn lifted_config() -> LiftedConfig {
    let pcs = p3_miden_lifted_fri::PcsParams {
        fri: FriParams {
            log_blowup: LOG_BLOWUP,
            fold: FriFold::ARITY_2,
            log_final_degree: 0,
            folding_pow_bits: 0,
        },
        deep: DeepParams { deep_pow_bits: 0 },
        num_queries: 1,
        query_pow_bits: 0,
    };
    let (_, sponge, compress) = bb::test_components();
    let lmcs: LiftedLmcs = LmcsConfig::new(sponge, compress);
    LiftedConfig::new(pcs, lmcs, Dft::default(), bb::test_challenger())
}

fn random_quotient_evals(n: usize, d: usize, seed: u64) -> Vec<Challenge> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..n * d).map(|_| rng.random()).collect()
}

// =============================================================================
// Benchmark
// =============================================================================

fn bench_quotient_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("quotient_commit");

    for log_n in [16, 17] {
        let n = 1usize << log_n;
        let b = 1usize << LOG_BLOWUP;
        let label = format!("N=2^{log_n}");

        let config = lifted_config();
        let coset = LiftedCoset::unlifted(log_n, LOG_BLOWUP);

        group.bench_function(BenchmarkId::new("lifted", &label), |bench| {
            bench.iter(|| {
                let mut q_evals = random_quotient_evals(n, D, 42);
                q_evals.reserve(n * b - n * D);
                let committed = commit_quotient(&config, q_evals, &coset);
                black_box(committed)
            });
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = criterion_config_long();
    targets = bench_quotient_commit
}
criterion_main!(benches);
