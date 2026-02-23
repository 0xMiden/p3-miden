//! FRI fold comparison: workspace vs lifted (arity-2 only).
//!
//! Compares the workspace `TwoAdicFriFolding` implementation against the lifted
//! `FriFold` implementation for arity-2 folding. This provides an apples-to-apples
//! comparison of the two approaches. Field-only (no hash functions involved).
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench fri_comparison
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench fri_comparison --features parallel
//!
//! # Filter by field
//! cargo bench --bench fri_comparison -- babybear
//! cargo bench --bench fri_comparison -- goldilocks
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::{BENCH_SEED, BenchScenario, LOG_HEIGHTS, PARALLEL_STR, criterion_config};
use p3_miden_fri::{FriFoldingStrategy, TwoAdicFriFolding};
use p3_miden_lifted_fri::fri::FriFold;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Benchmark FRI fold comparison for a specific field scenario.
fn bench_scenario<S: BenchScenario>(c: &mut Criterion)
where
    StandardUniform: Distribution<S::F> + Distribution<S::EF>,
{
    for &log_height in LOG_HEIGHTS {
        let n = 1usize << log_height;
        let group_name = format!("FRI_Fold_Arity2/{}/{}/{}", n, S::FIELD_NAME, PARALLEL_STR);
        let mut group = c.benchmark_group(&group_name);
        group.throughput(Throughput::Elements(n as u64));

        // Setup (shared between implementations)
        let mut rng = SmallRng::seed_from_u64(BENCH_SEED);
        let beta: S::EF = rng.sample(StandardUniform);

        // Generate matrix with width=2 (arity-2)
        let values: Vec<S::EF> = (0..n * 2).map(|_| rng.sample(StandardUniform)).collect();
        let mat = RowMajorMatrix::new(values, 2);

        // Workspace benchmark - TwoAdicFriFolding computes domain points internally
        let workspace_folding: TwoAdicFriFolding<(), ()> = TwoAdicFriFolding::default();
        group.bench_function(BenchmarkId::from_parameter("workspace"), |b| {
            b.iter(|| workspace_folding.fold_matrix(black_box(beta), black_box(mat.clone())));
        });

        // Lifted benchmark - FriFold requires precomputed s_inv values
        let s_invs: Vec<S::F> = (0..n).map(|_| rng.sample(StandardUniform)).collect();
        let lifted_fold = FriFold::ARITY_2;
        group.bench_function(BenchmarkId::from_parameter("lifted"), |b| {
            b.iter(|| {
                lifted_fold.fold_matrix(
                    black_box(mat.as_view()),
                    black_box(&s_invs),
                    black_box(beta),
                )
            });
        });

        group.finish();
    }
}

fn bench_fri_comparison(c: &mut Criterion) {
    use p3_miden_dev_utils::{BabyBearPoseidon2, GoldilocksPoseidon2};

    // Field-only benchmarks - use Poseidon2 scenarios for field types
    bench_scenario::<BabyBearPoseidon2>(c);
    bench_scenario::<GoldilocksPoseidon2>(c);
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_fri_comparison
}
criterion_main!(benches);
