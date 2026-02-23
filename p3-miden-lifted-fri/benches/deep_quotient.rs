//! DEEP quotient benchmarks.
//!
//! Benchmarks the barycentric evaluation used in DEEP quotient construction.
//! Runs benchmarks for BabyBear and Goldilocks fields with Poseidon2.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench deep_quotient
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench deep_quotient --features parallel
//!
//! # Filter by field
//! cargo bench --bench deep_quotient -- babybear
//! cargo bench --bench deep_quotient -- goldilocks
//! ```

mod utils;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_commit::Mmcs;
use p3_field::FieldArray;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::{
    LOG_HEIGHTS, PARALLEL_STR, RELATIVE_SPECS, criterion_config, generate_matrices_from_specs,
    total_elements,
};
use p3_miden_lifted_fri::deep::PointQuotients;
use p3_miden_lifted_fri::utils::bit_reversed_coset_points;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use utils::LmcsScenario;

/// Log blowup factor for LDE.
const LOG_BLOWUP: usize = 3;

// =============================================================================
// Benchmark implementation
// =============================================================================

/// Run benchmark for a specific scenario.
fn bench_scenario<S: LmcsScenario>(c: &mut Criterion)
where
    StandardUniform: Distribution<S::F> + Distribution<S::EF>,
{
    for &log_lde_height in LOG_HEIGHTS {
        let n_leaves = 1usize << log_lde_height;
        let group_name = format!(
            "DEEP_Quotient/{}/{}/{}/{}",
            n_leaves,
            S::FIELD_NAME,
            S::HASH_NAME,
            PARALLEL_STR
        );
        let mut group = c.benchmark_group(&group_name);

        // Generate matrices using canonical specs
        let matrix_groups: Vec<Vec<RowMajorMatrix<S::F>>> =
            generate_matrices_from_specs(RELATIVE_SPECS, log_lde_height);
        group.throughput(Throughput::Elements(total_elements(&matrix_groups)));

        // Setup LMCS and commit
        let lmcs = S::lmcs();

        let committed: Vec<_> = matrix_groups
            .iter()
            .map(|matrices| lmcs.commit(matrices.clone()))
            .collect();
        let prover_data: Vec<_> = committed.iter().map(|(_, pd)| pd).collect();

        // Precompute coset points (LDE domain matches max matrix height)
        let coset_points = bit_reversed_coset_points::<S::F>(log_lde_height);

        // Get matrix references from prover data
        let matrices_refs: Vec<Vec<_>> = prover_data
            .iter()
            .map(|pd| lmcs.get_matrices(*pd))
            .collect();

        // Benchmark: batch_eval_lifted with 1 point
        group.bench_function(BenchmarkId::from_parameter("batch_eval/N1"), |b| {
            let mut rng = SmallRng::seed_from_u64(789);
            b.iter(|| {
                let z: S::EF = rng.sample(StandardUniform);
                let quotient =
                    PointQuotients::<S::F, S::EF, 1>::new(FieldArray([z]), &coset_points);
                black_box(quotient.batch_eval_lifted(&matrices_refs, &coset_points, LOG_BLOWUP))
            });
        });

        // Benchmark: batch_eval_lifted with 2 points
        group.bench_function(BenchmarkId::from_parameter("batch_eval/N2"), |b| {
            let mut rng = SmallRng::seed_from_u64(789);
            b.iter(|| {
                let z1: S::EF = rng.sample(StandardUniform);
                let z2: S::EF = rng.sample(StandardUniform);
                let quotient =
                    PointQuotients::<S::F, S::EF, 2>::new(FieldArray([z1, z2]), &coset_points);
                black_box(quotient.batch_eval_lifted(&matrices_refs, &coset_points, LOG_BLOWUP))
            });
        });

        group.finish();
    }
}

fn bench_deep_quotient(c: &mut Criterion) {
    use p3_miden_dev_utils::{BabyBearPoseidon2, GoldilocksPoseidon2};

    bench_scenario::<BabyBearPoseidon2>(c);
    bench_scenario::<GoldilocksPoseidon2>(c);
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_deep_quotient
}
criterion_main!(benches);
