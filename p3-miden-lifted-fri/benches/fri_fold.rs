//! FRI folding benchmarks for lifted implementation.
//!
//! Benchmarks FRI fold operations at different arities (2, 4, 8).
//! Runs benchmarks for both BabyBear and Goldilocks fields (field-only, no hashing).
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench fri_fold
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench fri_fold --features parallel
//!
//! # Filter by field
//! cargo bench --bench fri_fold -- babybear
//! cargo bench --bench fri_fold -- goldilocks
//! ```

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::{BENCH_SEED, BenchScenario, LOG_HEIGHTS, PARALLEL_STR, criterion_config};
use p3_miden_lifted_fri::fri::FriFold;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

/// Target number of rows after all folding rounds.
const TARGET: usize = 8;

fn bench_lifted_fold<F, EF>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    fold: &FriFold,
    n_elems: usize,
) where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    StandardUniform: Distribution<F> + Distribution<EF>,
{
    let rng = &mut SmallRng::seed_from_u64(BENCH_SEED);
    let arity = fold.arity();

    let n_rows = n_elems / arity;
    let s_invs: Vec<F> = rng.sample_iter(StandardUniform).take(n_rows).collect();

    let values: Vec<EF> = rng.sample_iter(StandardUniform).take(n_elems).collect();
    let input = RowMajorMatrix::new(values, arity);

    group.bench_with_input(
        BenchmarkId::from_parameter(format!("arity{}", arity)),
        &n_elems,
        |b, &_n| {
            b.iter(|| {
                let mut current = input.clone();

                while current.height() > TARGET {
                    let rows = current.height();
                    let beta: EF = rng.sample(StandardUniform);
                    let evals = fold.fold_matrix(
                        black_box(current.as_view()),
                        black_box(&s_invs[..rows]),
                        black_box(beta),
                    );
                    current = RowMajorMatrix::new(evals, arity);
                }
                black_box(current)
            });
        },
    );
}

/// Run benchmark for a specific scenario (field-only, no hash).
fn bench_scenario<S: BenchScenario>(c: &mut Criterion)
where
    StandardUniform: Distribution<S::F> + Distribution<S::EF>,
{
    for &log_height in LOG_HEIGHTS {
        let n_elems = 1usize << log_height;
        let group_name = format!("FRI_Fold/{}/{}/{}", n_elems, S::FIELD_NAME, PARALLEL_STR);
        let mut group = c.benchmark_group(&group_name);
        group.throughput(Throughput::Elements(n_elems as u64));

        bench_lifted_fold::<S::F, S::EF>(&mut group, &FriFold::ARITY_2, n_elems);
        bench_lifted_fold::<S::F, S::EF>(&mut group, &FriFold::ARITY_4, n_elems);
        bench_lifted_fold::<S::F, S::EF>(&mut group, &FriFold::ARITY_8, n_elems);

        group.finish();
    }
}

fn bench_fri_fold(c: &mut Criterion) {
    use p3_miden_dev_utils::{BabyBearPoseidon2, GoldilocksPoseidon2};

    // Field-only benchmarks - use Poseidon2 scenarios for field types
    bench_scenario::<BabyBearPoseidon2>(c);
    bench_scenario::<GoldilocksPoseidon2>(c);
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_fri_fold
}
criterion_main!(benches);
