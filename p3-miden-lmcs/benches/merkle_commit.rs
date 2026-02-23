//! Merkle tree commit benchmarks for LMCS.
//!
//! Benchmarks LMCS commit operations including ExtensionMmcs for FRI.
//! Runs benchmarks for BabyBear and Goldilocks fields with Poseidon2.
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench merkle_commit
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench merkle_commit --features parallel
//!
//! # Filter by field
//! cargo bench --bench merkle_commit -- babybear
//! cargo bench --bench merkle_commit -- goldilocks
//! ```

mod utils;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::extension::FlatMatrixView;
use p3_miden_dev_utils::{
    LOG_HEIGHTS, PARALLEL_STR, RELATIVE_SPECS, criterion_config, generate_matrices_from_specs,
    total_elements,
};
use p3_miden_lmcs::{Lmcs, LmcsTree};
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use utils::LmcsScenario;

// =============================================================================
// Benchmark implementation
// =============================================================================

/// Run benchmark for a specific scenario.
fn bench_scenario<S: LmcsScenario>(c: &mut Criterion)
where
    StandardUniform: Distribution<S::F> + Distribution<S::EF>,
{
    for &log_max_height in LOG_HEIGHTS {
        let n_leaves = 1usize << log_max_height;
        let group_name = format!(
            "MerkleCommit/{}/{}/{}/{}",
            n_leaves,
            S::FIELD_NAME,
            S::HASH_NAME,
            PARALLEL_STR
        );
        let mut group = c.benchmark_group(&group_name);
        group.throughput(Throughput::Elements(total_elements(
            &generate_matrices_from_specs::<S::F>(RELATIVE_SPECS, log_max_height),
        )));

        // Generate matrices using canonical specs
        let matrix_groups: Vec<Vec<RowMajorMatrix<S::F>>> =
            generate_matrices_from_specs(RELATIVE_SPECS, log_max_height);

        // LMCS commit
        {
            let lmcs = S::lmcs();
            group.bench_with_input(
                BenchmarkId::from_parameter("lmcs"),
                &matrix_groups,
                |b, groups| {
                    b.iter(|| {
                        for matrices in groups {
                            let tree = lmcs.build_tree(matrices.clone());
                            black_box(tree.root());
                        }
                    });
                },
            );
        }

        // Extension field matrix with width-2 (simulates FRI arity-2 commit)
        // Uses FlatMatrixView to convert EF matrix to base field view
        {
            let lmcs = S::lmcs();

            let rng = &mut SmallRng::seed_from_u64(p3_miden_dev_utils::BENCH_SEED);
            let ext_matrix = RowMajorMatrix::<S::EF>::rand(rng, n_leaves, 2);

            group.bench_with_input(
                BenchmarkId::from_parameter("ext/arity2"),
                &ext_matrix,
                |b, matrix| {
                    b.iter(|| {
                        let flat = FlatMatrixView::new(matrix.clone());
                        let tree = lmcs.build_tree(vec![flat]);
                        black_box(tree.root())
                    });
                },
            );
        }

        // Extension field matrix with width-4 (simulates FRI arity-4 commit)
        {
            let lmcs = S::lmcs();

            let rng = &mut SmallRng::seed_from_u64(p3_miden_dev_utils::BENCH_SEED);
            let ext_matrix = RowMajorMatrix::<S::EF>::rand(rng, n_leaves, 4);

            group.bench_with_input(
                BenchmarkId::from_parameter("ext/arity4"),
                &ext_matrix,
                |b, matrix| {
                    b.iter(|| {
                        let flat = FlatMatrixView::new(matrix.clone());
                        let tree = lmcs.build_tree(vec![flat]);
                        black_box(tree.root())
                    });
                },
            );
        }

        group.finish();
    }
}

fn bench_merkle_commit(c: &mut Criterion) {
    use p3_miden_dev_utils::{BabyBearPoseidon2, GoldilocksPoseidon2};

    bench_scenario::<BabyBearPoseidon2>(c);
    bench_scenario::<GoldilocksPoseidon2>(c);
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_merkle_commit
}
criterion_main!(benches);
