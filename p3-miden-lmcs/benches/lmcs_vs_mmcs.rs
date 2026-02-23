//! LMCS vs MMCS comparison benchmarks.
//!
//! Compares the lifted LMCS implementation against the workspace MerkleTreeMmcs
//! using identical hash configurations. Runs benchmarks for:
//! - BabyBear + Poseidon2
//! - Goldilocks + Poseidon2
//! - BabyBear + Keccak
//! - Goldilocks + Keccak
//!
//! Run with:
//! ```bash
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench lmcs_vs_mmcs
//!
//! # With parallelism
//! RUSTFLAGS="-Ctarget-cpu=native" cargo bench --bench lmcs_vs_mmcs --features parallel
//!
//! # Filter by field/hash
//! cargo bench --bench lmcs_vs_mmcs -- babybear/poseidon2
//! cargo bench --bench lmcs_vs_mmcs -- keccak
//! ```

mod utils;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_commit::Mmcs;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::{
    LOG_HEIGHTS, PARALLEL_STR, RELATIVE_SPECS, criterion_config, generate_matrices_from_specs,
    total_elements,
};
use p3_miden_lmcs::{Lmcs, LmcsTree};
use rand::distr::{Distribution, StandardUniform};
use utils::LmcsScenario;

// =============================================================================
// Benchmark implementation
// =============================================================================

/// Run benchmark for a specific scenario.
fn bench_scenario<S: LmcsScenario>(c: &mut Criterion)
where
    StandardUniform: Distribution<S::F>,
{
    for &log_max_height in LOG_HEIGHTS {
        let n_leaves = 1usize << log_max_height;
        let group_name = format!(
            "LMCS_vs_MMCS/{}/{}/{}/{}",
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

        // LMCS (using Lmcs trait API)
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

        // MMCS
        let mmcs = S::mmcs();
        group.bench_with_input(
            BenchmarkId::from_parameter("mmcs"),
            &matrix_groups,
            |b, groups| {
                b.iter(|| {
                    for matrices in groups {
                        black_box(mmcs.commit(matrices.clone()));
                    }
                });
            },
        );

        group.finish();
    }
}

fn bench_lmcs_vs_mmcs(c: &mut Criterion) {
    use p3_miden_dev_utils::{
        BabyBearKeccak, BabyBearPoseidon2, GoldilocksKeccak, GoldilocksPoseidon2,
    };

    // Poseidon2 scenarios
    bench_scenario::<BabyBearPoseidon2>(c);
    bench_scenario::<GoldilocksPoseidon2>(c);

    // Keccak scenarios
    bench_scenario::<BabyBearKeccak>(c);
    bench_scenario::<GoldilocksKeccak>(c);
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_lmcs_vs_mmcs
}
criterion_main!(benches);
