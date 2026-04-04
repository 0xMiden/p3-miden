//! Lifted STARK end-to-end benchmark on Keccak with three traces of different
//! heights (2^15, 2^18, and 2^19). Prints a tracing span tree with per-phase timings.
//!
//! ```bash
//! cargo run -p p3-miden-lifted-examples --release --bin lifted_keccak
//! ```

use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_miden_lifted_stark::{
    air::{AirInstance, AirWitness, log2_strict_u8},
    prove_multi,
    testing::{
        airs::{
            ZeroAuxBuilder,
            keccak::{LiftedKeccakAir, generate_keccak_trace},
        },
        bench_configs::{self, Felt},
        configs::goldilocks_poseidon2 as gl,
        params, stats,
    },
};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use tracing::info_span;

const LOG_BLOWUP: u8 = 1;
const KECCAK_ROWS_PER_HASH: usize = 24;
// Trace S: 2^15 rows, Trace A: 2^18 rows, Trace B: 2^19 rows.
const NUM_HASHES_S: usize = (1 << 15) / KECCAK_ROWS_PER_HASH;
const NUM_HASHES_A: usize = (1 << 18) / KECCAK_ROWS_PER_HASH;
const NUM_HASHES_B: usize = (1 << 19) / KECCAK_ROWS_PER_HASH;

fn main() {
    let stats_handle = stats::init_tracing();
    let bench_iters = stats::bench_iters();

    let config = bench_configs::lifted_config(params::profile_pcs_params(LOG_BLOWUP));
    let air = LiftedKeccakAir;

    let mut rng = SmallRng::seed_from_u64(1);
    let inputs_s: Vec<[u64; 25]> = (0..NUM_HASHES_S).map(|_| rng.random()).collect();
    let inputs_a: Vec<[u64; 25]> = (0..NUM_HASHES_A).map(|_| rng.random()).collect();
    let inputs_b: Vec<[u64; 25]> = (0..NUM_HASHES_B).map(|_| rng.random()).collect();

    let trace_s: RowMajorMatrix<Felt> = info_span!("generate trace S", hashes = NUM_HASHES_S)
        .in_scope(|| generate_keccak_trace(inputs_s));
    let trace_a: RowMajorMatrix<Felt> = info_span!("generate trace A", hashes = NUM_HASHES_A)
        .in_scope(|| generate_keccak_trace(inputs_a));
    let trace_b: RowMajorMatrix<Felt> = info_span!("generate trace B", hashes = NUM_HASHES_B)
        .in_scope(|| generate_keccak_trace(inputs_b));

    tracing::info!(
        height_s = trace_s.height(),
        height_a = trace_a.height(),
        height_b = trace_b.height(),
        width = trace_a.width(),
        "trace dims"
    );

    let log_s = log2_strict_u8(trace_s.height());
    let log_a = log2_strict_u8(trace_a.height());
    let log_b = log2_strict_u8(trace_b.height());

    for i in 0..=bench_iters {
        if i == 0 {
            tracing::info!("warm-up iteration");
        } else {
            tracing::info!(iteration = i, total = bench_iters, "bench iteration");
        }

        // Ascending height order: trace_s (2^15) then trace_a (2^18) then trace_b (2^19).
        let aux = ZeroAuxBuilder::dummy();
        let instances: Vec<(&LiftedKeccakAir, AirWitness<'_, Felt>, &ZeroAuxBuilder)> = vec![
            (&air, AirWitness::new(&trace_s, &[], &[]), &aux),
            (&air, AirWitness::new(&trace_a, &[], &[]), &aux),
            (&air, AirWitness::new(&trace_b, &[], &[]), &aux),
        ];

        let output = info_span!("prove").in_scope(|| {
            prove_multi(&config, &instances, gl::test_challenger()).expect("proving failed")
        });

        if i == 1 {
            let size = stats::serialized_size(&output.proof);
            println!(
                "proof size: {} ({} field elems, {} commitments)",
                stats::format_bytes(size),
                output.proof.fields().len(),
                output.proof.commitments().len(),
            );
        }

        info_span!("verify").in_scope(|| {
            let verifier_instances: Vec<(&LiftedKeccakAir, AirInstance<'_, Felt>)> = vec![
                (
                    &air,
                    AirInstance {
                        log_trace_height: log_s,
                        public_values: &[],
                        var_len_public_inputs: &[],
                    },
                ),
                (
                    &air,
                    AirInstance {
                        log_trace_height: log_a,
                        public_values: &[],
                        var_len_public_inputs: &[],
                    },
                ),
                (
                    &air,
                    AirInstance {
                        log_trace_height: log_b,
                        public_values: &[],
                        var_len_public_inputs: &[],
                    },
                ),
            ];
            let digest = p3_miden_lifted_stark::verify_multi(
                &config,
                &verifier_instances,
                &output.proof,
                gl::test_challenger(),
            )
            .expect("verification failed");
            assert_eq!(output.digest, digest);
        });

        if i == 0 {
            stats_handle.clear();
        }
    }

    stats_handle.print_summary();
}
