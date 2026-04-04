//! Lifted STARK benchmark with three different hash AIRs (Poseidon2, Keccak, Blake3)
//! at different heights (2^15, 2^18, 2^19). Exercises the multi-trace architecture with
//! heterogeneous AIRs of different widths.
//!
//! Set `BENCH_ITERS` to control the number of measured iterations (default: 5).
//! The first iteration is a warm-up (tracing tree printed, timing discarded).
//!
//! ```bash
//! RUST_LOG=debug cargo run -p p3-miden-lifted-examples --release --features parallel --bin lifted_3_hashes
//! ```

use p3_field::Field;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_miden_lifted_stark::{
    air::{AirInstance, AirWitness, BaseAir, LiftedAir, LiftedAirBuilder, log2_strict_u8},
    prove_multi,
    testing::{
        airs::{
            ZeroAuxBuilder,
            blake3::{LiftedBlake3Air, generate_blake3_trace},
            keccak::{LiftedKeccakAir, generate_keccak_trace},
            poseidon2::{LiftedPoseidon2Air, generate_poseidon2_trace},
        },
        bench_configs::{self, Felt},
        configs::goldilocks_poseidon2 as gl,
        params, stats,
    },
};
use p3_poseidon2_air::RoundConstants;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use tracing::info_span;

const LOG_BLOWUP: u8 = 1;
const KECCAK_ROWS_PER_HASH: usize = 24;
// Blake3: 2^15 rows (1 row/hash), Keccak: 2^18 rows (24 rows/hash), Poseidon2: 2^19 rows (1 row/hash).
const NUM_BLAKE3_HASHES: usize = 1 << 15;
const NUM_KECCAK_HASHES: usize = (1 << 18) / KECCAK_ROWS_PER_HASH;
const NUM_POSEIDON2_HASHES: usize = 1 << 19;

// ---------------------------------------------------------------------------
// Enum wrapper for heterogeneous AIRs
// ---------------------------------------------------------------------------

enum HashAir {
    Poseidon2(Box<LiftedPoseidon2Air>),
    Keccak(LiftedKeccakAir),
    Blake3(LiftedBlake3Air),
}

impl BaseAir<Felt> for HashAir {
    fn width(&self) -> usize {
        match self {
            HashAir::Poseidon2(a) => BaseAir::<Felt>::width(a.as_ref()),
            HashAir::Keccak(a) => BaseAir::<Felt>::width(a),
            HashAir::Blake3(a) => BaseAir::<Felt>::width(a),
        }
    }
}

impl<EF: Field> LiftedAir<Felt, EF> for HashAir {
    fn num_randomness(&self) -> usize {
        1
    }

    fn aux_width(&self) -> usize {
        1
    }

    fn num_aux_values(&self) -> usize {
        0
    }

    fn num_var_len_public_inputs(&self) -> usize {
        0
    }

    fn eval<AB: LiftedAirBuilder<F = Felt>>(&self, builder: &mut AB) {
        match self {
            HashAir::Poseidon2(a) => LiftedAir::<Felt, EF>::eval(a.as_ref(), builder),
            HashAir::Keccak(a) => LiftedAir::<Felt, EF>::eval(a, builder),
            HashAir::Blake3(a) => LiftedAir::<Felt, EF>::eval(a, builder),
        }
    }
}

fn main() {
    let stats_handle = stats::init_tracing();
    let bench_iters = stats::bench_iters();

    let config = bench_configs::lifted_config(params::profile_pcs_params(LOG_BLOWUP));

    let mut rng = SmallRng::seed_from_u64(1);

    // --- Poseidon2 trace (2^19) ---
    let poseidon2_constants = RoundConstants::from_rng(&mut rng);
    let poseidon2_inputs: Vec<[Felt; 12]> =
        (0..NUM_POSEIDON2_HASHES).map(|_| rng.random()).collect();
    let trace_poseidon2: RowMajorMatrix<Felt> =
        info_span!("generate Poseidon2 trace", hashes = NUM_POSEIDON2_HASHES)
            .in_scope(|| generate_poseidon2_trace(poseidon2_inputs, &poseidon2_constants));

    // --- Keccak trace (2^18) ---
    let keccak_inputs: Vec<[u64; 25]> = (0..NUM_KECCAK_HASHES).map(|_| rng.random()).collect();
    let trace_keccak: RowMajorMatrix<Felt> =
        info_span!("generate Keccak trace", hashes = NUM_KECCAK_HASHES)
            .in_scope(|| generate_keccak_trace(keccak_inputs));

    // --- Blake3 trace (2^15) ---
    let blake3_inputs: Vec<[u32; 24]> = (0..NUM_BLAKE3_HASHES).map(|_| rng.random()).collect();
    let trace_blake3: RowMajorMatrix<Felt> =
        info_span!("generate Blake3 trace", hashes = NUM_BLAKE3_HASHES)
            .in_scope(|| generate_blake3_trace(blake3_inputs));

    tracing::info!(
        poseidon2_height = trace_poseidon2.height(),
        poseidon2_width = trace_poseidon2.width(),
        keccak_height = trace_keccak.height(),
        keccak_width = trace_keccak.width(),
        blake3_height = trace_blake3.height(),
        blake3_width = trace_blake3.width(),
        "trace dims"
    );

    let air_poseidon2 = HashAir::Poseidon2(Box::new(LiftedPoseidon2Air::new(poseidon2_constants)));
    let air_keccak = HashAir::Keccak(LiftedKeccakAir);
    let air_blake3 = HashAir::Blake3(LiftedBlake3Air);

    let log_p = log2_strict_u8(trace_poseidon2.height());
    let log_k = log2_strict_u8(trace_keccak.height());
    let log_b = log2_strict_u8(trace_blake3.height());

    // Run iterations: iteration 0 is warm-up (tracing tree printed, stats discarded).
    for i in 0..=bench_iters {
        if i == 0 {
            tracing::info!("warm-up iteration");
        } else {
            tracing::info!(iteration = i, total = bench_iters, "bench iteration");
        }

        // Ascending height order: blake3 (2^15) < keccak (2^18) < poseidon2 (2^19).
        let aux = ZeroAuxBuilder::dummy();
        let instances: Vec<(&HashAir, AirWitness<'_, Felt>, &ZeroAuxBuilder)> = vec![
            (&air_blake3, AirWitness::new(&trace_blake3, &[], &[]), &aux),
            (&air_keccak, AirWitness::new(&trace_keccak, &[], &[]), &aux),
            (
                &air_poseidon2,
                AirWitness::new(&trace_poseidon2, &[], &[]),
                &aux,
            ),
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
            let verifier_instances: Vec<(&HashAir, AirInstance<'_, Felt>)> = vec![
                (
                    &air_blake3,
                    AirInstance {
                        log_trace_height: log_b,
                        public_values: &[],
                        var_len_public_inputs: &[],
                    },
                ),
                (
                    &air_keccak,
                    AirInstance {
                        log_trace_height: log_k,
                        public_values: &[],
                        var_len_public_inputs: &[],
                    },
                ),
                (
                    &air_poseidon2,
                    AirInstance {
                        log_trace_height: log_p,
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
