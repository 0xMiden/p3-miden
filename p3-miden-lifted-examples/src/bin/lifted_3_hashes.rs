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

use alloc::vec;
use alloc::vec::Vec;

use p3_air::{BaseAir, BaseAirWithPublicValues};
use p3_baby_bear::BabyBear;
use p3_dft::Radix2DitParallel;
use p3_field::Field;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_air::{AirWithPeriodicColumns, LiftedAir, LiftedAirBuilder};
use p3_miden_lifted_examples::DummyAuxBuilder;
use p3_miden_lifted_examples::blake3::{LiftedBlake3Air, generate_blake3_trace};
use p3_miden_lifted_examples::keccak::{LiftedKeccakAir, generate_keccak_trace};
use p3_miden_lifted_examples::poseidon2::{LiftedPoseidon2Air, generate_poseidon2_trace};
use p3_miden_lifted_examples::stats;
use p3_miden_lifted_examples::stats::StatsLayer;
use p3_miden_lifted_prover::{
    AirWitness, DeepParams, FriFold, FriParams, LmcsConfig, PcsParams, ProverTranscript,
    prove_multi,
};
use p3_miden_lifted_verifier::{AirInstance, GenericStarkConfig, VerifierTranscript};
use p3_poseidon2_air::RoundConstants;
use p3_util::log2_strict_usize;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::info_span;
use tracing_forest::ForestLayer;
use tracing_subscriber::Layer as _;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

extern crate alloc;

// Blake3: 2^15 rows, 1 row/hash → 32768 hashes (widest, shortest).
const NUM_BLAKE3_HASHES: usize = 32768;
// Keccak: 2^18 rows, 24 rows/hash → floor(262144/24) = 10922 hashes.
const NUM_KECCAK_HASHES: usize = 10922;
// Poseidon2: 2^19 rows, 1 row/hash → 524288 hashes (narrowest, tallest).
const NUM_POSEIDON2_HASHES: usize = 524288;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

const LOG_BLOWUP: usize = 1;
const NUM_QUERIES: usize = 100;
const POW_BITS: usize = 16;

// ---------------------------------------------------------------------------
// Enum wrapper for heterogeneous AIRs
// ---------------------------------------------------------------------------

enum HashAir {
    Poseidon2(Box<LiftedPoseidon2Air>),
    Keccak(LiftedKeccakAir),
    Blake3(LiftedBlake3Air),
}

impl BaseAir<Val> for HashAir {
    fn width(&self) -> usize {
        match self {
            HashAir::Poseidon2(a) => BaseAir::<Val>::width(a.as_ref()),
            HashAir::Keccak(a) => BaseAir::<Val>::width(a),
            HashAir::Blake3(a) => BaseAir::<Val>::width(a),
        }
    }
}

impl BaseAirWithPublicValues<Val> for HashAir {}

impl AirWithPeriodicColumns<Val> for HashAir {
    fn periodic_columns(&self) -> &[Vec<Val>] {
        match self {
            HashAir::Poseidon2(a) => AirWithPeriodicColumns::<Val>::periodic_columns(a.as_ref()),
            HashAir::Keccak(a) => AirWithPeriodicColumns::<Val>::periodic_columns(a),
            HashAir::Blake3(a) => AirWithPeriodicColumns::<Val>::periodic_columns(a),
        }
    }
}

impl<EF: Field> LiftedAir<Val, EF> for HashAir {
    fn num_randomness(&self) -> usize {
        1
    }

    fn aux_width(&self) -> usize {
        1
    }

    fn num_aux_values(&self) -> usize {
        0
    }

    fn eval<AB: LiftedAirBuilder<F = Val>>(&self, builder: &mut AB) {
        match self {
            HashAir::Poseidon2(a) => LiftedAir::<Val, EF>::eval(a.as_ref(), builder),
            HashAir::Keccak(a) => LiftedAir::<Val, EF>::eval(a, builder),
            HashAir::Blake3(a) => LiftedAir::<Val, EF>::eval(a, builder),
        }
    }
}

fn main() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(tracing_forest::util::LevelFilter::DEBUG.into())
        .from_env_lossy();

    let stats = StatsLayer::new();
    let stats_handle = stats.handle();

    // Apply env filter only to ForestLayer so StatsLayer always sees all spans.
    Registry::default()
        .with(ForestLayer::default().with_filter(env_filter))
        .with(stats)
        .init();

    let bench_iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);

    type Lmcs = LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;
    type Dft = Radix2DitParallel<Val>;
    type Config = GenericStarkConfig<Val, Challenge, Lmcs, Dft, bb::Challenger>;

    let pcs = PcsParams {
        fri: FriParams {
            log_blowup: LOG_BLOWUP,
            fold: FriFold::ARITY_2,
            log_final_degree: 0,
            folding_pow_bits: POW_BITS,
        },
        deep: DeepParams { deep_pow_bits: 0 },
        num_queries: NUM_QUERIES,
        query_pow_bits: 0,
    };

    let (_, sponge, compress) = bb::test_components();
    let lmcs: Lmcs = LmcsConfig::new(sponge, compress);
    let dft = Dft::default();
    let config = Config::new(pcs, lmcs, dft, bb::test_challenger());

    let mut rng = SmallRng::seed_from_u64(1);

    // --- Poseidon2 trace (2^19) ---
    let poseidon2_constants = RoundConstants::from_rng(&mut rng);
    let poseidon2_inputs: Vec<[Val; 16]> =
        (0..NUM_POSEIDON2_HASHES).map(|_| rng.random()).collect();
    let trace_poseidon2: RowMajorMatrix<Val> =
        info_span!("generate Poseidon2 trace", hashes = NUM_POSEIDON2_HASHES)
            .in_scope(|| generate_poseidon2_trace(poseidon2_inputs, &poseidon2_constants));

    // --- Keccak trace (2^18) ---
    let keccak_inputs: Vec<[u64; 25]> = (0..NUM_KECCAK_HASHES).map(|_| rng.random()).collect();
    let trace_keccak: RowMajorMatrix<Val> =
        info_span!("generate Keccak trace", hashes = NUM_KECCAK_HASHES)
            .in_scope(|| generate_keccak_trace(keccak_inputs));

    // --- Blake3 trace (2^15) ---
    let blake3_inputs: Vec<[u32; 24]> = (0..NUM_BLAKE3_HASHES).map(|_| rng.random()).collect();
    let trace_blake3: RowMajorMatrix<Val> =
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

    let log_p = log2_strict_usize(trace_poseidon2.height());
    let log_k = log2_strict_usize(trace_keccak.height());
    let log_b = log2_strict_usize(trace_blake3.height());

    // Run iterations: iteration 0 is warm-up (tracing tree printed, stats discarded).
    for i in 0..=bench_iters {
        if i == 0 {
            tracing::info!("warm-up iteration");
        } else {
            tracing::info!(iteration = i, total = bench_iters, "bench iteration");
        }

        // Ascending height order: blake3 (2^15) < keccak (2^18) < poseidon2 (2^19).
        let dummy_aux = DummyAuxBuilder;
        let instances: Vec<(&HashAir, AirWitness<'_, Val>, &DummyAuxBuilder)> = vec![
            (
                &air_blake3,
                AirWitness::new(&trace_blake3, &[], &[]),
                &dummy_aux,
            ),
            (
                &air_keccak,
                AirWitness::new(&trace_keccak, &[], &[]),
                &dummy_aux,
            ),
            (
                &air_poseidon2,
                AirWitness::new(&trace_poseidon2, &[], &[]),
                &dummy_aux,
            ),
        ];

        let mut channel = ProverTranscript::new(bb::test_challenger());
        info_span!("prove").in_scope(|| {
            prove_multi(&config, &instances, &mut channel).expect("proving failed");
        });
        let transcript = channel.into_data();

        if i == 1 {
            let size = stats::serialized_size(&transcript);
            std::println!(
                "proof size: {} ({} field elems, {} commitments)",
                stats::format_bytes(size),
                transcript.fields().len(),
                transcript.commitments().len(),
            );
        }

        info_span!("verify").in_scope(|| {
            let verifier_instances: Vec<(&HashAir, AirInstance<'_, Val>)> = vec![
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
            let mut verifier_channel =
                VerifierTranscript::from_data(bb::test_challenger(), &transcript);
            p3_miden_lifted_verifier::verify_multi(
                &config,
                &verifier_instances,
                &mut verifier_channel,
            )
            .expect("verification failed");
        });

        if i == 0 {
            stats_handle.clear();
        }
    }

    stats_handle.print_summary();
}
