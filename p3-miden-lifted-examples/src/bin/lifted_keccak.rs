//! Lifted STARK end-to-end benchmark on Keccak with three traces of different
//! heights (2^15, 2^18, and 2^19). Prints a tracing span tree with per-phase timings.
//!
//! ```bash
//! cargo run -p p3-miden-lifted-examples --release --bin lifted_keccak
//! ```

use p3_baby_bear::BabyBear;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_air::EmptyAuxBuilder;
use p3_miden_lifted_examples::keccak::{LiftedKeccakAir, generate_keccak_trace};
use p3_miden_lifted_examples::stats;
use p3_miden_lifted_prover::{
    AirWitness, DeepParams, FriFold, FriParams, LmcsConfig, PcsParams, ProverTranscript,
    prove_multi,
};
use p3_miden_lifted_verifier::{AirInstance, GenericStarkConfig, VerifierTranscript};
use p3_util::log2_strict_usize;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use tracing::info_span;

// Trace S: 2^15 rows → floor(32768/24) = 1365 hashes.
const NUM_HASHES_S: usize = 1365;
// Trace A: 2^18 rows → floor(262144/24) = 10922 hashes.
const NUM_HASHES_A: usize = 10922;
// Trace B: 2^19 rows → floor(524288/24) = 21845 hashes.
const NUM_HASHES_B: usize = 21845;

type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

const LOG_BLOWUP: usize = 1;
const NUM_QUERIES: usize = 100;
const POW_BITS: usize = 16;

fn main() {
    let stats_handle = stats::init_tracing();
    let bench_iters = stats::bench_iters();

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
    let air = LiftedKeccakAir;

    let mut rng = SmallRng::seed_from_u64(1);
    let inputs_s: Vec<[u64; 25]> = (0..NUM_HASHES_S).map(|_| rng.random()).collect();
    let inputs_a: Vec<[u64; 25]> = (0..NUM_HASHES_A).map(|_| rng.random()).collect();
    let inputs_b: Vec<[u64; 25]> = (0..NUM_HASHES_B).map(|_| rng.random()).collect();

    let trace_s: RowMajorMatrix<Val> = info_span!("generate trace S", hashes = NUM_HASHES_S)
        .in_scope(|| generate_keccak_trace(inputs_s));
    let trace_a: RowMajorMatrix<Val> = info_span!("generate trace A", hashes = NUM_HASHES_A)
        .in_scope(|| generate_keccak_trace(inputs_a));
    let trace_b: RowMajorMatrix<Val> = info_span!("generate trace B", hashes = NUM_HASHES_B)
        .in_scope(|| generate_keccak_trace(inputs_b));

    tracing::info!(
        height_s = trace_s.height(),
        height_a = trace_a.height(),
        height_b = trace_b.height(),
        width = trace_a.width(),
        "trace dims"
    );

    let log_s = log2_strict_usize(trace_s.height());
    let log_a = log2_strict_usize(trace_a.height());
    let log_b = log2_strict_usize(trace_b.height());

    for i in 0..=bench_iters {
        if i == 0 {
            tracing::info!("warm-up iteration");
        } else {
            tracing::info!(iteration = i, total = bench_iters, "bench iteration");
        }

        // Ascending height order: trace_s (2^15) then trace_a (2^18) then trace_b (2^19).
        let no_aux = EmptyAuxBuilder;
        let instances: Vec<(&LiftedKeccakAir, AirWitness<'_, Val>, &EmptyAuxBuilder)> = vec![
            (&air, AirWitness::new(&trace_s, &[]), &no_aux),
            (&air, AirWitness::new(&trace_a, &[]), &no_aux),
            (&air, AirWitness::new(&trace_b, &[]), &no_aux),
        ];

        let mut channel = ProverTranscript::new(bb::test_challenger());
        info_span!("prove").in_scope(|| {
            prove_multi(&config, &instances, &mut channel).expect("proving failed");
        });
        let transcript = channel.into_data();

        if i == 1 {
            let size = stats::serialized_size(&transcript);
            println!(
                "proof size: {} ({} field elems, {} commitments)",
                stats::format_bytes(size),
                transcript.fields().len(),
                transcript.commitments().len(),
            );
        }

        info_span!("verify").in_scope(|| {
            let verifier_instances: Vec<(&LiftedKeccakAir, AirInstance<'_, Val>)> = vec![
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
