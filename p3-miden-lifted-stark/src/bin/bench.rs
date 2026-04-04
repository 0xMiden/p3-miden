//! Unified STARK prove/verify profiling binary.
//!
//! Proves and verifies a set of AIR instances, printing a configuration summary,
//! proof size, and total time. Trace specifications are passed as positional
//! arguments; all parameters have sensible defaults.
//!
//! By default the output is concise (config header + proof size + total time).
//! Pass `-v` for the full hierarchical tracing tree, or set `RUST_LOG` for
//! fine-grained control.
//!
//! ```bash
//! # Quick run with defaults (blake3:15 keccak:18 poseidon2:19):
//! cargo run --bin bench --features std,parallel --release
//!
//! # Custom traces:
//! cargo run --bin bench --features std,parallel --release -- keccak:15 keccak:18 keccak:19
//!
//! # Full tracing tree:
//! cargo run --bin bench --features std,parallel --release -- -v keccak:15
//!
//! # Multi-iteration with warm-up (reports min/median/mean/max):
//! cargo run --bin bench --features std,parallel --release -- -n 5 keccak:15
//!
//! # Miden-shaped AIR (auto log_blowup=3):
//! cargo run --bin bench --features std,parallel --release -- miden:18:51 miden:19:20
//!
//! # Batch-STARK comparison:
//! cargo run --bin bench --features std,parallel --release -- -m batch keccak:15 keccak:18
//! ```

use std::{fmt, str::FromStr, time::Instant};

use clap::{Parser, ValueEnum};
use p3_air::{Air, AirBuilder, AirLayout, BaseLeaf, SymbolicExpression, WindowAccess};
use p3_batch_stark::{ProverData, StarkInstance, prove_batch, verify_batch};
use p3_blake3_air::{Blake3Air, NUM_BLAKE3_COLS};
use p3_commit::ExtensionMmcs;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::GenericPoseidon2LinearLayersGoldilocks;
use p3_keccak::KeccakF;
use p3_keccak_air::{KeccakAir, NUM_KECCAK_COLS};
use p3_lookup::{
    LookupAir,
    lookup_traits::{Direction, Kind, Lookup},
};
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_merkle_tree::MerkleTreeMmcs;
use p3_miden_lifted_stark::{
    GenericStarkConfig, PcsParams, StarkConfig,
    air::{AirInstance, AirWitness, BaseAir, LiftedAir, LiftedAirBuilder, log2_strict_u8},
    prove_multi,
    testing::{
        airs::{
            ZeroAuxBuilder,
            blake3::{LiftedBlake3Air, generate_blake3_trace},
            keccak::{LiftedKeccakAir, generate_keccak_trace},
            miden::{DummyMidenAir, generate_dummy_trace},
            poseidon2::{
                HALF_FULL_ROUNDS, LiftedPoseidon2Air, NUM_POSEIDON2_COLS, PARTIAL_ROUNDS,
                SBOX_DEGREE, SBOX_REGISTERS, WIDTH, generate_poseidon2_trace,
            },
        },
        bench_configs::Felt,
        configs::{
            QuadFelt, goldilocks_blake3 as blake3, goldilocks_blake3_192 as blake3_192,
            goldilocks_keccak as keccak, goldilocks_poseidon2 as gl,
        },
        params,
    },
    verify_multi,
};
use p3_poseidon2_air::{Poseidon2Air, RoundConstants};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::SymbolicAirBuilder;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use tracing::info_span;
use tracing_subscriber::{Layer, Registry, layer::SubscriberExt, util::SubscriberInitExt};

// ─── Type aliases ────────────────────────────────────────────────────────────

type Gl = p3_goldilocks::Goldilocks;
type GlRoundConstants = RoundConstants<Gl, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

type BatchPoseidon2Air = Poseidon2Air<
    Felt,
    GenericPoseidon2LinearLayersGoldilocks,
    WIDTH,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS,
>;

const KECCAK_ROWS_PER_HASH: usize = 24;
const DEFAULT_MIDEN_WIDTH: usize = 51;
const DEFAULT_MIDEN_AUX_COLS: usize = 8;

// ═══════════════════════════════════════════════════════════════════════════════
// CLI
// ═══════════════════════════════════════════════════════════════════════════════

/// Prove and verify a set of AIR instances with the lifted or batch STARK prover.
///
/// Prints a configuration summary, proof size, and total time.
/// Pass -v for the full hierarchical tracing tree.
///
/// Trace spec format:  `AIR:LOG_HEIGHT[:WIDTH[:AUX_COLS]]`
///
/// Available AIR types (with short aliases):
///
///   keccak (k)     Keccak-f(1600) permutation, 24 rows/hash
///   poseidon2 (p)  Poseidon2 permutation (Goldilocks), 1 row/hash
///   blake3 (b)     Blake3 compression, 1 row/hash
///   miden (m)      Dummy degree-9 constraint (Miden VM shape)
///
/// WIDTH and AUX_COLS only apply to `miden` (defaults: 51, 8).
///
/// Examples:
///
///   bench                                             # default: blake3:15 keccak:18 poseidon2:19
///   bench keccak:15 keccak:18 keccak:19               # 3x Keccak at different heights
///   bench -v keccak:15                                 # full tracing tree
///   bench miden:18:51 miden:19:20                      # two Miden-shaped traces (auto blowup=3)
///   bench -m batch keccak:15 keccak:18 keccak:19       # batch-STARK comparison
///   bench -H keccak keccak:15                          # use Keccak hash for commitments
///   bench -H blake3 keccak:15                          # use BLAKE3 (32B) hash for commitments
///   bench -H blake3-192 keccak:15                      # use BLAKE3-192 (24B) hash for commitments
///   bench --log-blowup 2 --num-queries 50 keccak:18    # override PCS parameters
#[derive(Parser)]
#[command(name = "bench", verbatim_doc_comment)]
struct Cli {
    /// Trace specs (`AIR:LOG_HEIGHT[:WIDTH[:AUX_COLS]]`).
    ///
    /// When omitted, defaults to: blake3:15 keccak:18 poseidon2:19
    #[arg(value_name = "TRACE")]
    traces: Vec<TraceSpec>,

    /// Prover backend: `lifted` (LMCS-based) or `batch` (Plonky3 batch-STARK).
    #[arg(long, short = 'm', value_enum, default_value_t = Mode::Lifted)]
    mode: Mode,

    /// Hash function for the commitment scheme.
    ///
    /// Only applies to lifted mode; batch mode always uses poseidon2.
    #[arg(long, short = 'H', value_enum, default_value_t = HashFn::Poseidon2)]
    hash: HashFn,

    /// Print the full hierarchical tracing tree (default: summary only).
    ///
    /// RUST_LOG overrides this when set.
    #[arg(long, short = 'v')]
    verbose: bool,

    /// RNG seed for reproducible trace generation.
    #[arg(long, short = 's', default_value_t = 1)]
    seed: u64,

    /// Skip proof verification (prover-only profiling).
    #[arg(long)]
    no_verify: bool,

    // ── PCS Parameters ──────────────────────────────────────────────────
    /// Log₂ blowup factor for the LDE domain extension.
    ///
    /// Auto-detected when omitted: 1 for hash-only workloads, 3 when any
    /// `miden` trace is present (degree-9 constraints need more blowup).
    #[arg(long, help_heading = "PCS Parameters")]
    log_blowup: Option<u8>,

    /// Number of FRI query repetitions (higher = more soundness).
    #[arg(long, default_value_t = params::PROFILE_NUM_QUERIES, help_heading = "PCS Parameters")]
    num_queries: usize,

    /// Proof-of-work grinding bits for the DEEP challenge (lifted mode only).
    #[arg(long, default_value_t = params::PROFILE_POW_BITS, help_heading = "PCS Parameters")]
    deep_pow_bits: usize,

    /// Log₂ FRI folding arity (1, 2, or 3 for fold-by-2/4/8).
    #[arg(long, default_value_t = 2, help_heading = "PCS Parameters")]
    log_folding_arity: u8,

    /// Log₂ final polynomial degree bound.
    #[arg(long, default_value_t = 0, help_heading = "PCS Parameters")]
    log_final_degree: u8,

    /// Proof-of-work grinding bits per FRI folding round.
    #[arg(long, default_value_t = 0, help_heading = "PCS Parameters")]
    folding_pow_bits: usize,

    /// Proof-of-work grinding bits before query index sampling.
    #[arg(long, default_value_t = 0, help_heading = "PCS Parameters")]
    query_pow_bits: usize,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Lifted,
    Batch,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum HashFn {
    Poseidon2,
    Keccak,
    Blake3,
    #[value(name = "blake3-192")]
    Blake3_192,
}

impl fmt::Display for HashFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HashFn::Poseidon2 => write!(f, "poseidon2"),
            HashFn::Keccak => write!(f, "keccak"),
            HashFn::Blake3 => write!(f, "blake3"),
            HashFn::Blake3_192 => write!(f, "blake3-192"),
        }
    }
}

// ─── Trace spec parsing ──────────────────────────────────────────────────────

#[derive(Clone)]
struct TraceSpec {
    air_type: AirType,
    log_height: u8,
    /// Main trace width (miden only).
    width: usize,
    /// Extension-field auxiliary columns (miden only).
    num_aux_cols: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum AirType {
    Keccak,
    Poseidon2,
    Blake3,
    Miden,
}

impl fmt::Display for AirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Keccak => write!(f, "keccak"),
            Self::Poseidon2 => write!(f, "poseidon2"),
            Self::Blake3 => write!(f, "blake3"),
            Self::Miden => write!(f, "miden"),
        }
    }
}

impl FromStr for TraceSpec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() < 2 {
            return Err(format!(
                "expected <air>:<log_height>[:<width>[:<aux_cols>]], got '{s}'"
            ));
        }

        let air_type = match parts[0] {
            "keccak" | "k" => AirType::Keccak,
            "poseidon2" | "p" => AirType::Poseidon2,
            "blake3" | "b" => AirType::Blake3,
            "miden" | "m" => AirType::Miden,
            other => return Err(format!("unknown AIR type '{other}'")),
        };

        let log_height: u8 = parts[1]
            .parse()
            .map_err(|_| format!("invalid log_height '{}'", parts[1]))?;

        let width = if parts.len() > 2 {
            parts[2]
                .parse()
                .map_err(|_| format!("invalid width '{}'", parts[2]))?
        } else {
            DEFAULT_MIDEN_WIDTH
        };

        let num_aux_cols = if parts.len() > 3 {
            parts[3]
                .parse()
                .map_err(|_| format!("invalid aux_cols '{}'", parts[3]))?
        } else {
            DEFAULT_MIDEN_AUX_COLS
        };

        Ok(Self {
            air_type,
            log_height,
            width,
            num_aux_cols,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lifted mode — AIR enum
// ═══════════════════════════════════════════════════════════════════════════════

enum LiftedBenchAir {
    Keccak(LiftedKeccakAir),
    Poseidon2(Box<LiftedPoseidon2Air>),
    Blake3(LiftedBlake3Air),
    Miden(DummyMidenAir),
}

impl BaseAir<Felt> for LiftedBenchAir {
    fn width(&self) -> usize {
        match self {
            Self::Keccak(a) => BaseAir::<Felt>::width(a),
            Self::Poseidon2(a) => BaseAir::<Felt>::width(a.as_ref()),
            Self::Blake3(a) => BaseAir::<Felt>::width(a),
            Self::Miden(a) => BaseAir::<Felt>::width(a),
        }
    }
}

impl<EF: Field> LiftedAir<Felt, EF> for LiftedBenchAir {
    fn num_randomness(&self) -> usize {
        match self {
            Self::Miden(a) => LiftedAir::<Felt, EF>::num_randomness(a),
            _ => 1,
        }
    }

    fn aux_width(&self) -> usize {
        match self {
            Self::Miden(a) => LiftedAir::<Felt, EF>::aux_width(a),
            _ => 1,
        }
    }

    fn num_aux_values(&self) -> usize {
        match self {
            Self::Miden(a) => LiftedAir::<Felt, EF>::num_aux_values(a),
            _ => 0,
        }
    }

    fn num_var_len_public_inputs(&self) -> usize {
        0
    }

    fn eval<AB: LiftedAirBuilder<F = Felt>>(&self, builder: &mut AB) {
        match self {
            Self::Keccak(a) => LiftedAir::<Felt, EF>::eval(a, builder),
            Self::Poseidon2(a) => LiftedAir::<Felt, EF>::eval(a.as_ref(), builder),
            Self::Blake3(a) => LiftedAir::<Felt, EF>::eval(a, builder),
            Self::Miden(a) => LiftedAir::<Felt, EF>::eval(a, builder),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch mode — AIR enum with lookups
// ═══════════════════════════════════════════════════════════════════════════════

// ─── Keccak wrapper with a single local lookup ───────────────────────────────

/// Wraps [`KeccakAir`] and adds a single local LogUp lookup producing one
/// extension-field permutation column. Matches the lifted prover's unconditional
/// 1-column EF aux trace.
#[derive(Clone)]
struct KeccakWithLookup {
    num_lookups: usize,
}

impl<F> BaseAir<F> for KeccakWithLookup {
    fn width(&self) -> usize {
        NUM_KECCAK_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for KeccakWithLookup {
    fn eval(&self, builder: &mut AB) {
        Air::eval(&KeccakAir {}, builder);
    }
}

impl<F: Field> LookupAir<F> for KeccakWithLookup {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let idx = self.num_lookups;
        self.num_lookups += 1;
        vec![idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;
        let col0 = SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE));
        let one = SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE));
        let lookup_inputs = vec![
            (vec![col0.clone()], one.clone(), Direction::Send),
            (vec![col0], one, Direction::Receive),
        ];
        vec![LookupAir::register_lookup(
            self,
            Kind::Local,
            &lookup_inputs,
        )]
    }
}

// ─── Miden wrapper with N local lookups ──────────────────────────────────────

/// Wraps the Miden degree-9 constraint and adds `num_lookups_target` local LogUp
/// lookups, each producing one EF permutation column.
#[derive(Clone)]
struct MidenWithLookups {
    width: usize,
    num_lookups_target: usize,
    num_lookups: usize,
}

impl<F> BaseAir<F> for MidenWithLookups {
    fn width(&self) -> usize {
        self.width
    }
}

impl<AB: AirBuilder> Air<AB> for MidenWithLookups {
    fn eval(&self, builder: &mut AB) {
        // Same degree-9 constraint as DummyMidenAir.
        let main = builder.main();
        let local = main.current_slice();
        let product = (0..9).fold(AB::Expr::ONE, |acc, j| acc * local[j].into());
        builder.assert_zero(product);
    }
}

impl<F: Field> LookupAir<F> for MidenWithLookups {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        let idx = self.num_lookups;
        self.num_lookups += 1;
        vec![idx]
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        self.num_lookups = 0;
        let symbolic = SymbolicAirBuilder::<F>::new(AirLayout {
            main_width: self.width,
            ..AirLayout::default()
        });
        let main = symbolic.main();
        let local = main.current_slice();
        let col0: SymbolicExpression<F> = local[0].into();
        let one = SymbolicExpression::Leaf(BaseLeaf::Constant(F::ONE));
        let lookup_inputs = vec![
            (vec![col0.clone()], one.clone(), Direction::Send),
            (vec![col0], one, Direction::Receive),
        ];
        (0..self.num_lookups_target)
            .map(|_| LookupAir::register_lookup(self, Kind::Local, &lookup_inputs))
            .collect()
    }
}

// ─── Batch AIR enum ──────────────────────────────────────────────────────────

#[derive(Clone)]
enum BatchBenchAir {
    Keccak(KeccakWithLookup),
    Poseidon2(Box<BatchPoseidon2Air>),
    Blake3,
    Miden(MidenWithLookups),
}

impl<F> BaseAir<F> for BatchBenchAir {
    fn width(&self) -> usize {
        match self {
            Self::Keccak(a) => BaseAir::<F>::width(a),
            Self::Poseidon2(_) => NUM_POSEIDON2_COLS,
            Self::Blake3 => NUM_BLAKE3_COLS,
            Self::Miden(a) => BaseAir::<F>::width(a),
        }
    }
}

impl<AB: AirBuilder<F = Felt>> Air<AB> for BatchBenchAir {
    fn eval(&self, builder: &mut AB) {
        match self {
            Self::Keccak(a) => Air::eval(a, builder),
            Self::Poseidon2(a) => Air::eval(a.as_ref(), builder),
            Self::Blake3 => Air::eval(&Blake3Air {}, builder),
            Self::Miden(a) => Air::eval(a, builder),
        }
    }
}

impl<F: Field> LookupAir<F> for BatchBenchAir {
    fn add_lookup_columns(&mut self) -> Vec<usize> {
        match self {
            Self::Keccak(a) => <KeccakWithLookup as LookupAir<F>>::add_lookup_columns(a),
            Self::Miden(a) => <MidenWithLookups as LookupAir<F>>::add_lookup_columns(a),
            Self::Poseidon2(_) | Self::Blake3 => vec![],
        }
    }

    fn get_lookups(&mut self) -> Vec<Lookup<F>> {
        match self {
            Self::Keccak(a) => <KeccakWithLookup as LookupAir<F>>::get_lookups(a),
            Self::Miden(a) => <MidenWithLookups as LookupAir<F>>::get_lookups(a),
            Self::Poseidon2(_) | Self::Blake3 => vec![],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Run result
// ═══════════════════════════════════════════════════════════════════════════════

/// Captured output from a single prove/verify invocation.
struct RunResult {
    proof_size_bytes: usize,
    /// Number of field elements in the proof (lifted only, 0 for batch).
    field_elems: usize,
    /// Number of commitments in the proof (lifted only, 0 for batch).
    commitments: usize,
}

impl fmt::Display for RunResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "proof size: {}", format_bytes(self.proof_size_bytes))?;
        if self.field_elems > 0 || self.commitments > 0 {
            write!(
                f,
                " ({} field elems, {} commitments)",
                self.field_elems, self.commitments,
            )?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration summary
// ═══════════════════════════════════════════════════════════════════════════════

fn print_config(cli: &Cli, specs: &[TraceSpec], traces: &[RowMajorMatrix<Felt>], log_blowup: u8) {
    let mode = match cli.mode {
        Mode::Lifted => "lifted",
        Mode::Batch => "batch",
    };
    eprintln!("{:<20} {mode}", "mode:");
    eprintln!("{:<20} {}", "hash:", cli.hash);
    eprintln!("{:<20} {}", "seed:", cli.seed);
    for (i, (spec, trace)) in specs.iter().zip(traces).enumerate() {
        let label = if i == 0 { "traces:" } else { "" };
        eprintln!(
            "{:<20} {}:{} (width={}, 2^{} = {} rows)",
            label,
            spec.air_type,
            spec.log_height,
            trace.width(),
            spec.log_height,
            trace.height(),
        );
    }
    eprintln!("{:<20} {}", "log_blowup:", log_blowup);
    eprintln!("{:<20} {}", "log_folding_arity:", cli.log_folding_arity);
    eprintln!("{:<20} {}", "log_final_degree:", cli.log_final_degree);
    eprintln!("{:<20} {}", "num_queries:", cli.num_queries);
    eprintln!("{:<20} {}", "deep_pow_bits:", cli.deep_pow_bits);
    eprintln!("{:<20} {}", "folding_pow_bits:", cli.folding_pow_bits);
    eprintln!("{:<20} {}", "query_pow_bits:", cli.query_pow_bits);
    eprintln!();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Trace generation (shared between modes)
// ═══════════════════════════════════════════════════════════════════════════════

fn generate_traces(
    specs: &[TraceSpec],
    rng: &mut SmallRng,
    constants: Option<&GlRoundConstants>,
) -> Vec<RowMajorMatrix<Felt>> {
    specs
        .iter()
        .map(|spec| {
            info_span!("generate trace", air = %spec.air_type, log_height = spec.log_height)
                .in_scope(|| match spec.air_type {
                    AirType::Keccak => {
                        let n = (1usize << spec.log_height) / KECCAK_ROWS_PER_HASH;
                        let inputs: Vec<[u64; 25]> = (0..n).map(|_| rng.random()).collect();
                        generate_keccak_trace(inputs)
                    }
                    AirType::Poseidon2 => {
                        let n = 1usize << spec.log_height;
                        let inputs: Vec<[Felt; 12]> = (0..n).map(|_| rng.random()).collect();
                        generate_poseidon2_trace(
                            inputs,
                            constants.expect("poseidon2 constants required"),
                        )
                    }
                    AirType::Blake3 => {
                        let n = 1usize << spec.log_height;
                        let inputs: Vec<[u32; 24]> = (0..n).map(|_| rng.random()).collect();
                        generate_blake3_trace(inputs)
                    }
                    AirType::Miden => generate_dummy_trace(spec.width, spec.log_height),
                })
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Lifted prove/verify
// ═══════════════════════════════════════════════════════════════════════════════

fn run_lifted<SC>(
    config: &SC,
    specs: &[TraceSpec],
    traces: &[RowMajorMatrix<Felt>],
    constants: &Option<GlRoundConstants>,
    cli: &Cli,
) -> RunResult
where
    SC: StarkConfig<Felt, QuadFelt>,
    p3_miden_lifted_stark::StarkDigest<Felt, QuadFelt, SC>: PartialEq + fmt::Debug,
{
    let airs: Vec<LiftedBenchAir> = specs
        .iter()
        .map(|spec| match spec.air_type {
            AirType::Keccak => LiftedBenchAir::Keccak(LiftedKeccakAir),
            AirType::Poseidon2 => {
                let c = constants.as_ref().expect("poseidon2 constants required");
                LiftedBenchAir::Poseidon2(Box::new(LiftedPoseidon2Air::new(c.clone())))
            }
            AirType::Blake3 => LiftedBenchAir::Blake3(LiftedBlake3Air),
            AirType::Miden => {
                LiftedBenchAir::Miden(DummyMidenAir::new(spec.width, spec.num_aux_cols))
            }
        })
        .collect();

    let aux_builders: Vec<ZeroAuxBuilder> = specs
        .iter()
        .map(|spec| match spec.air_type {
            AirType::Miden => ZeroAuxBuilder {
                num_aux_cols: spec.num_aux_cols,
                num_aux_values: spec.num_aux_cols,
            },
            _ => ZeroAuxBuilder::dummy(),
        })
        .collect();

    let instances: Vec<_> = airs
        .iter()
        .zip(traces)
        .zip(&aux_builders)
        .map(|((air, trace), aux)| (air, AirWitness::new(trace, &[], &[]), aux))
        .collect();

    let output = info_span!("prove")
        .in_scope(|| prove_multi(config, &instances, config.challenger()).expect("proving failed"));

    let result = RunResult {
        proof_size_bytes: output.proof.size_in_bytes(),
        field_elems: output.proof.fields().len(),
        commitments: output.proof.commitments().len(),
    };

    if !cli.no_verify {
        info_span!("verify").in_scope(|| {
            let verifier_instances: Vec<_> = airs
                .iter()
                .zip(traces)
                .map(|(air, trace)| {
                    (
                        air,
                        AirInstance {
                            log_trace_height: log2_strict_u8(trace.height()),
                            public_values: &[],
                            var_len_public_inputs: &[],
                        },
                    )
                })
                .collect();
            let digest = verify_multi(
                config,
                &verifier_instances,
                &output.proof,
                config.challenger(),
            )
            .expect("verification failed");
            assert_eq!(output.digest, digest);
        });
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch prove/verify
// ═══════════════════════════════════════════════════════════════════════════════

fn run_batch<SC>(
    config: &SC,
    specs: &[TraceSpec],
    traces: &[RowMajorMatrix<Felt>],
    constants: &Option<GlRoundConstants>,
    cli: &Cli,
) -> RunResult
where
    SC: p3_uni_stark::StarkGenericConfig<Challenge = QuadFelt>,
    <SC::Pcs as p3_commit::Pcs<QuadFelt, SC::Challenger>>::Domain:
        p3_commit::PolynomialSpace<Val = Felt>,
{
    let mut airs: Vec<BatchBenchAir> = specs
        .iter()
        .map(|spec| match spec.air_type {
            AirType::Keccak => BatchBenchAir::Keccak(KeccakWithLookup { num_lookups: 0 }),
            AirType::Poseidon2 => {
                let c = constants.as_ref().expect("poseidon2 constants required");
                BatchBenchAir::Poseidon2(Box::new(BatchPoseidon2Air::new(c.clone())))
            }
            AirType::Blake3 => BatchBenchAir::Blake3,
            AirType::Miden => BatchBenchAir::Miden(MidenWithLookups {
                width: spec.width,
                num_lookups_target: spec.num_aux_cols,
                num_lookups: 0,
            }),
        })
        .collect();

    let degree_bits: Vec<usize> = traces
        .iter()
        .map(|t| log2_strict_u8(t.height()) as usize)
        .collect();
    let prover_data = ProverData::from_airs_and_degrees(config, &mut airs, &degree_bits);
    let common = &prover_data.common;

    let trace_refs: Vec<&RowMajorMatrix<Felt>> = traces.iter().collect();
    let pvs: Vec<Vec<Felt>> = specs.iter().map(|_| vec![]).collect();

    let instances = StarkInstance::new_multiple(&airs, &trace_refs, &pvs, common);

    let proof = info_span!("prove").in_scope(|| prove_batch(config, &instances, &prover_data));

    let result = RunResult {
        proof_size_bytes: postcard::to_allocvec(&proof)
            .expect("serialization failed")
            .len(),
        field_elems: 0,
        commitments: 0,
    };

    if !cli.no_verify {
        info_span!("verify").in_scope(|| {
            verify_batch(config, &airs, &proof, &pvs, common)
                .expect("batch-stark verification failed");
        });
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Batch config macro
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a `p3_uni_stark::StarkConfig` for batch-STARK from MMCS components.
///
/// Parameterized over packed field/digest types and digest size, since these
/// differ per hash function and cannot be inferred from the constructor.
macro_rules! batch_config {
    ($P:ty, $PD:ty, $DIGEST:expr, $leaf:expr, $compress:expr, $challenger:expr, $log_blowup:expr, $cli:expr) => {{
        type Dft = p3_dft::Radix2DitParallel<Felt>;
        let mmcs: MerkleTreeMmcs<$P, $PD, _, _, 2, $DIGEST> =
            MerkleTreeMmcs::new($leaf, $compress, 0);
        let challenge_mmcs = ExtensionMmcs::<Felt, QuadFelt, _>::new(mmcs.clone());
        let fri_params = FriParameters {
            log_blowup: $log_blowup as usize,
            log_final_poly_len: $cli.log_final_degree as usize,
            max_log_arity: $cli.log_folding_arity as usize,
            num_queries: $cli.num_queries,
            commit_proof_of_work_bits: $cli.folding_pow_bits,
            query_proof_of_work_bits: $cli.query_pow_bits,
            mmcs: challenge_mmcs,
        };
        let pcs = TwoAdicFriPcs::new(Dft::default(), mmcs, fri_params);
        p3_uni_stark::StarkConfig::new(pcs, $challenger)
    }};
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

fn main() {
    let cli = Cli::parse();

    // Apply defaults.
    let mut specs = if cli.traces.is_empty() {
        vec![
            TraceSpec {
                air_type: AirType::Blake3,
                log_height: 15,
                width: 0,
                num_aux_cols: 0,
            },
            TraceSpec {
                air_type: AirType::Keccak,
                log_height: 18,
                width: 0,
                num_aux_cols: 0,
            },
            TraceSpec {
                air_type: AirType::Poseidon2,
                log_height: 19,
                width: 0,
                num_aux_cols: 0,
            },
        ]
    } else {
        cli.traces.clone()
    };

    // Sort by ascending height (required by the lifted prover).
    specs.sort_by_key(|s| s.log_height);

    let has_miden = specs.iter().any(|s| s.air_type == AirType::Miden);
    let log_blowup = cli.log_blowup.unwrap_or(if has_miden { 3 } else { 1 });

    // Set up tracing subscriber (quiet by default, -v for full tree).
    init_tracing(cli.verbose);

    // Generate Poseidon2 round constants (from RNG, before trace inputs).
    let mut rng = SmallRng::seed_from_u64(cli.seed);
    let poseidon2_constants: Option<GlRoundConstants> =
        if specs.iter().any(|s| s.air_type == AirType::Poseidon2) {
            Some(RoundConstants::from_rng(&mut rng))
        } else {
            None
        };

    // Generate traces.
    let traces = generate_traces(&specs, &mut rng, poseidon2_constants.as_ref());

    // Print configuration summary.
    print_config(&cli, &specs, &traces, log_blowup);

    // Build PCS params (shared across hash functions).
    let pcs = PcsParams::new(
        log_blowup,
        cli.log_folding_arity,
        cli.log_final_degree,
        cli.folding_pow_bits,
        cli.deep_pow_bits,
        cli.num_queries,
        cli.query_pow_bits,
    )
    .expect("invalid PCS params");

    type Dft = p3_dft::Radix2DitParallel<Felt>;

    // Run prove/verify.
    let start = Instant::now();
    let result = match cli.mode {
        Mode::Lifted => match cli.hash {
            HashFn::Poseidon2 => {
                let config = GenericStarkConfig::new(
                    pcs,
                    gl::test_lmcs(),
                    Dft::default(),
                    gl::test_challenger(),
                );
                run_lifted(&config, &specs, &traces, &poseidon2_constants, &cli)
            }
            HashFn::Keccak => {
                let config = GenericStarkConfig::new(
                    pcs,
                    keccak::test_lmcs(),
                    Dft::default(),
                    keccak::test_challenger(),
                );
                run_lifted(&config, &specs, &traces, &poseidon2_constants, &cli)
            }
            HashFn::Blake3 => {
                let config = GenericStarkConfig::new(
                    pcs,
                    blake3::test_lmcs(),
                    Dft::default(),
                    blake3::test_challenger(),
                );
                run_lifted(&config, &specs, &traces, &poseidon2_constants, &cli)
            }
            HashFn::Blake3_192 => {
                let config = GenericStarkConfig::new(
                    pcs,
                    blake3_192::test_lmcs(),
                    Dft::default(),
                    blake3_192::test_challenger(),
                );
                run_lifted(&config, &specs, &traces, &poseidon2_constants, &cli)
            }
        },
        Mode::Batch => match cli.hash {
            HashFn::Poseidon2 => {
                let (perm, _, compress) = gl::test_components();
                let leaf = PaddingFreeSponge::<_, { gl::WIDTH }, { gl::RATE }, { gl::DIGEST }>::new(
                    perm.clone(),
                );
                let config = batch_config!(
                    gl::PackedFelt,
                    gl::PackedFelt,
                    { gl::DIGEST },
                    leaf,
                    compress,
                    gl::test_challenger(),
                    log_blowup,
                    &cli
                );
                run_batch(&config, &specs, &traces, &poseidon2_constants, &cli)
            }
            HashFn::Keccak => {
                type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
                let u64_hash = U64Hash::new(KeccakF);
                let leaf = SerializingHasher::new(u64_hash);
                let compress = CompressionFunctionFromHasher::<U64Hash, 2, 4>::new(u64_hash);
                let config = batch_config!(
                    [Felt; p3_keccak::VECTOR_LEN],
                    [u64; p3_keccak::VECTOR_LEN],
                    4,
                    leaf,
                    compress,
                    keccak::test_challenger(),
                    log_blowup,
                    &cli
                );
                run_batch(&config, &specs, &traces, &poseidon2_constants, &cli)
            }
            HashFn::Blake3 => {
                let leaf = SerializingHasher::new(p3_blake3::Blake3);
                let compress =
                    CompressionFunctionFromHasher::<p3_blake3::Blake3, 2, { blake3::DIGEST }>::new(
                        p3_blake3::Blake3,
                    );
                let config = batch_config!(
                    Felt,
                    u8,
                    { blake3::DIGEST },
                    leaf,
                    compress,
                    blake3::test_challenger(),
                    log_blowup,
                    &cli
                );
                run_batch(&config, &specs, &traces, &poseidon2_constants, &cli)
            }
            HashFn::Blake3_192 => {
                let h = blake3_192::Blake3_192::new(p3_blake3::Blake3);
                let leaf = SerializingHasher::new(h);
                let compress = CompressionFunctionFromHasher::<
                    blake3_192::Blake3_192,
                    2,
                    { blake3_192::DIGEST },
                >::new(h);
                let config = batch_config!(
                    Felt,
                    u8,
                    { blake3_192::DIGEST },
                    leaf,
                    compress,
                    blake3_192::test_challenger(),
                    log_blowup,
                    &cli
                );
                run_batch(&config, &specs, &traces, &poseidon2_constants, &cli)
            }
        },
    };
    let elapsed = start.elapsed();

    println!("{result}");
    println!("total time: {:.3} s", elapsed.as_secs_f64());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn init_tracing(verbose: bool) {
    let default_level = if verbose {
        tracing_forest::util::LevelFilter::DEBUG
    } else {
        tracing_forest::util::LevelFilter::WARN
    };

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(default_level.into())
        .from_env_lossy();

    Registry::default()
        .with(tracing_forest::ForestLayer::default().with_filter(env_filter))
        .init();
}

fn format_bytes(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}
