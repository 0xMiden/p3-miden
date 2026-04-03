//! Shared constants and parameter sets for tests and benchmarks.
//!
//! Centralizes magic numbers so they can be tuned in one place and
//! referenced consistently across unit tests, criterion benches, and
//! profiling binaries.

use crate::pcs::{
    deep::DeepParams,
    fri::{FriParams, fold::FriFold},
    params::PcsParams,
};

// =============================================================================
// Seeds
// =============================================================================

/// Standard seed for reproducible tests and benchmarks.
pub const TEST_SEED: u64 = 2025;

// =============================================================================
// Benchmark matrix shapes
// =============================================================================

/// Standard log heights for benchmarking: 2^16, 2^18, 2^20 leaves.
pub const LOG_HEIGHTS: &[u8] = &[16, 18, 20];

/// Standard relative specs for benchmark matrix groups.
///
/// Each inner slice is a separate commitment group.
/// Tuple format: `(offset_from_max, width)` where `log_height = log_max_height - offset`.
///
/// This gives realistic matrix configurations similar to STARK traces:
/// - Group 0: Main trace columns at various heights
/// - Group 1: Auxiliary/permutation columns
/// - Group 2: Quotient polynomial chunks
pub const RELATIVE_SPECS: &[&[(usize, usize)]] = &[
    &[(4, 10), (2, 100), (0, 50)],
    &[(4, 8), (2, 20), (0, 20)],
    &[(0, 16)],
];

/// Label for benchmark group names indicating parallelism mode.
pub const PARALLEL_STR: &str = if cfg!(feature = "parallel") {
    "parallel"
} else {
    "single"
};

// =============================================================================
// PCS parameter sets
// =============================================================================

/// PCS parameters for unit tests (fast, minimal security).
pub const TEST_PCS_PARAMS: PcsParams = PcsParams {
    deep: DeepParams { deep_pow_bits: 0 },
    fri: FriParams {
        log_blowup: 2,
        fold: FriFold::ARITY_4,
        log_final_degree: 2,
        folding_pow_bits: 0,
    },
    num_queries: 2,
    query_pow_bits: 0,
};

/// PCS parameters for benchmarks (realistic security, zero PoW).
pub const BENCH_PCS_PARAMS: PcsParams = PcsParams {
    deep: DeepParams { deep_pow_bits: 0 },
    fri: FriParams {
        log_blowup: 2,
        fold: FriFold::ARITY_4,
        log_final_degree: 8,
        folding_pow_bits: 0,
    },
    num_queries: 30,
    query_pow_bits: 0,
};

/// PCS parameters for quotient commit benchmarks (lower blowup, single query).
pub const QC_PCS_PARAMS: PcsParams = PcsParams {
    deep: DeepParams { deep_pow_bits: 0 },
    fri: FriParams {
        log_blowup: 1,
        fold: FriFold::ARITY_4,
        log_final_degree: 0,
        folding_pow_bits: 0,
    },
    num_queries: 1,
    query_pow_bits: 0,
};

/// Constraint degree used in quotient commit benchmarks (matches KeccakAir).
pub const QC_CONSTRAINT_DEGREE: usize = 2;

// =============================================================================
// Profiling binary parameters
// =============================================================================

pub const PROFILE_NUM_QUERIES: usize = 100;
pub const PROFILE_POW_BITS: usize = 16;

/// Build PCS parameters for profiling binaries with a custom blowup factor.
pub fn profile_pcs_params(log_blowup: u8) -> PcsParams {
    PcsParams {
        deep: DeepParams {
            deep_pow_bits: PROFILE_POW_BITS,
        },
        fri: FriParams {
            log_blowup,
            fold: FriFold::ARITY_4,
            log_final_degree: 0,
            folding_pow_bits: 0,
        },
        num_queries: PROFILE_NUM_QUERIES,
        query_pow_bits: 0,
    }
}
