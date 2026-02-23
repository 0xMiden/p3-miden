//! Test and benchmark fixtures (constants, scenarios).
//!
//! This module contains constants and helper functions that define
//! reproducible test/benchmark scenarios.

use alloc::vec;
use alloc::vec::Vec;

use p3_field::PackedValue;

// =============================================================================
// Seeds
// =============================================================================

/// Standard seed for reproducible tests/benchmarks.
pub const TEST_SEED: u64 = 2025;

/// Alias for benchmark seed (same value as TEST_SEED).
pub const BENCH_SEED: u64 = TEST_SEED;

// =============================================================================
// Benchmark constants
// =============================================================================

/// Standard log heights for benchmarking: 2^16, 2^18, 2^20 leaves.
pub const LOG_HEIGHTS: &[usize] = &[16, 18, 20];

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

// =============================================================================
// Matrix scenarios
// =============================================================================

/// Common matrix group scenarios for testing lifting with varying heights.
///
/// Each scenario is a list of (height, width) pairs, sorted by ascending height.
/// The `rate` parameter controls the RATE-based width scenarios.
///
/// # Parameters
/// - `pack_width`: The SIMD packing width (e.g., `P::WIDTH` for packed field)
/// - `rate`: The sponge rate for width alignment scenarios
pub fn matrix_scenarios<P: PackedValue>(rate: usize) -> Vec<Vec<(usize, usize)>> {
    let pack_width = P::WIDTH.max(2);
    vec![
        // Single matrices
        vec![(1, 1)],
        vec![(1, rate - 1)],
        // Multiple heights (must be ascending)
        vec![(2, 3), (4, 5), (8, rate)],
        vec![(1, 5), (1, 3), (2, 7), (4, 1), (8, rate + 1)],
        // Packing boundary tests
        vec![
            (pack_width / 2, rate - 1),
            (pack_width, rate),
            (pack_width * 2, rate + 3),
        ],
        vec![(pack_width, rate + 5), (pack_width * 2, 25)],
        vec![
            (1, rate * 2),
            (pack_width / 2, rate * 2 - 1),
            (pack_width, rate * 2),
            (pack_width * 2, rate * 3 - 2),
        ],
        // Same-height matrices
        vec![(4, rate - 1), (4, rate), (8, rate + 3), (8, rate * 2)],
        // Single tall matrix
        vec![(pack_width * 2, rate - 1)],
    ]
}
