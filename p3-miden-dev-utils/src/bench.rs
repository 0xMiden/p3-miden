//! Benchmark-specific utilities (Criterion config, parallel detection).

extern crate std;

use std::time::Duration;

use criterion::Criterion;

// =============================================================================
// Criterion configuration
// =============================================================================

/// Standard Criterion configuration for p3-miden benchmarks.
///
/// Settings: sample_size=10, measurement_time=12s, warm_up_time=3s
pub fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(12))
        .warm_up_time(Duration::from_secs(3))
}

/// Configuration for longer-running benchmarks (e.g., PCS).
///
/// Settings: sample_size=10, measurement_time=30s, warm_up_time=3s
pub fn criterion_config_long() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(30))
        .warm_up_time(Duration::from_secs(3))
}

// =============================================================================
// Parallelism detection
// =============================================================================

/// Parallelism mode string for benchmark grouping.
pub const PARALLEL_STR: &str = if cfg!(feature = "parallel") {
    "parallel"
} else {
    "single"
};
