//! Wraps Plonky3's [`Blake3Air`] as a [`LiftedAir`].

use alloc::vec::Vec;

use p3_air::Air;
use p3_blake3_air::{Blake3Air, NUM_BLAKE3_COLS};
use p3_field::{Field, PrimeField64};
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lifted_air::{AirWithPeriodicColumns, BaseAir, LiftedAir, LiftedAirBuilder};

use crate::compat::UpstreamCompat;

/// [`Blake3Air`] adapted for the lifted STARK prover.
///
/// Blake3 is a main-trace-only AIR with no preprocessed, periodic, or auxiliary columns.
/// Each row represents one full Blake3 compression (1 row per hash).
pub struct LiftedBlake3Air;

impl Default for LiftedBlake3Air {
    fn default() -> Self {
        Self
    }
}

impl<F> BaseAir<F> for LiftedBlake3Air {
    fn width(&self) -> usize {
        NUM_BLAKE3_COLS
    }
}

impl<F: Field> AirWithPeriodicColumns<F> for LiftedBlake3Air {
    fn periodic_columns(&self) -> &[Vec<F>] {
        &[]
    }
}

impl<F: PrimeField64, EF: Field> LiftedAir<F, EF> for LiftedBlake3Air {
    fn num_randomness(&self) -> usize {
        1
    }

    fn aux_width(&self) -> usize {
        1
    }

    fn num_aux_values(&self) -> usize {
        0
    }

    fn eval<AB: LiftedAirBuilder<F = F>>(&self, builder: &mut AB) {
        Air::eval(&Blake3Air {}, &mut UpstreamCompat(builder));
    }
}

/// Generate a Blake3 trace for the given inputs.
///
/// Each input is 24 `u32` values: 16 block words followed by 8 chaining values.
/// The trace has `inputs.len()` rows (must be a power of two) and
/// [`NUM_BLAKE3_COLS`] columns.
pub fn generate_blake3_trace<F: PrimeField64>(inputs: Vec<[u32; 24]>) -> RowMajorMatrix<F> {
    p3_blake3_air::generate_trace_rows(inputs, 0)
}
