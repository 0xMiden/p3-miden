//! Wraps Plonky3's [`Poseidon2Air`] as a [`LiftedAir`].
//!
//! Uses the standard BabyBear configuration: WIDTH=16, SBOX_DEGREE=7, SBOX_REGISTERS=1,
//! HALF_FULL_ROUNDS=4, PARTIAL_ROUNDS=20.

use alloc::vec::Vec;

use p3_air::{Air, BaseAir, BaseAirWithPublicValues};
use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lifted_air::{AirWithPeriodicColumns, LiftedAir, LiftedAirBuilder};
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, num_cols};

/// BabyBear Poseidon2 configuration constants.
pub const WIDTH: usize = 16;
pub const SBOX_DEGREE: u64 = 7;
pub const SBOX_REGISTERS: usize = 1;
pub const HALF_FULL_ROUNDS: usize = 4;
pub const PARTIAL_ROUNDS: usize = 20;

/// Number of trace columns for the BabyBear Poseidon2 AIR.
pub const NUM_POSEIDON2_COLS: usize =
    num_cols::<WIDTH, SBOX_DEGREE, SBOX_REGISTERS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>();

type BabyBearRoundConstants = RoundConstants<BabyBear, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>;

type InnerAir = Poseidon2Air<
    BabyBear,
    GenericPoseidon2LinearLayersBabyBear,
    WIDTH,
    SBOX_DEGREE,
    SBOX_REGISTERS,
    HALF_FULL_ROUNDS,
    PARTIAL_ROUNDS,
>;

/// [`Poseidon2Air`] adapted for the lifted STARK prover.
///
/// Poseidon2 is a main-trace-only AIR with no preprocessed, periodic, or auxiliary columns.
/// Each row represents one full Poseidon2 permutation (1 row per hash).
pub struct LiftedPoseidon2Air {
    inner: InnerAir,
}

impl LiftedPoseidon2Air {
    pub fn new(constants: BabyBearRoundConstants) -> Self {
        Self {
            inner: InnerAir::new(constants),
        }
    }
}

impl<F> BaseAir<F> for LiftedPoseidon2Air {
    fn width(&self) -> usize {
        NUM_POSEIDON2_COLS
    }
}

impl<F> BaseAirWithPublicValues<F> for LiftedPoseidon2Air {}

impl<F: Field> AirWithPeriodicColumns<F> for LiftedPoseidon2Air {
    fn periodic_columns(&self) -> &[Vec<F>] {
        &[]
    }
}

impl<EF: Field> LiftedAir<BabyBear, EF> for LiftedPoseidon2Air {
    fn num_randomness(&self) -> usize {
        1
    }

    fn aux_width(&self) -> usize {
        1
    }

    fn num_aux_values(&self) -> usize {
        0
    }

    fn eval<AB: LiftedAirBuilder<F = BabyBear>>(&self, builder: &mut AB) {
        Air::<AB>::eval(&self.inner, builder);
    }
}

/// Generate a Poseidon2 trace for the given inputs.
///
/// Each input is a WIDTH=16 element BabyBear array. The number of inputs must
/// be a power of two. The trace has `inputs.len()` rows and
/// [`NUM_POSEIDON2_COLS`] columns.
pub fn generate_poseidon2_trace(
    inputs: Vec<[BabyBear; WIDTH]>,
    constants: &BabyBearRoundConstants,
) -> RowMajorMatrix<BabyBear> {
    p3_poseidon2_air::generate_trace_rows::<
        BabyBear,
        GenericPoseidon2LinearLayersBabyBear,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(inputs, constants, 0)
}
