mod common;

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_air::{
    AirBuilder, AirWithPeriodicColumns, AuxBuilder, BaseAir, BaseAirWithPublicValues,
    ExtensionBuilder, LiftedAir, LiftedAirBuilder, ReducedAuxValues, VarLenPublicInputs,
};
use p3_miden_lifted_prover::prove_multi;
use p3_miden_lifted_stark::AirInstance;
use p3_miden_lifted_verifier::verify_multi;
use p3_miden_transcript::{ProverTranscript, VerifierTranscript};
use p3_util::log2_strict_usize;

use common::test_config;

extern crate alloc;

// ---------------------------------------------------------------------------
// BusTestAir: exercises reduced_aux_values with multiset + logup buses.
//
// Main trace: 1 column, power-of-4 chain (same as TinyAir).
// Aux trace: 2 constant columns (all rows identical):
//   col 0: 1/(pi_0 + challenge[0])  — inverse for multiset bus
//   col 1: pi_1 + challenge[1]      — accumulator for logup bus
//
// Aux values (committed to transcript, constrained to match aux trace last row):
//   aux_values[0] = col 0 value = 1/(pi_0 + c0)
//   aux_values[1] = col 1 value = pi_1 + c1
//
// reduced_aux_values (verifier-side bus identity check):
//   Bus 0 (multiset): prod = aux_values[0] * (c0 + pi_0) == 1
//   Bus 1 (logup):    sum  = (aux_values[1] - c1) - pi_1 == 0
//
// pi_0, pi_1 appear in two places:
//   - public_values[1..]: used by eval() for aux trace constraints
//   - var_len_public_inputs: used by reduced_aux_values() for bus check
// Both must agree for the proof to verify.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct BusTestAir;

impl BaseAir<bb::F> for BusTestAir {
    fn width(&self) -> usize {
        1
    }
}

impl BaseAirWithPublicValues<bb::F> for BusTestAir {
    fn num_public_values(&self) -> usize {
        3 // [start, pi_0, pi_1]
    }
}

impl AirWithPeriodicColumns<bb::F> for BusTestAir {
    fn periodic_columns(&self) -> &[Vec<bb::F>] {
        &[]
    }
}

impl LiftedAir<bb::F, bb::EF> for BusTestAir {
    fn aux_width(&self) -> usize {
        2
    }

    fn num_aux_values(&self) -> usize {
        2
    }

    fn num_randomness(&self) -> usize {
        2
    }

    fn reduced_aux_values(
        &self,
        aux_values: &[bb::EF],
        challenges: &[bb::EF],
        _public_values: &[bb::F],
        var_len_public_inputs: VarLenPublicInputs<'_, bb::F>,
    ) -> ReducedAuxValues<bb::EF> {
        // Bus 0 (multiset): prod = aux_values[0] * (challenges[0] + pi_0)
        // aux_values[0] = 1/(pi_0 + c0), so prod == 1 when pi_0 matches.
        let pi_0 = bb::EF::from(var_len_public_inputs[0][0][0]);
        let prod = aux_values[0] * (challenges[0] + pi_0);

        // Bus 1 (logup): sum = (aux_values[1] - challenges[1]) - pi_1
        // aux_values[1] = pi_1 + c1, so sum == 0 when pi_1 matches.
        let pi_1 = bb::EF::from(var_len_public_inputs[1][0][0]);
        let sum = (aux_values[1] - challenges[1]) - pi_1;

        ReducedAuxValues { prod, sum }
    }

    fn eval<AB: LiftedAirBuilder<F = bb::F>>(&self, builder: &mut AB) {
        // Copy public values upfront (PublicVar: Copy) to release borrow.
        let pv0 = builder.public_values()[0];
        let pv1 = builder.public_values()[1];
        let pv2 = builder.public_values()[2];

        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("empty matrix"),
            main.row_slice(1).expect("single row matrix"),
        );

        // Main trace: power-of-4 chain
        builder.when_first_row().assert_eq(local[0].clone(), pv0);
        let main_pow4: AB::Expr = local[0].clone().into().exp_power_of_2(2);
        builder
            .when_transition()
            .assert_eq(next[0].clone(), main_pow4);

        // Copy challenges and aux values (RandomVar/VarEF: Copy) to release borrow.
        let c0: AB::RandomVar = builder.permutation_randomness()[0];
        let c1: AB::RandomVar = builder.permutation_randomness()[1];
        let av0: AB::VarEF = builder.aux_values()[0];
        let av1: AB::VarEF = builder.aux_values()[1];

        let aux = builder.permutation();
        let aux_local = aux.row_slice(0).expect("empty aux");
        let aux_next = aux.row_slice(1).expect("single row aux");

        // pi_0 = public_values[1], pi_1 = public_values[2]
        let pi_0: AB::ExprEF = Into::<AB::Expr>::into(pv1).into();
        let pi_1: AB::ExprEF = Into::<AB::Expr>::into(pv2).into();
        let c0: AB::ExprEF = c0.into();
        let c1: AB::ExprEF = c1.into();

        // First row: aux[0] * (pi_0 + c0) == 1
        let a0: AB::ExprEF = aux_local[0].into();
        builder
            .when_first_row()
            .assert_eq_ext(a0 * (pi_0 + c0), AB::ExprEF::ONE);

        // First row: aux[1] == pi_1 + c1
        let a1: AB::ExprEF = aux_local[1].into();
        builder.when_first_row().assert_eq_ext(a1, pi_1 + c1);

        // Transition: constant columns
        builder
            .when_transition()
            .assert_eq_ext::<AB::ExprEF, AB::ExprEF>(aux_next[0].into(), aux_local[0].into());
        builder
            .when_transition()
            .assert_eq_ext::<AB::ExprEF, AB::ExprEF>(aux_next[1].into(), aux_local[1].into());

        // Last row: aux columns match aux_values
        builder
            .when_last_row()
            .assert_eq_ext::<AB::ExprEF, AB::ExprEF>(aux_local[0].into(), av0.into());
        builder
            .when_last_row()
            .assert_eq_ext::<AB::ExprEF, AB::ExprEF>(aux_local[1].into(), av1.into());
    }
}

// ---------------------------------------------------------------------------
// AuxBuilder: constant aux columns.
// ---------------------------------------------------------------------------

struct BusTestAuxBuilder {
    pi_0: bb::F,
    pi_1: bb::F,
}

impl AuxBuilder<bb::F, bb::EF> for BusTestAuxBuilder {
    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<bb::F>,
        challenges: &[bb::EF],
    ) -> (RowMajorMatrix<bb::EF>, Vec<bb::EF>) {
        let height = main.height();
        let c0 = challenges[0];
        let c1 = challenges[1];

        // col 0: 1/(pi_0 + c0), col 1: pi_1 + c1
        let col0_val = (bb::EF::from(self.pi_0) + c0).inverse();
        let col1_val = bb::EF::from(self.pi_1) + c1;

        let mut values = Vec::with_capacity(height * 2);
        for _ in 0..height {
            values.push(col0_val);
            values.push(col1_val);
        }

        let aux_trace = RowMajorMatrix::new(values, 2);
        let aux_values = vec![col0_val, col1_val];
        (aux_trace, aux_values)
    }
}

// ---------------------------------------------------------------------------
// Trace generation (same power-of-4 chain as TinyAir)
// ---------------------------------------------------------------------------

fn generate_trace(start: bb::F, height: usize) -> RowMajorMatrix<bb::F> {
    let mut values = Vec::with_capacity(height);
    let mut current = start;
    for _ in 0..height {
        values.push(current);
        current = current.exp_power_of_2(2);
    }
    RowMajorMatrix::new(values, 1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn bus_identity_check() {
    let config = test_config();

    let pi_0 = bb::F::from_u64(42);
    let pi_1 = bb::F::from_u64(67);
    let start = bb::F::from_u64(2);
    let height = 8;

    let air = BusTestAir;
    let aux_builder = BusTestAuxBuilder { pi_0, pi_1 };
    let trace = generate_trace(start, height);
    let public_values = vec![start, pi_0, pi_1];

    // Prove
    let prover_instances = [(
        &air,
        p3_miden_lifted_prover::AirWitness::new(&trace, &public_values),
        &aux_builder,
    )];
    let mut prover_channel = ProverTranscript::new(bb::test_challenger());
    prove_multi(&config, &prover_instances, &mut prover_channel).expect("proving should succeed");
    let transcript = prover_channel.into_data();

    // Build var_len_public_inputs for the verifier
    let pi_0_row = [pi_0];
    let pi_1_row = [pi_1];
    let bus_0_rows: &[&[bb::F]] = &[&pi_0_row];
    let bus_1_rows: &[&[bb::F]] = &[&pi_1_row];
    let buses: [&[&[bb::F]]; 2] = [bus_0_rows, bus_1_rows];

    let instance = AirInstance {
        log_trace_height: log2_strict_usize(height),
        public_values: &public_values,
        var_len_public_inputs: &buses,
    };

    // Verify
    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    verify_multi(&config, &[(&air, instance)], &mut verifier_channel)
        .expect("verification should succeed");
}

#[test]
fn bus_wrong_var_len_pi_fails() {
    let config = test_config();

    let pi_0 = bb::F::from_u64(42);
    let pi_1 = bb::F::from_u64(67);
    let start = bb::F::from_u64(2);
    let height = 8;

    let air = BusTestAir;
    let aux_builder = BusTestAuxBuilder { pi_0, pi_1 };
    let trace = generate_trace(start, height);
    let public_values = vec![start, pi_0, pi_1];

    // Prove with correct values
    let prover_instances = [(
        &air,
        p3_miden_lifted_prover::AirWitness::new(&trace, &public_values),
        &aux_builder,
    )];
    let mut prover_channel = ProverTranscript::new(bb::test_challenger());
    prove_multi(&config, &prover_instances, &mut prover_channel).expect("proving should succeed");
    let transcript = prover_channel.into_data();

    // Verify with WRONG var_len_public_inputs (99 instead of 42)
    let wrong_pi_0 = bb::F::from_u64(99);
    let wrong_row = [wrong_pi_0];
    let pi_1_row = [pi_1];
    let bus_0_rows: &[&[bb::F]] = &[&wrong_row];
    let bus_1_rows: &[&[bb::F]] = &[&pi_1_row];
    let buses: [&[&[bb::F]]; 2] = [bus_0_rows, bus_1_rows];

    let instance = AirInstance {
        log_trace_height: log2_strict_usize(height),
        public_values: &public_values,
        var_len_public_inputs: &buses,
    };

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    let err = verify_multi(&config, &[(&air, instance)], &mut verifier_channel)
        .expect_err("wrong var_len_pi should fail verification");

    assert!(
        matches!(
            err,
            p3_miden_lifted_verifier::VerifierError::InvalidReducedAux
        ),
        "expected InvalidReducedAux, got {err:?}"
    );
}
