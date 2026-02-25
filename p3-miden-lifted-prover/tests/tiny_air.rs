mod common;

use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_air::{
    AirBuilder, AirWithPeriodicColumns, AuxBuilder, BaseAir, BaseAirWithPublicValues,
    ExtensionBuilder, LiftedAir, LiftedAirBuilder,
};
use p3_miden_lifted_prover::prove_single;
use p3_miden_lifted_verifier::{VerifierError, verify_single};
use p3_miden_transcript::{ProverTranscript, TranscriptData, VerifierTranscript};
use p3_util::log2_strict_usize;

use common::{prove_and_verify, test_config};

extern crate alloc;

// ---------------------------------------------------------------------------
// TinyAir: main[0] starts at public_values[0], each row is previous^4.
// Optional periodic columns with pattern [1, 0, ..., 0, 1] per period.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct TinyAir {
    /// Pre-computed periodic column data.
    periodic_cols: Vec<Vec<bb::F>>,
}

impl TinyAir {
    fn new(periods: Vec<usize>) -> Self {
        let periodic_cols = periods
            .iter()
            .map(|&p| {
                let mut col = vec![bb::F::ZERO; p];
                col[0] = bb::F::ONE;
                col[p - 1] = bb::F::ONE;
                col
            })
            .collect();
        Self { periodic_cols }
    }
}

impl BaseAir<bb::F> for TinyAir {
    fn width(&self) -> usize {
        1
    }
}

impl BaseAirWithPublicValues<bb::F> for TinyAir {
    fn num_public_values(&self) -> usize {
        1
    }
}

impl AirWithPeriodicColumns<bb::F> for TinyAir {
    fn periodic_columns(&self) -> &[Vec<bb::F>] {
        &self.periodic_cols
    }
}

impl LiftedAir<bb::F, bb::EF> for TinyAir {
    fn aux_width(&self) -> usize {
        1
    }

    fn num_aux_values(&self) -> usize {
        1
    }

    fn num_randomness(&self) -> usize {
        1
    }

    fn eval<AB: LiftedAirBuilder<F = bb::F>>(&self, builder: &mut AB) {
        let main = builder.main();
        let start = builder.public_values()[0];
        let periodic = builder.periodic_values().to_vec();
        let (local, next) = (
            main.row_slice(0).expect("empty matrix"),
            main.row_slice(1).expect("single row matrix"),
        );

        // First row: main[0] = public_values[0]
        builder.when_first_row().assert_eq(local[0].clone(), start);

        // Transition: main_next = main^4
        let main_pow4: AB::Expr = local[0].clone().into().exp_power_of_2(2);
        builder
            .when_transition()
            .assert_eq(next[0].clone(), main_pow4);

        // Periodic column constraints: first and last row see 1
        for p in &periodic {
            let p_expr: AB::Expr = (*p).into();
            builder.when_first_row().assert_one(p_expr.clone());
            builder.when_last_row().assert_one(p_expr);
        }

        // Aux trace constraints
        let aux = builder.permutation();
        let aux_local = aux.row_slice(0).expect("empty aux");
        let aux_next = aux.row_slice(1).expect("single row aux");
        let challenge: AB::ExprEF = builder.permutation_randomness()[0].into();

        let aux_local_ef: AB::ExprEF = aux_local[0].into();
        builder
            .when_first_row()
            .assert_eq_ext(aux_local_ef.clone(), challenge);

        let aux_pow4: AB::ExprEF = aux_local_ef.exp_power_of_2(2);
        builder
            .when_transition()
            .assert_eq_ext(aux_next[0].into(), aux_pow4);
    }
}

/// AuxBuilder for TinyAir: aux column = challenge^{4^row}.
struct TinyAuxBuilder;

impl AuxBuilder<bb::F, bb::EF> for TinyAuxBuilder {
    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<bb::F>,
        challenges: &[bb::EF],
    ) -> (RowMajorMatrix<bb::EF>, Vec<bb::EF>) {
        let height = main.height();
        let challenge = challenges[0];

        let mut col_values = Vec::with_capacity(height);
        let mut current = challenge;
        for _ in 0..height {
            col_values.push(current);
            current = current.exp_power_of_2(2);
        }

        let aux_trace = RowMajorMatrix::new(col_values.clone(), 1);
        let aux_values = vec![col_values[height - 1]];
        (aux_trace, aux_values)
    }
}

/// Generate a trace: [start, start^4, start^16, start^64, ...]
fn generate_trace(start: bb::F, height: usize) -> RowMajorMatrix<bb::F> {
    let mut values = Vec::with_capacity(height);
    let mut current = start;
    for _ in 0..height {
        values.push(current);
        current = current.exp_power_of_2(2);
    }
    RowMajorMatrix::new(values, 1)
}

/// Build a (trace, public_values) pair for instance `idx`.
fn instance(idx: usize, height: usize) -> (RowMajorMatrix<bb::F>, Vec<bb::F>) {
    let start = bb::F::from_u64((idx + 2) as u64);
    (generate_trace(start, height), vec![start])
}

// ---------------------------------------------------------------------------
// Single-trace tests
// ---------------------------------------------------------------------------

#[test]
fn single_trace() {
    prove_and_verify(&TinyAir::new(vec![]), &TinyAuxBuilder, &[instance(0, 8)]);
}

#[test]
fn malformed_transcript_is_rejected() {
    let config = test_config();
    let air = TinyAir::new(vec![]);

    let (trace, public_values) = instance(0, 4);
    let log_trace_height = log2_strict_usize(trace.height());

    let mut prover_channel = ProverTranscript::new(bb::test_challenger());
    prove_single(
        &config,
        &air,
        &trace,
        &public_values,
        &TinyAuxBuilder,
        &mut prover_channel,
    )
    .expect("proving should succeed");
    let transcript = prover_channel.into_data();

    // Baseline should verify
    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    verify_single(
        &config,
        &air,
        log_trace_height,
        &public_values,
        &mut verifier_channel,
    )
    .expect("baseline proof should verify");

    // Extra field element should cause rejection
    let (mut fields, commitments) = transcript.clone().into_parts();
    fields.push(bb::F::ONE);
    let bad_transcript = TranscriptData::new(fields, commitments);

    let mut bad_channel = VerifierTranscript::from_data(bb::test_challenger(), &bad_transcript);
    let err = verify_single(
        &config,
        &air,
        log_trace_height,
        &public_values,
        &mut bad_channel,
    )
    .expect_err("extra transcript data should fail verification");
    assert!(matches!(err, VerifierError::TranscriptNotConsumed));
}

// ---------------------------------------------------------------------------
// Multi-trace tests
// ---------------------------------------------------------------------------

#[test]
fn two_traces_same_height() {
    prove_and_verify(
        &TinyAir::new(vec![]),
        &TinyAuxBuilder,
        &[instance(0, 8), instance(1, 8)],
    );
}

#[test]
fn two_traces_different_heights() {
    prove_and_verify(
        &TinyAir::new(vec![]),
        &TinyAuxBuilder,
        &[instance(0, 4), instance(1, 8)],
    );
}

#[test]
fn three_traces_ascending_heights() {
    prove_and_verify(
        &TinyAir::new(vec![]),
        &TinyAuxBuilder,
        &[instance(0, 4), instance(1, 8), instance(2, 16)],
    );
}

// ---------------------------------------------------------------------------
// Periodic column tests
// ---------------------------------------------------------------------------

#[test]
fn single_periodic_column() {
    prove_and_verify(&TinyAir::new(vec![2]), &TinyAuxBuilder, &[instance(0, 8)]);
}

#[test]
fn periodic_column_period_4() {
    prove_and_verify(&TinyAir::new(vec![4]), &TinyAuxBuilder, &[instance(0, 8)]);
}

#[test]
fn multiple_periodic_columns() {
    prove_and_verify(
        &TinyAir::new(vec![2, 4]),
        &TinyAuxBuilder,
        &[instance(0, 8)],
    );
}

#[test]
fn periodic_columns_multi_trace_same_height() {
    prove_and_verify(
        &TinyAir::new(vec![2]),
        &TinyAuxBuilder,
        &[instance(0, 8), instance(1, 8)],
    );
}

#[test]
fn periodic_columns_multi_trace_different_heights() {
    prove_and_verify(
        &TinyAir::new(vec![2, 4]),
        &TinyAuxBuilder,
        &[instance(0, 4), instance(1, 8)],
    );
}

#[test]
fn periodic_columns_three_traces() {
    prove_and_verify(
        &TinyAir::new(vec![2, 4]),
        &TinyAuxBuilder,
        &[instance(0, 4), instance(1, 8), instance(2, 16)],
    );
}
