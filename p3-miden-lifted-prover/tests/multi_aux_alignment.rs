mod common;

use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_air::{
    AirBuilder, AirWithPeriodicColumns, AuxBuilder, BaseAir, BaseAirWithPublicValues,
    ExtensionBuilder, LiftedAir, LiftedAirBuilder,
};
use p3_miden_lifted_prover::AirWitness;
use p3_miden_lifted_verifier::{VerifierError, verify_multi};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{ProverTranscript, TranscriptData, VerifierTranscript};

use common::test_config;

#[derive(Clone, Debug)]
struct PaddingAir {
    width: usize,
    aux_width: usize,
}

impl PaddingAir {
    fn new(width: usize, aux_width: usize) -> Self {
        Self { width, aux_width }
    }
}

impl BaseAir<bb::F> for PaddingAir {
    fn width(&self) -> usize {
        self.width
    }
}

impl BaseAirWithPublicValues<bb::F> for PaddingAir {
    fn num_public_values(&self) -> usize {
        1
    }
}

impl AirWithPeriodicColumns<bb::F> for PaddingAir {
    fn periodic_columns(&self) -> &[Vec<bb::F>] {
        &[]
    }
}

impl LiftedAir<bb::F, bb::EF> for PaddingAir {
    fn aux_width(&self) -> usize {
        self.aux_width
    }

    fn num_aux_values(&self) -> usize {
        0
    }

    fn num_randomness(&self) -> usize {
        1
    }

    fn eval<AB: LiftedAirBuilder<F = bb::F>>(&self, builder: &mut AB) {
        let main = builder.main();
        let start = builder.public_values()[0];
        let (local, next) = (
            main.row_slice(0).expect("empty matrix"),
            main.row_slice(1).expect("single row matrix"),
        );

        builder.when_first_row().assert_eq(local[0].clone(), start);
        builder
            .when_transition()
            .assert_eq(next[0].clone(), local[0].clone());

        let aux = builder.permutation();
        let aux_local = aux.row_slice(0).expect("empty aux");
        let aux_next = aux.row_slice(1).expect("single row aux");
        let challenge: AB::ExprEF = builder.permutation_randomness()[0].into();
        builder
            .when_first_row()
            .assert_eq_ext(aux_local[0].into(), challenge);
        builder
            .when_transition()
            .assert_eq_ext(aux_next[0].into(), aux_local[0].into());
    }
}

impl AuxBuilder<bb::F, bb::EF> for PaddingAir {
    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<bb::F>,
        challenges: &[bb::EF],
    ) -> (RowMajorMatrix<bb::EF>, Vec<bb::EF>) {
        let height = main.height();
        let mut values = Vec::with_capacity(height * self.aux_width);
        let challenge = challenges[0];
        for _ in 0..height {
            values.push(challenge);
            values.extend(std::iter::repeat_n(bb::EF::ZERO, self.aux_width - 1));
        }
        (RowMajorMatrix::new(values, self.aux_width), vec![])
    }
}

fn generate_trace(start: bb::F, height: usize, width: usize) -> RowMajorMatrix<bb::F> {
    let mut values = Vec::with_capacity(height * width);
    for _ in 0..height {
        values.push(start);
        values.extend(std::iter::repeat_n(bb::F::ZERO, width - 1));
    }
    RowMajorMatrix::new(values, width)
}

fn instance(idx: usize, height: usize, width: usize) -> (RowMajorMatrix<bb::F>, Vec<bb::F>) {
    let start = bb::F::from_u64((idx + 2) as u64);
    (generate_trace(start, height, width), vec![start])
}

#[test]
fn multi_trace_with_aux_padding() {
    let config = test_config();
    let alignment = config.lmcs.alignment();
    let width = alignment + 1;
    let aux_width = alignment + 1;

    let air = PaddingAir::new(width, aux_width);
    let instances = [instance(0, 8, width), instance(1, 16, width)];

    let prover_instances: Vec<_> = instances
        .iter()
        .map(|(t, pv)| (&air, AirWitness::new(t, pv, &[]), &air))
        .collect();

    let mut prover_channel = ProverTranscript::new(bb::test_challenger());
    p3_miden_lifted_prover::prove_multi(&config, &prover_instances, &mut prover_channel)
        .expect("proving should succeed");
    let transcript = prover_channel.into_data();

    let verifier_instances: Vec<_> = prover_instances
        .iter()
        .map(|(a, w, _)| (*a, w.to_instance()))
        .collect();

    let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &transcript);
    verify_multi(&config, &verifier_instances, &mut verifier_channel)
        .expect("verification should succeed");
}

#[test]
fn multi_trace_rejects_trailing_transcript_data() {
    let config = test_config();
    let alignment = config.lmcs.alignment();
    let width = alignment + 1;
    let aux_width = alignment + 1;

    let air = PaddingAir::new(width, aux_width);
    let instances = [instance(0, 8, width), instance(1, 16, width)];

    let prover_instances: Vec<_> = instances
        .iter()
        .map(|(t, pv)| (&air, AirWitness::new(t, pv, &[]), &air))
        .collect();

    let mut prover_channel = ProverTranscript::new(bb::test_challenger());
    p3_miden_lifted_prover::prove_multi(&config, &prover_instances, &mut prover_channel)
        .expect("proving should succeed");
    let transcript = prover_channel.into_data();

    let (mut fields, commitments) = transcript.clone().into_parts();
    fields.push(bb::F::ONE);
    let bad_transcript = TranscriptData::new(fields, commitments);

    let verifier_instances: Vec<_> = prover_instances
        .iter()
        .map(|(a, w, _)| (*a, w.to_instance()))
        .collect();

    let mut bad_channel = VerifierTranscript::from_data(bb::test_challenger(), &bad_transcript);
    let err = verify_multi(&config, &verifier_instances, &mut bad_channel)
        .expect_err("extra transcript data should fail verification");
    assert!(matches!(err, VerifierError::TranscriptNotConsumed));
}
