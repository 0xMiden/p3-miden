mod common;

use alloc::vec::Vec;

use p3_field::PrimeCharacteristicRing;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_air::{
    AirBuilder, AuxBuilder, BaseAir, ExtensionBuilder, LiftedAir, LiftedAirBuilder, WindowAccess,
};
use p3_miden_lifted_stark::{AirWitness, VerifierError, verify_multi};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{TranscriptData, TranscriptError};

use common::test_config;

extern crate alloc;

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

    fn num_public_values(&self) -> usize {
        1
    }
}

impl LiftedAir<bb::F, bb::EF> for PaddingAir {
    fn num_randomness(&self) -> usize {
        1
    }

    fn aux_width(&self) -> usize {
        self.aux_width
    }

    fn num_aux_values(&self) -> usize {
        0
    }

    fn num_var_len_public_inputs(&self) -> usize {
        0
    }

    fn eval<AB: LiftedAirBuilder<F = bb::F>>(&self, builder: &mut AB) {
        let main = builder.main();
        let start = builder.public_values()[0];
        let (local, next) = (main.current_slice(), main.next_slice());

        builder.when_first_row().assert_eq(local[0], start);
        builder.when_transition().assert_eq(next[0], local[0]);

        let aux = builder.permutation();
        let aux_local = aux.current_slice();
        let aux_next = aux.next_slice();
        let challenge: AB::ExprEF = builder.permutation_randomness()[0].into();
        builder
            .when_first_row()
            .assert_eq_ext(aux_local[0].into(), challenge);
        builder
            .when_transition()
            .assert_eq_ext(aux_next[0].into(), aux_local[0].into());
    }
}

struct PaddingAuxBuilder {
    aux_width: usize,
}

impl AuxBuilder<bb::F, bb::EF> for PaddingAuxBuilder {
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
        let aux_trace = RowMajorMatrix::new(values, self.aux_width);
        (aux_trace, vec![])
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
    let aux_builder = PaddingAuxBuilder { aux_width };
    let instances = [instance(0, 8, width), instance(1, 16, width)];

    let prover_instances: Vec<_> = instances
        .iter()
        .map(|(t, pv)| (&air, AirWitness::new(t, pv, &[]), &aux_builder))
        .collect();

    let output =
        p3_miden_lifted_stark::prove_multi(&config, &prover_instances, bb::test_challenger())
            .expect("proving should succeed");

    let verifier_instances: Vec<_> = prover_instances
        .iter()
        .map(|(a, w, _)| (*a, w.to_instance().unwrap()))
        .collect();

    let verifier_digest = verify_multi(
        &config,
        &verifier_instances,
        &output.proof,
        bb::test_challenger(),
    )
    .expect("verification should succeed");
    assert_eq!(
        output.digest, verifier_digest,
        "prover and verifier digests must match"
    );
}

#[test]
fn multi_trace_rejects_trailing_transcript_data() {
    let config = test_config();
    let alignment = config.lmcs.alignment();
    let width = alignment + 1;
    let aux_width = alignment + 1;

    let air = PaddingAir::new(width, aux_width);
    let aux_builder = PaddingAuxBuilder { aux_width };
    let instances = [instance(0, 8, width), instance(1, 16, width)];

    let prover_instances: Vec<_> = instances
        .iter()
        .map(|(t, pv)| (&air, AirWitness::new(t, pv, &[]), &aux_builder))
        .collect();

    let output =
        p3_miden_lifted_stark::prove_multi(&config, &prover_instances, bb::test_challenger())
            .expect("proving should succeed");

    let (mut fields, commitments) = output.proof.into_parts();
    fields.push(bb::F::ONE);
    let bad_transcript = TranscriptData::new(fields, commitments);

    let verifier_instances: Vec<_> = prover_instances
        .iter()
        .map(|(a, w, _)| (*a, w.to_instance().unwrap()))
        .collect();

    let err = verify_multi(
        &config,
        &verifier_instances,
        &bad_transcript,
        bb::test_challenger(),
    )
    .expect_err("extra transcript data should fail finalize");
    assert!(matches!(
        err,
        VerifierError::Transcript(TranscriptError::TrailingData)
    ));
}
