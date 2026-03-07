#![allow(dead_code)]

use p3_matrix::dense::RowMajorMatrix;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_lifted_air::{AuxBuilder, LiftedAir};
use p3_miden_lifted_fri::PcsParams;
use p3_miden_lifted_fri::deep::DeepParams;
use p3_miden_lifted_fri::fri::{FriFold, FriParams};
use p3_miden_lifted_stark::{AirWitness, GenericStarkConfig, verify_multi};
use p3_miden_lmcs::LmcsConfig;

pub type TestLmcs =
    LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;
pub type TestDft = p3_dft::Radix2DitParallel<bb::F>;
pub type TestConfig = GenericStarkConfig<bb::F, bb::EF, TestLmcs, TestDft, bb::Challenger>;

pub fn test_config() -> TestConfig {
    let pcs = PcsParams {
        fri: FriParams {
            log_blowup: 2,
            fold: FriFold::ARITY_2,
            log_final_degree: 2,
            folding_pow_bits: 0,
        },
        deep: DeepParams { deep_pow_bits: 0 },
        num_queries: 2,
        query_pow_bits: 0,
    };

    let (_, sponge, compress) = bb::test_components();
    let lmcs: TestLmcs = LmcsConfig::new(sponge, compress);
    let dft = TestDft::default();

    GenericStarkConfig::new(pcs, lmcs, dft, bb::test_challenger())
}

/// Prove and verify multiple traces, each with its own public values.
///
/// `instances` is a slice of `(trace, public_values)` pairs in ascending height order.
pub fn prove_and_verify<A, B>(
    air: &A,
    aux_builder: &B,
    instances: &[(RowMajorMatrix<bb::F>, Vec<bb::F>)],
) where
    A: LiftedAir<bb::F, bb::EF>,
    B: AuxBuilder<bb::F, bb::EF>,
{
    let config = test_config();

    let prover_instances: Vec<_> = instances
        .iter()
        .map(|(t, pv)| (air, AirWitness::new(t, pv, &[]), aux_builder))
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
