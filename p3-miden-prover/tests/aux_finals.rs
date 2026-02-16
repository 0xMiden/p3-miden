use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_miden_air::{
    AuxFinalsError, MidenAir, MidenAirBuilder, VarLenPublicInputs,
    reduce_multiset_bus_boundary_varlen,
};
use p3_miden_fri::{TwoAdicFriPcs, create_test_fri_params};
use p3_miden_prover::{StarkConfig, VerificationError, prove, verify};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::SeedableRng;
use rand::rngs::SmallRng;

type Val = Goldilocks;
type Perm = Poseidon2Goldilocks<16>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type Challenge = BinomialExtensionField<Val, 2>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type Dft = Radix2DitParallel<Val>;
type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

struct AuxBoundaryAir;

impl AuxBoundaryAir {
    fn new() -> Self {
        Self
    }
}

impl<F: Field, EF: ExtensionField<F>> MidenAir<F, EF> for AuxBoundaryAir {
    fn width(&self) -> usize {
        1
    }

    fn aux_width(&self) -> usize {
        1
    }

    fn num_randomness(&self) -> usize {
        2
    }

    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<F>,
        challenges: &[EF],
    ) -> Option<RowMajorMatrix<F>> {
        let expected =
            reduce_multiset_bus_boundary_varlen::<F, EF, _>(challenges, main.row_slices());

        let mut values = vec![F::ZERO; main.height() * EF::DIMENSION];
        let offset = (main.height() - 1) * EF::DIMENSION;
        values[offset..offset + EF::DIMENSION]
            .copy_from_slice(expected.as_basis_coefficients_slice());
        Some(RowMajorMatrix::new(values, EF::DIMENSION))
    }

    fn verify_aux_finals(
        &self,
        randomness: &[EF],
        aux_finals: &[EF],
        _public_values: &[F],
        var_len_public_inputs: VarLenPublicInputs<'_, F>,
    ) -> Result<(), AuxFinalsError> {
        let bus_inputs = var_len_public_inputs
            .first()
            .ok_or(AuxFinalsError::MissingBusPublicInputs { bus_index: 0 })?;
        let expected =
            reduce_multiset_bus_boundary_varlen::<F, EF, _>(randomness, bus_inputs.iter().copied());
        let aux_final = aux_finals
            .first()
            .ok_or(AuxFinalsError::InvalidAuxFinalsLength {
                expected: 1,
                got: aux_finals.len(),
            })?;

        if *aux_final == expected {
            Ok(())
        } else {
            Err(AuxFinalsError::InvalidBoundary { bus_index: 0 })
        }
    }

    fn eval<AB: MidenAirBuilder<F = F>>(&self, _builder: &mut AB) {}
}

fn generate_trace(values: &[u64]) -> RowMajorMatrix<Val> {
    assert!(values.len().is_power_of_two());
    let rows = values.iter().map(|v| Val::from_u64(*v)).collect::<Vec<_>>();
    RowMajorMatrix::new(rows, 1)
}

fn setup_config() -> MyConfig {
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let val_mmcs = ValMmcs::new(hash, compress);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let dft = Dft::default();
    let fri_params = create_test_fri_params(challenge_mmcs, 1);
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);
    MyConfig::new(pcs, challenger)
}

#[test]
fn test_verify_aux_finals_ok() {
    let trace = generate_trace(&[1, 2, 3, 4]);
    let bus_rows: Vec<&[Val]> = trace.row_slices().collect();
    let var_len_pis = vec![bus_rows.as_slice()];

    let config = setup_config();
    let air = AuxBoundaryAir::new();

    let proof = prove(&config, &air, &trace, &[]);
    verify(&config, &air, &proof, &[], &var_len_pis).expect("verification failed");
}

#[test]
fn test_verify_aux_finals_failure() {
    let trace = generate_trace(&[1, 2, 3, 4]);
    let bus_rows: Vec<&[Val]> = trace.row_slices().collect();
    let mut wrong_rows = bus_rows.clone();
    let wrong_row = [Val::from_u64(999)];
    wrong_rows[0] = &wrong_row;
    let wrong_var_len_pis = vec![wrong_rows.as_slice()];

    let config = setup_config();
    let air = AuxBoundaryAir::new();

    let proof = prove(&config, &air, &trace, &[]);
    let err = verify(&config, &air, &proof, &[], &wrong_var_len_pis)
        .expect_err("expected verification to fail");
    assert!(matches!(err, VerificationError::InvalidAuxFinals(_)));
}
