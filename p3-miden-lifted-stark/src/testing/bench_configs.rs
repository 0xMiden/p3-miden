//! Shared benchmark configuration constructors for Goldilocks + Poseidon2.
//!
//! Provides fully-assembled configs for the lifted STARK and batch STARK
//! provers so that example binaries don't duplicate type aliases and
//! configuration boilerplate.

use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriParameters, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_symmetric::PaddingFreeSponge;

use crate::{
    config::GenericStarkConfig, pcs::params::PcsParams,
    testing::configs::goldilocks_poseidon2 as gl,
};

// ─── Common type aliases ─────────────────────────────────────────────────────

pub type Val = Goldilocks;
pub type Challenge = BinomialExtensionField<Val, 2>;
pub type Dft = Radix2DitParallel<Val>;

// ─── Batch STARK ─────────────────────────────────────────────────────────────

type MmcsSponge = PaddingFreeSponge<gl::Perm, { gl::WIDTH }, { gl::RATE }, { gl::DIGEST }>;
type ValMmcs = MerkleTreeMmcs<gl::P, gl::P, MmcsSponge, gl::Compress, 2, { gl::DIGEST }>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type BatchPcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
type BatchChallenger = DuplexChallenger<Val, gl::Perm, { gl::WIDTH }, { gl::RATE }>;

pub type BatchConfig = p3_uni_stark::StarkConfig<BatchPcs, Challenge, BatchChallenger>;

/// Build a batch-STARK config (Goldilocks + Poseidon2) with the given FRI parameters.
pub fn batch_config(log_blowup: usize, num_queries: usize, pow_bits: usize) -> BatchConfig {
    let (perm, _, compress) = gl::test_components();
    let mmcs_sponge = MmcsSponge::new(perm.clone());
    let mmcs = ValMmcs::new(mmcs_sponge, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(mmcs.clone());
    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries,
        commit_proof_of_work_bits: pow_bits,
        query_proof_of_work_bits: 0,
        mmcs: challenge_mmcs,
    };
    let pcs = BatchPcs::new(Dft::default(), mmcs, fri_params);
    let challenger = BatchChallenger::new(perm);
    BatchConfig::new(pcs, challenger)
}

// ─── Lifted STARK ────────────────────────────────────────────────────────────

pub type LiftedConfig = GenericStarkConfig<Val, Challenge, gl::Lmcs, Dft, gl::Challenger>;

/// Build a lifted-STARK config (Goldilocks + Poseidon2) with the given PCS parameters.
pub fn lifted_config(pcs: PcsParams) -> LiftedConfig {
    LiftedConfig::new(pcs, gl::test_lmcs(), Dft::default(), gl::test_challenger())
}
