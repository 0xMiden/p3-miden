#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod commit;
pub mod constraints;
mod periodic;
mod prover;
pub mod quotient;

// Re-exports from dependencies
pub use p3_miden_lifted_air::{AuxBuilder, EmptyAuxBuilder};
pub use p3_miden_lifted_stark::{
    AirWitness, DeepParams, FriFold, FriParams, GenericStarkConfig, Lmcs, LmcsConfig, PcsParams,
    StarkConfig, TranscriptData,
};
pub use p3_miden_lifted_verifier::StarkTranscript;
pub use p3_miden_transcript::{ProverChannel, ProverTranscript};

// Public API
pub use prover::{ProverError, prove_multi, prove_single};
