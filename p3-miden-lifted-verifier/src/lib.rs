#![doc = include_str!("../README.md")]
#![no_std]

extern crate alloc;

mod constraints;
mod periodic;
mod proof;
mod verifier;

pub use p3_miden_lifted_air::{LiftedAir, ReducedAuxValues, ReductionError, VarLenPublicInputs};
pub use p3_miden_lifted_fri::PcsTranscript;
pub use p3_miden_lifted_stark::{
    AirInstance, DeepParams, FriFold, FriParams, GenericStarkConfig, Lmcs, LmcsConfig, PcsParams,
    StarkConfig, TranscriptData,
};
pub use p3_miden_transcript::{VerifierChannel, VerifierTranscript};
pub use proof::StarkTranscript;
pub use verifier::{VerifierError, verify_multi, verify_single};
