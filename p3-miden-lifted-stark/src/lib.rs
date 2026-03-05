//! Lifted STARK prover and verifier (LMCS-based).
//!
//! This crate contains the full lifted STARK implementation:
//!
//! - [`LiftedCoset`]: Central abstraction for domain operations (selectors, vanishing, etc.)
//! - [`StarkConfig`]: Minimal configuration wrapping PCS params, LMCS, and DFT
//! - [`Selectors`]: Constraint selectors for OOD and coset evaluation
//! - [`AirWitness`]: Prover witness (trace + public values)
//! - [`AirInstance`]: Verifier instance (log height + public values)
//! - [`prove_single`] / [`prove_multi`]: Prover entry points
//! - [`verify_single`] / [`verify_multi`]: Verifier entry points
//! - [`StarkTranscript`]: Structured transcript view for the full protocol
//!
//! # AIR Trust Model
//!
//! The lifted STARK has three trust domains:
//!
//! 1. **AIR = trusted** — [`LiftedAir`] implementations are
//!    correct application code. It is the AIR implementer's responsibility to satisfy the
//!    contract below. [`LiftedAir::validate`]
//!    checks the statically-verifiable subset.
//!
//! 2. **Instance = validated** — The prover validates that its witness matches the AIR spec
//!    The verifier validates instance metadata.
//!    Both return structured errors.
//!
//! 3. **Proof = untrusted** — Transcript data is verified cryptographically (PCS errors,
//!    constraint mismatch, etc.).
//!
//! ## Validated properties
//!
//! These are checked by [`LiftedAir::validate`]
//! and [`AirInstance::validate`](p3_miden_lifted_air::AirInstance::validate), and enforced
//! by both prover and verifier before proceeding:
//!
//! - **No preprocessed trace** — the lifted protocol does not support them.
//! - **Positive aux width** — every AIR must have an auxiliary trace.
//! - **Periodic columns** — each has positive, power-of-two length ≤ trace height.
//! - **Constraint degree** — `log_quotient_degree() ≤ log_blowup`.
//! - **Instance dimensions** — trace width, public values length, var-len public
//!   inputs count, and trace height (power of two) all match the AIR specification.
//!
//! ## Unchecked trust assumptions
//!
//! These cannot be verified statically and are the AIR implementer's responsibility:
//!
//! 1. **Window size** — Only transition window size 2.
//! 2. **Deterministic constraints** — `eval()` emits the same number and types of
//!    constraints regardless of builder implementation.
//! 3. **Consistent aux builder** — `AuxBuilder::build_aux_trace` returns
//!    width = `aux_width()`, height = main trace height, and exactly
//!    `num_aux_values()` values. (The prover asserts these at runtime as a
//!    defense-in-depth sanity check.)
//! 4. **Sound `reduced_aux_values`** — Returns correct bus contributions for valid inputs.

#![no_std]

extern crate alloc;

mod config;
mod coset;
mod selectors;

// Prover and verifier modules
pub mod prover;
pub mod verifier;

pub use config::*;
pub use coset::*;
pub use p3_miden_lifted_air::{AirInstance, AirWitness};
pub use selectors::*;

// Re-export PCS parameter types from p3-miden-lifted-fri.
pub use p3_miden_lifted_fri::PcsParams;
pub use p3_miden_lifted_fri::deep::DeepParams;
pub use p3_miden_lifted_fri::fri::{FriFold, FriParams};

// Re-export LMCS types from p3-miden-lmcs.
pub use p3_miden_lmcs::{Lmcs, LmcsConfig};

// Re-export transcript data type (needed to transfer proofs between prover and verifier).
pub use p3_miden_transcript::TranscriptData;

// Prover flat re-exports
pub use prover::{ProverError, prove_multi, prove_single};

// Verifier flat re-exports
pub use verifier::proof::StarkTranscript;
pub use verifier::{VerifierError, verify_multi, verify_single};

// Re-exports from dependencies (previously via prover/verifier crate lib.rs)
pub use p3_miden_lifted_air::{
    AirValidationError, AuxBuilder, BuilderMismatchError, LiftedAir, ReducedAuxValues,
    ReductionError, VarLenPublicInputs,
};
pub use p3_miden_lifted_fri::PcsTranscript;
pub use p3_miden_transcript::{
    ProverChannel, ProverTranscript, VerifierChannel, VerifierTranscript,
};
