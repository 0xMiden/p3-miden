//! Lifted STARK shared scaffolding (LMCS-based).
//!
//! This crate contains shared types and utilities used by the lifted STARK
//! prover and verifier crates:
//!
//! - [`LiftedCoset`]: Central abstraction for domain operations (selectors, vanishing, etc.)
//! - [`StarkConfig`]: Minimal configuration wrapping PCS params, LMCS, and DFT
//! - [`Selectors`]: Constraint selectors for OOD and coset evaluation
//! - [`AirWitness`]: Prover witness (trace + public values)
//! - [`AirInstance`]: Verifier instance (log height + public values)

#![no_std]

extern crate alloc;

mod config;
mod coset;
mod instance;
mod selectors;
mod util;

pub use config::*;
pub use coset::*;
pub use instance::*;
pub use selectors::*;
pub use util::*;

// Re-export PCS parameter types from p3-miden-lifted-fri.
pub use p3_miden_lifted_fri::PcsParams;
pub use p3_miden_lifted_fri::deep::DeepParams;
pub use p3_miden_lifted_fri::fri::{FriFold, FriParams};

// Re-export LMCS types from p3-miden-lmcs.
pub use p3_miden_lmcs::{Lmcs, LmcsConfig};

// Re-export transcript data type (needed to transfer proofs between prover and verifier).
pub use p3_miden_transcript::TranscriptData;
