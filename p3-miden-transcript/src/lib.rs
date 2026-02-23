//! Transcript channels for Fiat-Shamir protocols with raw field/commitment storage.
//!
//! This crate provides:
//! - [`ProverTranscript`] and [`ProverChannel`] for prover-side recording.
//! - [`VerifierTranscript`] and [`VerifierChannel`] for verifier-side reading.

#![no_std]

extern crate alloc;

mod data;
mod prover;
mod verifier;

// Public API re-exports.
pub use data::TranscriptData;
pub use prover::{ProverChannel, ProverTranscript};
pub use verifier::{TranscriptError, VerifierChannel, VerifierTranscript};
