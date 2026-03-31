//! # Lifted PCS
//!
//! A polynomial commitment scheme (PCS) combining DEEP quotient construction with FRI
//! for efficient low-degree testing over two-adic fields.
//!
//! ## Overview
//!
//! This module provides:
//!
//! - **[`deep`]**: DEEP (Domain Extension for Eliminating Pretenders) quotient construction
//!   for batching polynomial evaluation claims into a single low-degree polynomial.
//!
//! - **[`fri`]**: FRI (Fast Reed-Solomon IOP) protocol for low-degree testing, with
//!   configurable folding arities and final polynomial degree.
//!
//! - **PCS API (module root)**: complete PCS implementation combining DEEP quotient and
//!   FRI via `prover::open_with_channel` and `verifier::verify`, plus `PcsParams`.
//!
//! ## Alignment Padding
//!
//! Alignment padding is a transcript formatting convention. For trace commitments, the
//! padded columns are treated as extra polynomials and are checked for low degree by the PCS;
//! they need not be zero unless the caller enforces that. The PCS is deliberately agnostic
//! about which columns are "real" vs "padding" — enforcing zero-valued padding is the
//! caller's (or AIR's) responsibility. (FRI openings still ignore the padded tail because
//! FRI expects a fixed single-column width.)

/// DEEP quotient construction for batched polynomial evaluation.
pub mod deep;

/// FRI protocol for low-degree testing.
pub mod fri;

mod params;
mod proof;
pub(crate) mod prover;
pub(crate) mod verifier;

pub(crate) mod utils;

pub use deep::OpenedValues;
pub use params::{MAX_LOG_DOMAIN_SIZE, PcsParams, PcsParamsError};
pub use proof::PcsTranscript;
pub use verifier::PcsError;

#[cfg(test)]
mod tests;
