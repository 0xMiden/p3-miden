//! Auxiliary trace types: builder and cross-AIR identity checking.
//!
//! # Protocol Overview
//!
//! The auxiliary trace enables cross-AIR buses (multiset / logup) in the lifted STARK.
//!
//! ## Prover
//!
//! 1. [`AuxBuilder::build_aux_trace`] constructs the aux trace and returns
//!    aux values (extension-field scalars whose meaning is AIR-defined).
//! 2. The aux trace is committed (Merkle commitment).
//! 3. Aux values are sent via the Fiat-Shamir transcript.
//!
//! ## AIR constraints ([`eval`](crate::LiftedAir::eval))
//!
//! 4. The AIR defines how aux values relate to the committed aux trace.
//!    A common pattern is to constrain them to equal the aux trace's last row,
//!    but the protocol does not impose this — the AIR is free to define
//!    whatever relationship it needs.
//! 5. Transition constraints enforce the aux trace's internal logic
//!    (e.g. running product accumulation).
//!
//! ## Verifier
//!
//! 6. The verifier receives aux values from the transcript.
//! 7. Constraint evaluation (steps 4–5) is checked at a random point.
//! 8. [`reduced_aux_values`](crate::LiftedAir::reduced_aux_values) computes each
//!    AIR's bus contribution from the aux values, challenges, and public inputs.
//! 9. Global check: all contributions combine to identity (prod=1, sum=0).

mod builder;
mod values;

pub use builder::{AuxBuilder, EmptyAuxBuilder};
pub use values::ReducedAuxValues;

/// Per-bus public inputs for an AIR instance.
///
/// `var_len_public_inputs[bus_idx][row_idx]` is one row of base-field elements
/// for the given bus. Passed to [`LiftedAir::reduced_aux_values`](crate::LiftedAir::reduced_aux_values)
/// so the verifier can independently compute expected bus values.
pub type VarLenPublicInputs<'a, F> = &'a [&'a [&'a [F]]];
