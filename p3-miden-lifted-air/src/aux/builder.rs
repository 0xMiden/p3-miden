//! The `AuxBuilder` trait for constructing auxiliary traces.
//!
//! This trait decouples auxiliary trace *building* from the AIR definition,
//! allowing the prover to supply a separate builder per instance.

use alloc::vec::Vec;

use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;

/// Builder for constructing the auxiliary trace from a main trace and challenges.
///
/// Decoupled from [`LiftedAir`](crate::LiftedAir) so that prover-side trace
/// construction is not part of the AIR trait. Each prover instance can supply
/// its own `AuxBuilder`.
///
/// The prover only calls [`build_aux_trace`](AuxBuilder::build_aux_trace) when
/// the AIR's `aux_width() > 0`. For AIRs without aux columns, use
/// [`EmptyAuxBuilder`] which panics if called (indicating a bug).
pub trait AuxBuilder<F: Field, EF: ExtensionField<F>> {
    /// Build the auxiliary trace and return aux values.
    ///
    /// # Arguments
    /// - `main`: The main trace matrix
    /// - `challenges`: Extension-field challenges for aux trace construction
    ///
    /// # Returns
    /// `(aux_trace, aux_values)` where:
    /// - `aux_trace`: The auxiliary trace matrix (EF columns)
    /// - `aux_values`: Extension-field scalars committed to the Fiat-Shamir transcript.
    ///   Their meaning is AIR-defined — typically the aux trace's last row, but the
    ///   protocol does not require this. The AIR's [`eval`](crate::LiftedAir::eval)
    ///   should constrain how they relate to the committed trace, and
    ///   [`reduced_aux_values`](crate::LiftedAir::reduced_aux_values) uses them for
    ///   cross-AIR bus identity checking.
    fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<F>,
        challenges: &[EF],
    ) -> (RowMajorMatrix<EF>, Vec<EF>);
}

/// Placeholder aux builder for AIRs without auxiliary columns.
///
/// Panics if [`build_aux_trace`](AuxBuilder::build_aux_trace) is called,
/// which indicates a bug: the AIR's `aux_width()` should have been 0.
pub struct EmptyAuxBuilder;

impl<F: Field, EF: ExtensionField<F>> AuxBuilder<F, EF> for EmptyAuxBuilder {
    fn build_aux_trace(
        &self,
        _main: &RowMajorMatrix<F>,
        _challenges: &[EF],
    ) -> (RowMajorMatrix<EF>, Vec<EF>) {
        panic!(
            "EmptyAuxBuilder::build_aux_trace called, but the AIR's aux_width() was not 0; \
             provide a real AuxBuilder for AIRs with auxiliary columns"
        )
    }
}
