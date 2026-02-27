//! Shared instance types for the lifted STARK prover and verifier.
//!
//! - [`AirWitness`]: Prover witness — trace + public values
//! - [`AirInstance`]: Verifier instance — log trace height + public values
//! - [`validate_instances`]: Shared validation for a slice of instances

use p3_field::Field;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use thiserror::Error;

use crate::VarLenPublicInputs;

/// Prover witness: trace matrix, public values, and variable-length public inputs.
///
/// Validates on construction that the trace height is a power of two.
///
/// **Commitment:** callers **must** bind both `public_values` and
/// `var_len_public_inputs` to the Fiat-Shamir challenger state before proving.
pub struct AirWitness<'a, F> {
    /// Main trace matrix.
    pub trace: &'a RowMajorMatrix<F>,
    /// Public values for this AIR.
    pub public_values: &'a [F],
    /// Variable-length public inputs (reducible inputs for bus identity checks).
    pub var_len_public_inputs: VarLenPublicInputs<'a, F>,
}

impl<'a, F> AirWitness<'a, F> {
    /// Create a new prover witness with validation.
    ///
    /// # Panics
    ///
    /// - If `trace.height()` is not a power of two
    pub fn new(
        trace: &'a RowMajorMatrix<F>,
        public_values: &'a [F],
        var_len_public_inputs: VarLenPublicInputs<'a, F>,
    ) -> Self
    where
        F: Field,
    {
        assert!(
            trace.height().is_power_of_two(),
            "trace height must be power of two, got {}",
            trace.height()
        );
        Self {
            trace,
            public_values,
            var_len_public_inputs,
        }
    }

    /// Convert to a verifier instance (drops the trace, keeps log height).
    pub fn to_instance(&self) -> AirInstance<'a, F>
    where
        F: Clone + Send + Sync,
    {
        AirInstance {
            log_trace_height: log2_strict_usize(self.trace.height()),
            public_values: self.public_values,
            var_len_public_inputs: self.var_len_public_inputs,
        }
    }
}

/// Verifier instance: log trace height, public values, and variable-length inputs.
///
/// Both the prover and verifier carry `var_len_public_inputs`. The verifier uses
/// them in [`LiftedAir::reduced_aux_values`](crate::LiftedAir::reduced_aux_values)
/// for the cross-AIR identity check.
#[derive(Clone, Copy)]
pub struct AirInstance<'a, F> {
    /// Log₂ of the trace height.
    pub log_trace_height: usize,
    /// Public values for this AIR.
    pub public_values: &'a [F],
    /// Reducible inputs for the cross-AIR identity check. Empty slice if no buses.
    pub var_len_public_inputs: VarLenPublicInputs<'a, F>,
}

/// Errors from instance validation.
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("no instances provided")]
    Empty,
    #[error("instances not in ascending height order")]
    NotAscending,
    #[error("trace width mismatch")]
    WidthMismatch,
}

/// Validate that instances are non-empty and sorted by ascending log height.
pub fn validate_instances<F>(instances: &[AirInstance<'_, F>]) -> Result<(), ValidationError> {
    if instances.is_empty() {
        return Err(ValidationError::Empty);
    }

    let mut prev = 0;
    for inst in instances {
        if inst.log_trace_height < prev {
            return Err(ValidationError::NotAscending);
        }
        prev = inst.log_trace_height;
    }

    Ok(())
}
