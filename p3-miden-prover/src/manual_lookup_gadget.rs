//! Auxiliary Trace Configuration for manual aux trace building.
//!
//! This module provides [`AuxTraceConfig`], a configuration for auxiliary trace
//! generation that allows users to provide their own aux trace builder function.
//!
//! # Example
//!
//! ```rust,ignore
//! use p3_miden_prover::AuxTraceConfig;
//!
//! // Create a config with a custom aux trace builder
//! let aux_config = AuxTraceConfig::new(
//!     2,  // num_challenges
//!     3,  // aux_width (number of extension field columns)
//!     |main: &RowMajorMatrix<F>, challenges: &[EF]| {
//!         // Build your aux trace here
//!         build_logup_trace(main, challenges)
//!     },
//! );
//!
//! prove(&config, &air, &trace, &pis, Some(aux_config));
//! ```

use alloc::boxed::Box;

use p3_matrix::dense::RowMajorMatrix;

/// Configuration for auxiliary trace generation.
///
/// This struct holds the parameters needed for auxiliary trace building:
/// - Number of extension field challenges to sample
/// - Width of the auxiliary trace (in extension field columns)
/// - A function that builds the auxiliary trace from the main trace and challenges
///
/// # Type Parameters
///
/// - `F`: The base field type
/// - `EF`: The extension field type (for challenges and aux trace values)
pub struct AuxTraceConfig<F, EF> {
    /// Number of extension field challenges to sample.
    pub num_challenges: usize,
    /// Number of extension field columns in the auxiliary trace.
    pub aux_width: usize,
    /// Function that builds the aux trace from (main_trace, challenges).
    /// The returned matrix should have `aux_width * EF::DIMENSION` base field columns.
    aux_builder: Box<dyn Fn(&RowMajorMatrix<F>, &[EF]) -> RowMajorMatrix<F> + Send + Sync>,
}

impl<F, EF> AuxTraceConfig<F, EF> {
    /// Creates a new `AuxTraceConfig` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `num_challenges` - Number of extension field challenges to sample.
    /// * `aux_width` - Number of extension field columns in the auxiliary trace.
    /// * `aux_builder` - Function that builds the aux trace from (main_trace, challenges).
    pub fn new<B>(num_challenges: usize, aux_width: usize, aux_builder: B) -> Self
    where
        B: Fn(&RowMajorMatrix<F>, &[EF]) -> RowMajorMatrix<F> + Send + Sync + 'static,
    {
        Self {
            num_challenges,
            aux_width,
            aux_builder: Box::new(aux_builder),
        }
    }

    /// Builds the auxiliary trace using the configured builder function.
    pub fn build_aux_trace(
        &self,
        main: &RowMajorMatrix<F>,
        challenges: &[EF],
    ) -> RowMajorMatrix<F> {
        (self.aux_builder)(main, challenges)
    }

    /// Returns a verification config (without the builder) for use with the verifier.
    pub fn as_verify_config(&self) -> AuxVerifyConfig {
        AuxVerifyConfig {
            num_challenges: self.num_challenges,
            aux_width: self.aux_width,
        }
    }
}

/// Configuration for auxiliary trace verification.
///
/// This struct holds the parameters needed for auxiliary trace verification
/// (number of challenges and width). Unlike [`AuxTraceConfig`], this does not
/// include a builder function since the verifier doesn't build traces.
///
/// # Usage
///
/// You can create this directly or from an [`AuxTraceConfig`]:
///
/// ```rust,ignore
/// // Create directly
/// let verify_config = AuxVerifyConfig::new(2, 3);
///
/// // Or from an AuxTraceConfig
/// let verify_config = aux_config.as_verify_config();
///
/// verify(&config, &air, &proof, &pis, Some(&verify_config));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AuxVerifyConfig {
    /// Number of extension field challenges to sample.
    pub num_challenges: usize,
    /// Number of extension field columns in the auxiliary trace.
    pub aux_width: usize,
}

impl AuxVerifyConfig {
    /// Creates a new `AuxVerifyConfig` with the given parameters.
    pub const fn new(num_challenges: usize, aux_width: usize) -> Self {
        Self {
            num_challenges,
            aux_width,
        }
    }
}
