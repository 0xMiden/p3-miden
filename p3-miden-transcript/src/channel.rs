//! Shared channel abstraction over a challenger and transcript data.

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, GrindingChallenger,
};
use p3_field::{BasedVectorSpace, Field};

/// Shared channel abstraction for Fiat–Shamir transcript operations,
/// providing a unified interface for observing and sampling from the challenger.
pub trait Channel {
    /// Base field over which the transcript operates.
    type F: Field;

    /// Commitment type carried by the transcript.
    type Commitment: Copy;

    /// Underlying challenger type providing Fiat–Shamir randomness.
    type Challenger: CanObserve<Self::F>
        + CanObserve<Self::Commitment>
        + CanSample<Self::F>
        + CanSampleBits<usize>
        + CanSampleUniformBits<Self::F>
        + GrindingChallenger<Witness = Self::F>;

    /// Mutable access to the underlying challenger.
    fn challenger(&mut self) -> &mut Self::Challenger;

    /// Observe public field elements into the challenger.
    fn observe_public_fields(&mut self, values: &[Self::F]) {
        self.challenger().observe_slice(values);
    }

    /// Observe public commitments into the challenger.
    fn observe_public_commitments(&mut self, values: &[Self::Commitment]) {
        self.challenger().observe_slice(values);
    }

    /// Sample a base field element from the underlying challenger.
    fn sample(&mut self) -> Self::F {
        self.challenger().sample()
    }

    /// Sample an integer from a given number of bits.
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.challenger().sample_bits(bits)
    }

    /// Sample uniformly from a given number of bits with optional resampling.
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, p3_challenger::ResamplingError> {
        self.challenger().sample_uniform_bits::<RESAMPLE>(bits)
    }

    /// Sample an extension field element by sampling base field coefficients.
    fn sample_algebra_element<A: BasedVectorSpace<Self::F>>(&mut self) -> A {
        A::from_basis_coefficients_fn(|_| self.sample())
    }
}
