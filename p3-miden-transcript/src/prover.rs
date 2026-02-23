//! Prover-side transcript channel.

use alloc::vec::Vec;

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, GrindingChallenger,
};
use p3_field::{BasedVectorSpace, Field};

use crate::TranscriptData;

/// Prover channel that records transcript data and observes into the challenger.
#[derive(Clone, Debug)]
pub struct ProverTranscript<F, C, Ch> {
    challenger: Ch,
    fields: Vec<F>,
    commitments: Vec<C>,
}

impl<F, C, Ch> ProverTranscript<F, C, Ch> {
    /// Creates a new prover transcript backed by the provided challenger.
    pub fn new(challenger: Ch) -> Self {
        Self {
            challenger,
            fields: Vec::new(),
            commitments: Vec::new(),
        }
    }

    /// Consume the prover transcript and return the raw transcript data.
    pub fn into_data(self) -> TranscriptData<F, C> {
        TranscriptData::new(self.fields, self.commitments)
    }

    /// Returns a snapshot of the recorded transcript data.
    pub fn data(&self) -> TranscriptData<F, C>
    where
        F: Clone,
        C: Clone,
    {
        TranscriptData::new(self.fields.clone(), self.commitments.clone())
    }
}

/// Prover-side channel interface for transcript operations.
pub trait ProverChannel {
    type F: Field;
    type Commitment: Copy;

    fn send_field_slice(&mut self, values: &[Self::F]);

    fn send_commitment_slice(&mut self, values: &[Self::Commitment]);

    fn send_field_element(&mut self, value: Self::F) {
        self.send_field_slice(core::slice::from_ref(&value));
    }

    fn send_algebra_element<A>(&mut self, value: A)
    where
        A: BasedVectorSpace<Self::F>,
    {
        self.send_field_slice(value.as_basis_coefficients_slice());
    }

    fn send_algebra_slice<A>(&mut self, values: &[A])
    where
        A: BasedVectorSpace<Self::F>,
    {
        for value in values {
            self.send_field_slice(value.as_basis_coefficients_slice());
        }
    }

    fn send_commitment(&mut self, value: Self::Commitment) {
        self.send_commitment_slice(core::slice::from_ref(&value));
    }

    fn hint_field_slice(&mut self, values: &[Self::F]);

    fn hint_commitment_slice(&mut self, values: &[Self::Commitment]);

    fn hint_field_element(&mut self, value: Self::F) {
        self.hint_field_slice(core::slice::from_ref(&value));
    }

    fn hint_commitment(&mut self, value: Self::Commitment) {
        self.hint_commitment_slice(core::slice::from_ref(&value));
    }

    fn grind(&mut self, bits: usize) -> Self::F;

    fn sample_algebra_element<A: BasedVectorSpace<Self::F>>(&mut self) -> A
    where
        Self: CanSample<Self::F>,
    {
        A::from_basis_coefficients_fn(|_| self.sample())
    }
}

impl<F, C, Ch> ProverChannel for ProverTranscript<F, C, Ch>
where
    F: Field,
    C: Copy,
    Ch: CanObserve<F> + CanObserve<C> + GrindingChallenger<Witness = F>,
{
    type F = F;
    type Commitment = C;

    fn send_field_slice(&mut self, values: &[Self::F]) {
        self.fields.extend_from_slice(values);
        self.challenger.observe_slice(values);
    }

    fn send_commitment_slice(&mut self, values: &[Self::Commitment]) {
        self.commitments.extend_from_slice(values);
        self.challenger.observe_slice(values);
    }

    fn hint_field_slice(&mut self, values: &[Self::F]) {
        self.fields.extend_from_slice(values);
    }

    fn hint_commitment_slice(&mut self, values: &[Self::Commitment]) {
        self.commitments.extend_from_slice(values);
    }

    fn grind(&mut self, bits: usize) -> Self::F {
        let witness = self.challenger.grind(bits);
        self.fields.push(witness);
        witness
    }
}

impl<F, C, Ch, T> CanSample<T> for ProverTranscript<F, C, Ch>
where
    Ch: CanSample<T>,
{
    #[inline]
    fn sample(&mut self) -> T {
        self.challenger.sample()
    }
}

impl<F, C, Ch> CanSampleBits<usize> for ProverTranscript<F, C, Ch>
where
    Ch: CanSampleBits<usize>,
{
    #[inline]
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.challenger.sample_bits(bits)
    }
}

impl<F, C, Ch> CanSampleUniformBits<F> for ProverTranscript<F, C, Ch>
where
    Ch: CanSampleUniformBits<F>,
{
    #[inline]
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, p3_challenger::ResamplingError> {
        self.challenger.sample_uniform_bits::<RESAMPLE>(bits)
    }
}
