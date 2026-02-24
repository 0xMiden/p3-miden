//! Transcript data container for external transport.

use alloc::vec::Vec;

use serde::Serialize;

/// Raw transcript data captured by a prover and replayed by a verifier.
#[derive(Clone, Debug, Serialize)]
#[serde(bound(serialize = "F: Serialize, C: Serialize"))]
pub struct TranscriptData<F, C> {
    fields: Vec<F>,
    commitments: Vec<C>,
}

impl<F, C> TranscriptData<F, C> {
    /// Create transcript data from field and commitment streams.
    pub fn new(fields: Vec<F>, commitments: Vec<C>) -> Self {
        Self {
            fields,
            commitments,
        }
    }

    /// Returns the recorded field elements.
    pub fn fields(&self) -> &[F] {
        &self.fields
    }

    /// Returns the recorded commitments.
    pub fn commitments(&self) -> &[C] {
        &self.commitments
    }

    /// Returns field and commitment slices for verifier construction.
    pub fn as_slices(&self) -> (&[F], &[C]) {
        (&self.fields, &self.commitments)
    }

    /// Consume and return the underlying vectors.
    pub fn into_parts(self) -> (Vec<F>, Vec<C>) {
        (self.fields, self.commitments)
    }

    /// Returns the in-memory size of this transcript in bytes.
    ///
    /// This is computed from the element counts and their `size_of::<T>()`,
    /// and does not require serializing the transcript.
    pub fn size_in_bytes(&self) -> usize {
        core::mem::size_of::<F>() * self.fields.len()
            + core::mem::size_of::<C>() * self.commitments.len()
    }
}
