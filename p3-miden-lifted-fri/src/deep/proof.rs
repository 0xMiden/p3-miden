//! DEEP transcript data structures.

use crate::deep::DeepEvals;
use crate::deep::DeepParams;
use alloc::vec::Vec;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_miden_transcript::{Channel, TranscriptError, VerifierChannel};

/// Structured transcript view for the DEEP interaction.
///
/// This records the prover's PoW witness and the two challenges sampled
/// from the Fiat-Shamir transcript after observing evaluations.
pub struct DeepTranscript<F: Field, EF: ExtensionField<F>> {
    /// `evals.point(idx).as_slice()` gives all column values for a single point.
    pub evals: DeepEvals<EF>,
    /// Proof-of-work witness sampled before DEEP challenges.
    pub pow_witness: F,
    /// Challenge `α` for batching columns into `f_reduced`.
    pub challenge_columns: EF,
    /// Challenge `β` for batching opening points.
    pub challenge_points: EF,
}

impl<F, EF> DeepTranscript<F, EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    /// Parse DEEP transcript data from a verifier channel.
    ///
    /// Reads OOD evaluations, verifies the PoW witness, and samples batching
    /// challenges. Does not verify the DEEP quotient itself; that validation
    /// happens in the DEEP verifier. Commitment widths must match the
    /// committed rows (including any alignment padding).
    pub fn from_verifier_channel<Ch>(
        params: &DeepParams,
        commitments: &[(<Ch as Channel>::Commitment, Vec<usize>)],
        num_eval_points: usize,
        channel: &mut Ch,
    ) -> Result<Self, TranscriptError>
    where
        Ch: VerifierChannel<F = F>,
    {
        let widths: Vec<&[usize]> = commitments
            .iter()
            .map(|(_, widths)| widths.as_slice())
            .collect();
        let evals = DeepEvals::read_from_channel(&widths, num_eval_points, channel)?;

        let pow_witness = channel.grind(params.deep_pow_bits)?;
        let challenge_columns: EF = channel.sample_algebra_element();
        let challenge_points: EF = channel.sample_algebra_element();

        Ok(Self {
            evals,
            pow_witness,
            challenge_columns,
            challenge_points,
        })
    }
}
