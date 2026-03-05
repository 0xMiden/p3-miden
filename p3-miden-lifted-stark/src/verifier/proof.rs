//! Structured STARK transcript.
//!
//! [`StarkTranscript`] captures the full lifted STARK protocol interaction
//! (commitments, challenges, OOD point, and PCS sub-transcript) as a typed struct
//! with a [`from_verifier_channel`](StarkTranscript::from_verifier_channel) constructor
//! that parses it from a channel.
//!
//! This is a parse-only view that exists alongside [`verify_multi`](crate::verify_multi),
//! following the same pattern as [`PcsTranscript`] alongside
//! [`verify`](p3_miden_lifted_fri::verifier::verify).

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_miden_lifted_air::LiftedAir;
use p3_miden_lifted_fri::PcsTranscript;
use p3_miden_lmcs::Lmcs;
use p3_miden_lmcs::utils::aligned_len;
use p3_miden_transcript::VerifierChannel;

use crate::{AirInstance, LiftedCoset, StarkConfig};

use super::VerifierError;

/// Structured transcript view for the full lifted STARK protocol.
///
/// Captures all commitments, sampled challenges, the OOD evaluation point, and
/// the PCS sub-transcript (DEEP evals, FRI rounds, query openings).
///
/// Constructed via [`from_verifier_channel`](Self::from_verifier_channel), which
/// mirrors steps 1–9 of [`verify_multi`](crate::verify_multi) (parse only, no
/// constraint checks).
pub struct StarkTranscript<EF, L>
where
    L: Lmcs,
    L::F: Field,
    EF: ExtensionField<L::F>,
{
    /// Main trace commitment.
    pub main_commit: L::Commitment,
    /// Randomness sampled for auxiliary traces.
    pub randomness: Vec<EF>,
    /// Auxiliary trace commitment.
    pub aux_commit: L::Commitment,
    /// Aux values per AIR instance, observed into the transcript after the aux commitment.
    pub all_aux_values: Vec<Vec<EF>>,
    /// Constraint folding challenge alpha.
    pub alpha: EF,
    /// AIR accumulation challenge beta.
    pub beta: EF,
    /// Quotient polynomial commitment.
    pub quotient_commit: L::Commitment,
    /// Out-of-domain evaluation point z.
    pub z: EF,
    /// PCS sub-transcript (DEEP evals, FRI rounds, query openings).
    pub pcs_transcript: PcsTranscript<EF, L>,
}

impl<EF, L> StarkTranscript<EF, L>
where
    L: Lmcs,
    L::F: TwoAdicField,
    EF: ExtensionField<L::F>,
{
    /// Parse a STARK transcript from a verifier channel without constraint checks.
    ///
    /// Mirrors steps 1–9 of [`verify_multi`](crate::verify_multi):
    /// 1. Receive main trace commitment
    /// 2. Sample randomness for auxiliary traces
    /// 3. Receive auxiliary trace commitment
    /// 4. Receive aux values (per AIR instance)
    /// 5. Sample constraint folding alpha and accumulation beta
    /// 6. Receive quotient commitment
    /// 7. Sample OOD point z
    /// 8. Build commitment widths for PCS
    /// 9. Parse PCS sub-transcript via [`PcsTranscript::from_verifier_channel`]
    ///
    /// Does **not** verify constraints or check the quotient identity.
    pub fn from_verifier_channel<A, SC, Ch>(
        config: &SC,
        instances: &[(&A, AirInstance<'_, L::F>)],
        channel: &mut Ch,
    ) -> Result<Self, VerifierError>
    where
        A: LiftedAir<L::F, EF>,
        SC: StarkConfig<L::F, EF, Lmcs = L>,
        Ch: VerifierChannel<F = L::F, Commitment = L::Commitment>,
    {
        let log_blowup = config.pcs().fri.log_blowup;
        let alignment = config.lmcs().alignment();

        // Infer constraint degree from symbolic AIR analysis (max across all AIRs)
        let constraint_degree = instances
            .iter()
            .map(|(air, _)| air.constraint_degree())
            .max()
            .unwrap_or(2);

        // Max trace height determines the LDE domain
        let log_max_trace_height = instances.last().unwrap().1.log_trace_height;
        let log_lde_height = log_max_trace_height + log_blowup;

        // Max LDE coset (for the largest trace, no lifting)
        let max_lde_coset = LiftedCoset::unlifted(log_max_trace_height, log_blowup);

        // 1. Receive main trace commitment
        let main_commit = channel.receive_commitment()?.clone();

        // 2. Sample randomness for aux traces
        let max_num_randomness = instances
            .iter()
            .map(|(air, _)| air.num_randomness())
            .max()
            .unwrap_or(0);

        let randomness: Vec<EF> = (0..max_num_randomness)
            .map(|_| channel.sample_algebra_element::<EF>())
            .collect();

        // 3. Receive aux trace commitment
        let aux_commit = channel.receive_commitment()?.clone();

        // 4. Receive aux values from the transcript (one EF element per aux value, per instance).
        let all_aux_values: Vec<Vec<EF>> = instances
            .iter()
            .map(|(air, _)| {
                let count = air.num_aux_values();
                (0..count)
                    .map(|_| channel.receive_algebra_element::<EF>())
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;

        // 5. Sample constraint folding alpha and accumulation beta
        let alpha: EF = channel.sample_algebra_element::<EF>();
        let beta: EF = channel.sample_algebra_element::<EF>();

        // 6. Receive quotient commitment
        let quotient_commit = channel.receive_commitment()?.clone();

        // 7. Sample OOD point (outside max trace domain H and max LDE coset gK)
        let z: EF = max_lde_coset.sample_ood_point(channel);
        let h = L::F::two_adic_generator(log_max_trace_height);
        let z_next = z * h;

        // 8. Build commitment widths for PCS.
        //
        // The LMCS commits to rows padded to `alignment` boundary, so DEEP evals and
        // batch openings are stored at aligned widths in the transcript. We must use
        // aligned widths here to parse the transcript correctly.
        // (The verifier's `verify_aligned` does the same alignment internally, then
        // truncates the returned evals back to original widths for constraint checking.)
        let main_widths: Vec<usize> = instances
            .iter()
            .map(|(air, _)| aligned_len(air.width(), alignment))
            .collect();
        let quotient_width = aligned_len(constraint_degree * EF::DIMENSION, alignment);

        let aux_widths: Vec<usize> = instances
            .iter()
            .map(|(air, _)| aligned_len(air.aux_width() * EF::DIMENSION, alignment))
            .collect();

        let commitments = vec![
            (main_commit.clone(), main_widths),
            (aux_commit.clone(), aux_widths),
            (quotient_commit.clone(), vec![quotient_width]),
        ];

        // 9. Parse PCS sub-transcript
        let pcs_transcript = PcsTranscript::from_verifier_channel::<Ch, 2>(
            config.pcs(),
            config.lmcs(),
            &commitments,
            log_lde_height,
            [z, z_next],
            channel,
        )?;

        if !channel.is_empty() {
            return Err(VerifierError::TranscriptNotConsumed);
        }

        Ok(Self {
            main_commit,
            randomness,
            aux_commit,
            all_aux_values,
            alpha,
            beta,
            quotient_commit,
            z,
            pcs_transcript,
        })
    }
}
