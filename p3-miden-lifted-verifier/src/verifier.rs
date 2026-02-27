//! Lifted STARK verifier.
//!
//! This module provides:
//! - [`verify_single`]: Verify a single AIR instance.
//! - [`verify_multi`]: Verify multiple AIR instances with traces of different heights.
//!
//! These functions read the proof *from* a [`p3_miden_transcript::VerifierChannel`]
//! and sample Fiat-Shamir challenges from the challenger state backing the channel.
//!
//! # Fiat-Shamir / transcript binding (initial challenger state)
//!
//! This crate intentionally does **not** prescribe the *initial* transcript state.
//! In particular, the verifier APIs here take the statement out-of-band (AIR(s),
//! instance metadata like trace heights, and `public_values`).
//!
//! The protocol implementation assumes that all inputs that may vary (including
//! `public_values`) have already been observed by the challenger inside `channel`.
//! This is required so callers can avoid including large public inputs in the proof
//! when they are available out-of-band.
//!
//! If your application treats any of these inputs as untrusted, you must
//! authenticate them by binding them into the Fiat-Shamir challenger state
//! (domain separation + an AIR/version tag + public inputs), in the same way on
//! both prover and verifier.
//!
//! The module-level docs in `p3-miden-lifted-prover` show the recommended ergonomic
//! pattern: pre-seed the challenger before constructing the transcript, so you can
//! bind public inputs without bloating the proof.
//!
//! # Transcript boundaries (strict consumption)
//!
//! [`verify_multi`] rejects trailing transcript data via
//! [`VerifierError::TranscriptNotConsumed`]. This is intentional: it makes it harder
//! to accidentally accept a proof embedded inside a larger transcript with confusing
//! boundaries.
//!
//! If you want to bundle extra data alongside the proof in the same transcript,
//! you must manage boundaries yourself (e.g. parse and validate that data first,
//! then call [`verify_multi`] on the remaining transcript).
//!
//! # Multi-trace ordering
//!
//! For [`verify_multi`], `instances` must be provided in ascending trace height order
//! (smallest first). This is a protocol-level requirement.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_field::{ExtensionField, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lifted_air::{LiftedAir, ReducedAuxValues, ReductionError, VarLenPublicInputs};
use p3_miden_lifted_fri::verifier::{PcsError, verify_aligned};
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::{TranscriptError, VerifierChannel};
use thiserror::Error;

use p3_miden_lifted_stark::{
    AirInstance, LiftedCoset, StarkConfig, ValidationError, sample_ood_point,
};

use crate::constraints::{ConstraintFolder, reconstruct_quotient, row_to_packed_ext};
use crate::periodic::PeriodicPolys;

/// Errors that can occur during verification.
#[derive(Debug, Error)]
pub enum VerifierError {
    #[error("invalid instances: {0}")]
    Validation(#[from] ValidationError),
    #[error("PCS verification failed: {0}")]
    Pcs(#[from] PcsError),
    #[error("transcript error: {0}")]
    Transcript(#[from] TranscriptError),
    #[error("invalid aux shape")]
    InvalidAuxShape,
    #[error("constraint mismatch: quotient * vanishing != folded constraints")]
    ConstraintMismatch,
    #[error("transcript not fully consumed")]
    TranscriptNotConsumed,
    #[error("invalid periodic table")]
    InvalidPeriodicTable,
    #[error("mixed aux traces: either all AIRs must have aux columns or none")]
    MixedAuxTraces,
    #[error("global reduced aux identity check failed")]
    InvalidReducedAux,
    #[error("aux value reduction failed: {0}")]
    Reduction(ReductionError),
}

/// Verify a single AIR.
///
/// Transcript warning: the protocol assumes the challenger inside `channel` has
/// already observed all variable statement inputs (in particular `public_values`).
/// This lets callers keep public inputs out of the proof when they are available
/// out-of-band. See the module-level docs for guidance.
/// This is a convenience wrapper around [`verify_multi`] for the single-AIR case.
///
/// # Arguments
/// - `config`: STARK configuration (PCS params, LMCS, DFT)
/// - `air`: The AIR definition
/// - `log_trace_height`: Log₂ of the trace height
/// - `public_values`: Public values for this AIR
/// - `var_len_public_inputs`: Variable-length public inputs for bus identity checks
/// - `channel`: Verifier channel for transcript
///
/// # Returns
/// `Ok(())` on success, or a `VerifierError` if verification fails.
pub fn verify_single<F, EF, A, SC, Ch>(
    config: &SC,
    air: &A,
    log_trace_height: usize,
    public_values: &[F],
    var_len_public_inputs: VarLenPublicInputs<'_, F>,
    channel: &mut Ch,
) -> Result<(), VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    SC: StarkConfig<F, EF>,
    A: LiftedAir<F, EF>,
    Ch: VerifierChannel<F = F, Commitment = <SC::Lmcs as Lmcs>::Commitment>,
{
    let instance = AirInstance {
        log_trace_height,
        public_values,
        var_len_public_inputs,
    };
    verify_multi(config, &[(air, instance)], channel)
}

/// Verify multiple AIRs with traces of different heights.
///
/// Transcript warning: the protocol assumes the challenger inside `channel` has
/// already observed all variable statement inputs (in particular each instance's
/// `public_values`). This lets callers keep public inputs out of the proof when they
/// are available out-of-band.
///
/// Instances must be provided in ascending height order (smallest first). Each trace
/// may have a different height that is a power of 2. The verifier mirrors the prover's
/// protocol:
///
/// 1. Receive commitments and sample challenges in the same order as the prover
/// 2. For each AIR, evaluate constraints at the lifted OOD point yⱼ = z^{rⱼ}
/// 3. Accumulate folded constraints with β: acc = acc·β + foldedⱼ
/// 4. Check quotient identity: `acc == Q(z) * Z_{H_max}(z)`
///
/// Lifting ensures correctness: for a trace of height nⱼ lifted by factor rⱼ,
/// the committed codeword corresponds to the lifted polynomial `p_lift(X) = p(X^{rⱼ})`.
/// Opening at `[z, z * h_max]` therefore yields
/// `[p(z^{r_j}), p(h_{n_j} * z^{r_j})]`, i.e. the local/next row pair for the original
/// (unlifted) trace domain.
///
/// # Arguments
/// - `config`: STARK configuration (PCS params, LMCS, DFT)
/// - `instances`: Pairs of (AIR, instance) sorted by trace height (ascending)
/// - `channel`: Verifier channel for transcript/proof I/O
///
/// # Returns
/// `Ok(())` on success, or a `VerifierError` if verification fails.
pub fn verify_multi<F, EF, A, SC, Ch>(
    config: &SC,
    instances: &[(&A, AirInstance<'_, F>)],
    channel: &mut Ch,
) -> Result<(), VerifierError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    SC: StarkConfig<F, EF>,
    A: LiftedAir<F, EF>,
    Ch: VerifierChannel<F = F, Commitment = <SC::Lmcs as Lmcs>::Commitment>,
{
    let air_instances: Vec<_> = instances.iter().map(|(_, inst)| *inst).collect();
    p3_miden_lifted_stark::validate_instances(&air_instances)?;

    let aux_widths: Vec<_> = instances.iter().map(|(air, _)| air.aux_width()).collect();
    if !(aux_widths.iter().all(|&w| w > 0) || aux_widths.iter().all(|&w| w == 0)) {
        return Err(VerifierError::MixedAuxTraces);
    }
    let has_aux = aux_widths.iter().any(|&w| w > 0);

    let log_blowup = config.pcs().fri.log_blowup;

    // Infer constraint degree from symbolic AIR analysis (max across all AIRs)
    let constraint_degree = instances
        .iter()
        .map(|(air, _)| air.constraint_degree())
        .max()
        .unwrap_or(2);

    // Max trace height determines the LDE domain
    // validate_instances rejects empty input, so last() is always Some.
    let log_max_trace_height = instances
        .last()
        .expect("non-empty instances")
        .1
        .log_trace_height;
    let max_trace_height = 1usize << log_max_trace_height;
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

    // 3. Receive aux trace commitment (only when AIRs have aux columns)
    let aux_commit = if has_aux {
        Some(channel.receive_commitment()?.clone())
    } else {
        None
    };

    // Receive aux values from the transcript (one EF element per aux value, per instance).
    // When no AIR has aux columns, each entry is empty so nothing is received.
    let all_aux_values: Vec<Vec<EF>> = instances
        .iter()
        .map(|(air, _)| {
            let count = air.num_aux_values();
            (0..count)
                .map(|_| channel.receive_algebra_element::<EF>())
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;

    // 4. Sample constraint folding alpha and accumulation beta
    let alpha: EF = channel.sample_algebra_element::<EF>();
    let beta: EF = channel.sample_algebra_element::<EF>();

    // 5. Receive quotient commitment
    let quotient_commit = channel.receive_commitment()?.clone();

    // 6. Sample OOD point (outside max trace domain H and max LDE coset gK)
    let z: EF = sample_ood_point(channel, &max_lde_coset);
    let h = F::two_adic_generator(log_max_trace_height);
    let z_next = z * h;

    // 7. Widths per commitment group (unpadded data widths).
    let main_widths: Vec<usize> = instances.iter().map(|(air, _)| air.width()).collect();
    let aux_widths: Vec<usize> = instances
        .iter()
        .map(|(air, _)| air.aux_width() * EF::DIMENSION)
        .collect();
    let quotient_widths: Vec<usize> = vec![constraint_degree * EF::DIMENSION];

    // Build commitments with original (unpadded) widths.
    // The PCS aligned wrapper handles alignment and truncation internally.
    let commitments = match aux_commit {
        Some(aux_commit) => vec![
            (main_commit, main_widths),
            (aux_commit, aux_widths),
            (quotient_commit, quotient_widths),
        ],
        None => vec![
            (main_commit, main_widths),
            (quotient_commit, quotient_widths),
        ],
    };

    // 8. Verify PCS openings (returns per-matrix RowMajorMatrix<EF>, truncated to original widths)
    let opened = verify_aligned::<F, EF, SC::Lmcs, _, 2>(
        config.pcs(),
        config.lmcs(),
        &commitments,
        log_lde_height,
        [z, z_next],
        channel,
    )?;

    // Group indices for accessing opened matrices.
    let (main_g, aux_g, quot_g) = if has_aux {
        (0, Some(1), 2)
    } else {
        (0, None, 1)
    };

    // 10. Per-AIR constraint evaluation and beta accumulation
    //
    // opened[g] has one matrix per AIR (for main/aux) or one matrix total (quotient).
    // Each matrix has N=2 rows: row 0 = local (z), row 1 = next (z·h).
    debug_assert_eq!(opened[main_g].len(), instances.len());
    debug_assert!(aux_g.is_none_or(|g| opened[g].len() == instances.len()));
    let mut accumulated = EF::ZERO;
    let mut reduced_aux = ReducedAuxValues::<EF>::identity();

    for (j, (air, inst)) in instances.iter().enumerate() {
        let coset_j = LiftedCoset::new(inst.log_trace_height, log_blowup, log_max_trace_height);

        // opened[main_g][j] is a 2-row RowMajorMatrix (local, next) already truncated.
        let main_pair = opened[main_g][j].clone();

        // Extract aux trace opened values (reconstitute EF from base field components).
        let aux_ef_width = air.aux_width();
        let (aux_local, aux_next) = match aux_g {
            Some(g) => {
                let mut rows = opened[g][j].row_slices();
                let local = row_to_packed_ext::<F, EF>(rows.next().expect("row 0 (local)"))?;
                let next = row_to_packed_ext::<F, EF>(rows.next().expect("row 1 (next)"))?;
                (local, next)
            }
            None => (vec![], vec![]),
        };
        let aux_pair = RowMajorMatrix::new([aux_local, aux_next].concat(), aux_ef_width);

        // Selectors at the lifted OOD point yⱼ = z^{rⱼ} (encapsulated in LiftedCoset).
        let selectors = coset_j.selectors_at::<F, _>(z);

        // Periodic values: for a column with period p, eval_at computes z^{n/p}.
        // Using (max_trace_height, z) gives z^{max_n / p}, which equals
        // y_j^{n_j / p} = (z^{max_n/n_j})^{n_j/p} = z^{max_n/p}. This avoids
        // computing y_j = z^{r_j} explicitly.
        let periodic_polys = PeriodicPolys::new(air.periodic_columns())
            .ok_or(VerifierError::InvalidPeriodicTable)?;
        let periodic_values = periodic_polys.eval_at::<EF>(max_trace_height, z);

        let aux_values_j = &all_aux_values[j];

        let num_rand = air.num_randomness();
        let mut folder = ConstraintFolder {
            main: main_pair,
            aux: aux_pair,
            randomness: &randomness[..num_rand],
            public_values: inst.public_values,
            periodic_values: &periodic_values,
            aux_values: aux_values_j,
            selectors,
            alpha,
            accumulator: EF::ZERO,
            _phantom: PhantomData,
        };

        air.eval(&mut folder);

        // Accumulate: acc = acc * beta + folded_j
        accumulated = accumulated * beta + folder.accumulator;

        // Compute reduced aux contribution and accumulate
        let contribution = air
            .reduced_aux_values(
                aux_values_j,
                &randomness[..num_rand],
                inst.public_values,
                inst.var_len_public_inputs,
            )
            .map_err(VerifierError::Reduction)?;
        reduced_aux.combine_in_place(&contribution);
    }

    // 11. Reconstruct Q(z) and check quotient identity Q(z) * Z_{H_max}(z)
    // Quotient group has a single matrix; row 0 is the evaluation at z.
    let mut quot_rows = opened[quot_g][0].row_slices();
    let quotient_chunks = row_to_packed_ext::<F, EF>(quot_rows.next().expect("quotient row 0"))?;
    let quotient_z = reconstruct_quotient::<F, EF>(z, &max_lde_coset, &quotient_chunks);

    let vanishing = max_lde_coset.vanishing_at::<F, _>(z);
    if accumulated != quotient_z * vanishing {
        return Err(VerifierError::ConstraintMismatch);
    }

    // 12. Check global reduced aux identity (all bus contributions combine to identity)
    if !reduced_aux.is_identity() {
        return Err(VerifierError::InvalidReducedAux);
    }

    // 13. Ensure transcript is fully consumed
    if !channel.is_empty() {
        return Err(VerifierError::TranscriptNotConsumed);
    }

    Ok(())
}
