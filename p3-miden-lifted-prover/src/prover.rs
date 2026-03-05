//! Lifted STARK prover.
//!
//! This module provides:
//! - [`prove_single`]: Prove a single AIR instance.
//! - [`prove_multi`]: Prove multiple AIR instances with traces of different heights.
//!
//! These functions write the proof into a [`p3_miden_transcript::ProverChannel`]
//! (commitments, grinding witnesses, and openings).
//!
//! # Fiat-Shamir / transcript binding (initial challenger state)
//!
//! This crate intentionally does **not** prescribe the *initial* transcript state.
//! Different applications may transport some statement data out-of-band (e.g. public
//! inputs shipped alongside a proof) and do not want it duplicated inside the proof.
//!
//! The flip side is that the caller must ensure the Fiat-Shamir challenger inside
//! `channel` is initialized and bound to the statement in whatever way is appropriate
//! for their use case.
//!
//! The protocol implementation assumes that *all inputs that may vary* (including
//! `public_values` and `var_len_public_inputs`) have been observed by the challenger.
//! This is required so callers can avoid including public inputs in the proof when
//! they are available out-of-band.
//!
//! In particular, the caller **MUST** bind both `public_values` and
//! `var_len_public_inputs` to the challenger state. Otherwise, Fiat-Shamir
//! challenges sampled during proving/verification are independent of the public
//! inputs.
//!
//! Because the `air` is a concrete Rust type, you often do not need to explicitly
//! observe it into the challenger *if your application has a single fixed AIR version
//! compiled in*. If you support multiple AIRs/versions/protocol configurations, you
//! should also bind some `air_id` / version tag / domain separator to prevent
//! cross-protocol replay.
//!
//! ## Recommended pattern: pre-seed the challenger (no proof bloat)
//!
//! The transcript channel traits do not currently expose an “observe-only” operation
//! (observe into the challenger without recording into the proof). If you want to
//! bind public inputs without bloating the proof, the most ergonomic pattern is to
//! pre-seed the challenger and then construct the transcript.
//!
//!
//! ```ignore
//! use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
//! use p3_challenger::{CanObserve, DuplexChallenger};
//! use p3_dft::Radix2DitParallel;
//! use p3_field::extension::BinomialExtensionField;
//! use p3_miden_lifted_prover::{AirWitness, StarkConfig, prove_multi};
//! use p3_miden_lifted_verifier::AirInstance;
//! use p3_miden_lmcs::LmcsConfig;
//! use p3_miden_stateful_hasher::StatefulSponge;
//! use p3_miden_transcript::{ProverTranscript, VerifierTranscript};
//! use p3_symmetric::TruncatedPermutation;
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//!
//! // Concrete instantiation matching the repository's BabyBear+Poseidon2 defaults.
//! const WIDTH: usize = 16;
//! const RATE: usize = 8;
//! const DIGEST: usize = 8;
//!
//! type F = BabyBear;
//! type EF = BinomialExtensionField<F, 4>;
//! type P = <F as p3_field::Field>::Packing;
//! type Perm = Poseidon2BabyBear<WIDTH>;
//! type Challenger = DuplexChallenger<F, Perm, WIDTH, RATE>;
//! type Sponge = StatefulSponge<Perm, WIDTH, RATE, DIGEST>;
//! type Compress = TruncatedPermutation<Perm, 2, DIGEST, WIDTH>;
//! type Lmcs = LmcsConfig<P, P, Sponge, Compress, WIDTH, DIGEST>;
//! type Dft = Radix2DitParallel<F>;
//!
//! // --- Build config ---
//! let mut rng = SmallRng::seed_from_u64(0);
//! let perm = Perm::new_from_rng_128(&mut rng);
//! let sponge = Sponge::new(perm.clone());
//! let compress = Compress::new(perm.clone());
//! let config = StarkConfig { pcs: /* ... */, lmcs: Lmcs::new(sponge, compress), dft: Dft::default() };
//!
//! // --- Statement (out-of-band) ---
//! let public_values: Vec<F> = /* ... */;
//! let log_trace_height: usize = /* ... */;
//!
//! // --- Prover: bind statement into Fiat-Shamir ---
//! let mut ch = Challenger::new(perm.clone());
//! // Domain separator: one BabyBear element per ASCII byte.
//! ch.observe_slice(&b"LSTARK0".map(|b| F::from_u8(b)));
//! ch.observe_slice(&public_values);
//! // If your app supports multiple AIRs/versions, also observe an application-level air_id here.
//! let mut prover_channel = ProverTranscript::new(ch);
//!
//! // Prove writes into `prover_channel`.
//! let witness = AirWitness::new(&trace, &public_values, &[]);
//! prove_multi(&config, &[(&air, witness, &aux_builder)], &mut prover_channel)?;
//! let transcript = prover_channel.into_data();
//!
//! // --- Verifier: same binding, then consume transcript ---
//! let mut ch = Challenger::new(perm);
//! ch.observe_slice(&b"LSTARK0".map(|b| F::from_u8(b)));
//! ch.observe_slice(&public_values);
//! let mut verifier_channel = VerifierTranscript::from_data(ch, &transcript);
//!
//! let instance = AirInstance { log_trace_height, public_values: &public_values, var_len_public_inputs: &[] };
//! p3_miden_lifted_verifier::verify_multi(&config, &[(&air, instance)], &mut verifier_channel)?;
//! ```
//!
//! ## Alternative: write statement data into the transcript
//!
//! If you do want the proof to be self-contained, you can `send_field_slice` the
//! statement data into the transcript before calling [`prove_single`] / [`prove_multi`].
//! In that case, the verifier must *first* read and validate those values from the
//! transcript, and only then call [`p3_miden_lifted_verifier::verify_multi`].
//!
//! More generally, you can choose to obtain some parameters from the channel. If you
//! do, you are responsible for validating them before use.
//!
//! # Multi-trace ordering
//!
//! For [`prove_multi`], `instances` must be provided in ascending trace height order
//! (smallest first). This is a protocol-level requirement.
//!
//! Internally, quotient numerators are accumulated using cyclic extension before a
//! single vanishing division on the largest quotient domain.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use p3_field::{BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_lifted_air::{AuxBuilder, LiftedAir, VarLenPublicInputs, get_constraint_layout};
use p3_miden_lifted_fri::prover::open_with_channel;
use p3_miden_lmcs::Lmcs;
use p3_miden_transcript::ProverChannel;
use p3_util::log2_strict_usize;
use thiserror::Error;
use tracing::{info_span, instrument};

use p3_miden_lifted_stark::{
    AirWitness, LiftedCoset, StarkConfig, ValidationError, sample_ood_point,
};

use crate::commit::commit_traces;
use crate::constraints::evaluate_constraints_into;
use crate::quotient;

use crate::periodic::PeriodicLde;

/// Errors that can occur during proving.
#[derive(Debug, Error)]
pub enum ProverError {
    #[error("invalid instances: {0}")]
    Validation(#[from] ValidationError),
}

/// Prove a single AIR.
///
/// Transcript warning: the protocol assumes the challenger inside `channel` has
/// already observed all variable statement inputs (in particular `public_values`
/// and `var_len_public_inputs`). This lets callers keep public inputs out of the
/// proof when they are available out-of-band. See the module-level docs for
/// recommended patterns.
///
/// This is a convenience wrapper around [`prove_multi`] for the single-AIR case.
///
/// # Returns
/// `Ok(())` on success, or a `ProverError` if validation fails.
pub fn prove_single<F, EF, A, B, SC, Ch>(
    config: &SC,
    air: &A,
    trace: &RowMajorMatrix<F>,
    public_values: &[F],
    var_len_public_inputs: VarLenPublicInputs<'_, F>,
    aux_builder: &B,
    channel: &mut Ch,
) -> Result<(), ProverError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    SC: StarkConfig<F, EF>,
    A: LiftedAir<F, EF>,
    B: AuxBuilder<F, EF>,
    Ch: ProverChannel<F = F, Commitment = <SC::Lmcs as Lmcs>::Commitment>,
{
    let witness = AirWitness::new(trace, public_values, var_len_public_inputs);
    prove_multi(config, &[(air, witness, aux_builder)], channel)
}

/// Prove multiple AIRs with traces of different heights.
///
/// Transcript warning: the protocol assumes the challenger inside `channel` has
/// already observed all variable statement inputs (in particular each instance's
/// `public_values`). This lets callers keep public inputs out of the proof when they
/// are available out-of-band.
///
/// Instances must be provided in ascending height order (smallest first). Each trace
/// may have a different height that is a power of 2. The quotient numerators are
/// accumulated using cyclic extension:
///
/// 1. Compute numerator N₀ on the smallest quotient domain.
/// 2. For each subsequent trace `j`:
///    - cyclically extend the accumulator to the new (larger) quotient domain,
///    - fold with the random challenge β: acc = acc·β + Nⱼ.
/// 3. Divide by `Z_H` once on the largest quotient domain.
///
/// The ordering is important: cyclic extension only grows the accumulator (it does not
/// shrink), and both prover and verifier must assign the same powers of `beta` to each
/// instance's contribution.
///
/// # Arguments
/// - `config`: STARK configuration (PCS params, LMCS, DFT)
/// - `instances`: Pairs of (AIR, witness) sorted by trace height (ascending).
/// - `channel`: Prover channel for transcript/proof I/O
///
/// # Returns
/// `Ok(())` on success, or a `ProverError` if validation fails.
#[instrument(name = "prove", skip_all)]
pub fn prove_multi<F, EF, A, B, SC, Ch>(
    config: &SC,
    instances: &[(&A, AirWitness<'_, F>, &B)],
    channel: &mut Ch,
) -> Result<(), ProverError>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    SC: StarkConfig<F, EF>,
    A: LiftedAir<F, EF>,
    B: AuxBuilder<F, EF>,
    Ch: ProverChannel<F = F, Commitment = <SC::Lmcs as Lmcs>::Commitment>,
{
    validate_inputs(instances)?;

    let log_blowup = config.pcs().fri.log_blowup;

    // Infer constraint degree from symbolic AIR analysis (max across all AIRs)
    let log_constraint_degree = instances
        .iter()
        .map(|(air, _, _)| air.log_quotient_degree())
        .max()
        .unwrap_or(1);

    // Max trace height determines the LDE domain
    let max_trace_height = instances.last().unwrap().1.trace.height();
    let log_max_trace_height = log2_strict_usize(max_trace_height);
    let log_lde_height = log_max_trace_height + log_blowup;

    // Max LDE coset (for the largest trace, no lifting)
    let max_lde_coset = LiftedCoset::unlifted(log_max_trace_height, log_blowup);
    let max_quotient_coset = max_lde_coset.quotient_domain(log_constraint_degree);
    let max_quotient_height = max_quotient_coset.lde_height();

    // 1. Commit all main traces
    // Clone with blowup × capacity so the DFT resize doesn't reallocate.
    let blowup = 1usize << log_blowup;
    let main_traces: Vec<_> = instances
        .iter()
        .map(|(_, w, _)| {
            let src = &w.trace.values;
            let mut values = Vec::with_capacity(src.len() * blowup);
            values.extend_from_slice(src);
            RowMajorMatrix::new(values, w.trace.width())
        })
        .collect();
    let main_committed =
        info_span!("commit to main traces").in_scope(|| commit_traces(config, main_traces));
    channel.send_commitment(main_committed.root());

    // 2. Sample randomness and build aux traces for all AIRs
    let max_num_randomness = instances
        .iter()
        .map(|(air, _, _)| air.num_randomness())
        .max()
        .unwrap_or(0);

    let randomness: Vec<EF> = (0..max_num_randomness)
        .map(|_| channel.sample_algebra_element::<EF>())
        .collect();

    // Build aux traces via AuxBuilder
    let (aux_traces_ef, all_aux_values): (Vec<RowMajorMatrix<EF>>, Vec<Vec<EF>>) =
        tracing::debug_span!("build aux traces").in_scope(|| {
            let mut traces = Vec::with_capacity(instances.len());
            let mut values = Vec::with_capacity(instances.len());
            for (air, w, aux_builder) in instances {
                let num_rand = air.num_randomness();
                let (aux, aux_vals) = aux_builder.build_aux_trace(w.trace, &randomness[..num_rand]);

                assert_eq!(aux.width(), air.aux_width(), "aux trace width mismatch");
                assert_eq!(
                    aux_vals.len(),
                    air.num_aux_values(),
                    "aux values length mismatch: build_aux_trace returned {} values, \
                     but num_aux_values() is {}",
                    aux_vals.len(),
                    air.num_aux_values()
                );
                assert_eq!(aux.height(), w.trace.height());
                traces.push(aux);
                values.push(aux_vals);
            }
            (traces, values)
        });

    // Flatten EF -> F and commit aux traces
    let aux_traces: Vec<RowMajorMatrix<F>> = aux_traces_ef
        .into_iter()
        .map(|aux| {
            let base_width = aux.width() * EF::DIMENSION;
            let base_values = <EF as BasedVectorSpace<F>>::flatten_to_base(aux.values);
            RowMajorMatrix::new(base_values, base_width)
        })
        .collect();

    let aux_committed =
        info_span!("commit to aux traces").in_scope(|| commit_traces(config, aux_traces));
    channel.send_commitment(aux_committed.root());

    // Observe aux values into the transcript (binds to Fiat-Shamir state).
    // When no AIR has aux columns, each entry is empty so nothing is sent.
    for vals in &all_aux_values {
        for &val in vals {
            channel.send_algebra_element(val);
        }
    }

    // 4. Sample constraint folding alpha and accumulation beta
    let alpha: EF = channel.sample_algebra_element::<EF>();
    let beta: EF = channel.sample_algebra_element::<EF>();

    // 5. Evaluate constraints and accumulate with beta folding.
    //
    // Single accumulator, processed in ascending trace height order:
    //   1. Cyclically extend accumulator to the next quotient height
    //   2. Multiply every element by beta
    //   3. Add constraint evaluations in-place: acc[i] += eval(i)
    //
    // Pre-allocate with LDE capacity so commit_quotient's resize doesn't reallocate.
    let constraint_degree = 1 << log_constraint_degree;
    let mut accumulator: Vec<EF> = Vec::with_capacity(max_quotient_height * blowup);

    // Pre-compute constraint layouts for each AIR (base/ext index mapping)
    let layouts: Vec<_> = instances
        .iter()
        .map(|(air, w, _)| get_constraint_layout::<F, EF, A>(*air, w.public_values.len()))
        .collect();

    info_span!("evaluate constraints").in_scope(|| {
        for (i, (air, w, _)) in instances.iter().enumerate() {
            let trace_height = w.trace.height();
            let log_trace_height = log2_strict_usize(trace_height);

            // Create LiftedCoset for this trace (may be lifted relative to max)
            let this_lde_coset =
                LiftedCoset::new(log_trace_height, log_blowup, log_max_trace_height);
            let this_quotient_coset = this_lde_coset.quotient_domain(log_constraint_degree);
            let this_quotient_height = this_quotient_coset.lde_height();

            // Truncate the committed LDE to the quotient evaluation domain gJ (size N·D).
            // Since B ≥ D, the committed LDE on gK (size N·B) contains gJ as a prefix in
            // bit-reversed storage, so this is a zero-copy view.
            let main_on_gj = main_committed.evals_on_quotient_domain(i, constraint_degree);
            let aux_on_gj = aux_committed.evals_on_quotient_domain(i, constraint_degree);

            // Build periodic LDE for this trace via coset method
            let periodic_lde = PeriodicLde::build(&this_quotient_coset, air.periodic_columns());

            // Cyclically extend accumulator to this quotient height and scale by beta.
            // On the first iteration the accumulator is empty, so this is a no-op
            // and evaluate_constraints_into writes into a zero-filled buffer.
            tracing::debug_span!(
                "cyclic_extend",
                acc_len = accumulator.len(),
                target = this_quotient_height
            )
            .in_scope(|| {
                quotient::cyclic_extend_and_scale(&mut accumulator, this_quotient_height, beta);
            });

            let aux_values_i = &all_aux_values[i];

            // Add constraint evaluations in-place: accumulator[i] += eval(i)
            tracing::debug_span!("eval_instance", instance = i, height = this_quotient_height)
                .in_scope(|| {
                    evaluate_constraints_into::<F, EF, A, _>(
                        &mut accumulator,
                        *air,
                        &main_on_gj,
                        &aux_on_gj,
                        &this_quotient_coset,
                        alpha,
                        &randomness[..air.num_randomness()],
                        w.public_values,
                        &periodic_lde,
                        &layouts[i],
                        aux_values_i,
                    );
                });
        }
    });

    // Verify we have the expected size (max quotient domain)
    assert_eq!(accumulator.len(), max_quotient_height);

    // 6. Divide by vanishing polynomial once on full gJ (in-place)
    tracing::debug_span!("divide_by_vanishing", height = max_quotient_height).in_scope(|| {
        quotient::divide_by_vanishing_in_place::<F, EF>(&mut accumulator, &max_quotient_coset);
    });

    // 7. Commit quotient
    let quotient_committed = info_span!("commit to quotient poly chunks")
        .in_scope(|| quotient::commit_quotient(config, accumulator, &max_lde_coset));
    channel.send_commitment(quotient_committed.root());

    // 8. Sample OOD point (outside H and gK)
    let z: EF = sample_ood_point(channel, &max_lde_coset);
    let h = F::two_adic_generator(log_max_trace_height);
    let z_next = z * h;

    // 9. Open via PCS
    let trees = vec![
        main_committed.tree(),
        aux_committed.tree(),
        quotient_committed.tree(),
    ];

    info_span!("open").in_scope(|| {
        open_with_channel::<F, EF, SC::Lmcs, RowMajorMatrix<F>, _, 2>(
            config.pcs(),
            config.lmcs(),
            log_lde_height,
            [z, z_next],
            &trees,
            channel,
        )
    });

    Ok(())
}

/// Validate prover inputs: width match, non-empty, ascending height.
///
/// Power-of-two height is enforced by [`AirWitness::new`].
fn validate_inputs<F, EF, A, B>(
    instances: &[(&A, AirWitness<'_, F>, &B)],
) -> Result<(), ProverError>
where
    F: Field,
    EF: ExtensionField<F>,
    A: LiftedAir<F, EF>,
{
    for (air, w, _) in instances {
        if w.trace.width() != air.width() {
            return Err(ValidationError::WidthMismatch.into());
        }
        if w.var_len_public_inputs.len() != air.num_var_len_public_inputs() {
            return Err(ValidationError::VarLenPublicInputsMismatch.into());
        }
    }
    let verifier_instances: Vec<_> = instances.iter().map(|(_, w, _)| w.to_instance()).collect();
    p3_miden_lifted_stark::validate_instances(&verifier_instances)?;
    Ok(())
}
