//! FRI Verifier
//!
//! Verifies that a committed polynomial is close to low-degree.
//!
//! # Domain Structure
//!
//! The prover commits to evaluations on domain D of size 2^log_domain_size in bit-reversed order.
//! Each folding round groups `arity` consecutive evaluations into cosets and folds them.
//!
//! For arity = 2:
//!   - Row i contains evaluations at coset {s, −s} where s = g^{bitrev(i)}
//!   - g is the generator of D (has order n)
//!
//! For arity = 4:
//!   - Row i contains evaluations at coset {s, −s, ωs, −ωs} where ω = √−1
//!
//! # Index Semantics
//!
//! The query `index` has two parts:
//!   - High bits: which row (coset) in the committed matrix
//!   - Low bits: which position within the coset
//!
//! After each fold, we shift off `log_arity` bits, moving to the parent coset.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use p3_challenger::CanSample;
use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_lmcs::{Lmcs, LmcsError};
use p3_miden_transcript::{TranscriptError, VerifierChannel};
use p3_util::reverse_bits_len;
use thiserror::Error;

use crate::fri::FriParams;
use crate::utils::horner;

/// FRI low-degree test oracle.
///
/// Created via [`FriOracle::new`], which samples folding challenges from
/// the Fiat-Shamir transcript. The oracle tests that evaluations are close
/// to a low-degree polynomial.
///
/// Uses a single base-field LMCS. Opened base field values are reconstructed
/// to extension field for folding verification.
pub struct FriOracle<F, EF, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    L: Lmcs<F = F>,
{
    /// Log₂ of the initial domain size.
    log_domain_size: usize,
    /// Per-round commitment and folding challenge.
    rounds: Vec<FriRoundOracle<L::Commitment, EF>>,
    /// Coefficients of the final low-degree polynomial in descending degree order
    /// `[cₙ, ..., c₁, c₀]`, ready for direct Horner evaluation.
    final_poly: Vec<EF>,
}

struct FriRoundOracle<Commitment, EF> {
    commitment: Commitment,
    beta: EF,
}

impl<F, EF, L> FriOracle<F, EF, L>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + Clone,
    L: Lmcs<F = F>,
{
    /// Create oracle by reading from a verifier channel.
    pub fn new<Ch>(
        params: &FriParams,
        log_domain_size: usize,
        channel: &mut Ch,
    ) -> Result<Self, FriError>
    where
        Ch: VerifierChannel<F = F, Commitment = L::Commitment> + CanSample<F>,
    {
        let num_rounds = params.num_rounds(log_domain_size);
        let mut rounds = Vec::with_capacity(num_rounds);

        for _ in 0..num_rounds {
            let commitment = channel.receive_commitment()?.clone();

            channel.grind(params.folding_pow_bits)?;

            let beta: EF = channel.sample_algebra_element();
            rounds.push(FriRoundOracle { commitment, beta });
        }

        let final_degree = params.final_poly_degree(log_domain_size);
        let final_poly = channel.receive_algebra_slice(final_degree)?;

        Ok(Self {
            log_domain_size,
            rounds,
            final_poly,
        })
    }

    /// Test low-degree proximity by reading openings from a verifier channel.
    ///
    /// `evals` maps tree indices to DEEP evaluations.
    /// Tree index = bitrev(exp, log_domain_size) where domain point = `g·ω^{exp}`.
    ///
    /// Empty `evals` will fail at the first round's LMCS `open_batch` call,
    /// which rejects empty indices.
    pub fn test_low_degree<Ch>(
        &self,
        lmcs: &L,
        params: &FriParams,
        mut evals: BTreeMap<usize, EF>,
        channel: &mut Ch,
    ) -> Result<(), FriError>
    where
        Ch: VerifierChannel<F = F, Commitment = L::Commitment>,
    {
        let log_arity = params.fold.log_arity();
        let arity = params.fold.arity();
        // FRI commits base-field values; each extension element spans DIMENSION base elements.
        let base_width = arity * EF::DIMENSION;
        let widths = [base_width];

        let mut log_domain_size = self.log_domain_size;
        let mut g_inv = F::two_adic_generator(log_domain_size).inverse();

        for (round_idx, round) in self.rounds.iter().enumerate() {
            let log_folded_domain_size = log_domain_size - log_arity;

            // Compute row indices: shift off position-within-coset bits
            let row_indices = evals.keys().map(|&idx| idx >> log_arity);

            let opened_rows = lmcs
                .open_batch(
                    &round.commitment,
                    &widths,
                    log_folded_domain_size,
                    row_indices,
                    channel,
                )
                .map_err(|source| FriError::LmcsError {
                    source,
                    round: round_idx,
                })?;

            // Drain, verify, fold, and rebuild with new keys.
            //
            // SOUNDNESS NOTE: Multiple indices can map to the same row_idx after folding
            // (they differ only in their low log_arity bits). This is safe because:
            //
            // 1. Each closure verifies its specific position: `row[position] == eval`.
            //    All closures execute (Rust's collect drives the full iterator).
            //
            // 2. The folded value depends only on (row, s_inv, beta), not on position.
            //    Indices in the same coset share the same row and s_inv, so they fold
            //    to identical values. Keeping any one in the BTreeMap is correct.
            //
            // 3. The prover cannot provide different row data for the same row_idx.
            //    LMCS opens each row exactly once via `opened_rows[&row_idx]`.
            evals = evals
                .into_iter()
                .map(|(idx, eval)| {
                    // Decompose tree index: high bits = row (coset), low bits = position within coset
                    let row_idx = idx >> log_arity;
                    let position = idx & (arity - 1);

                    // open_batch guarantees all requested indices are returned with
                    // the correct widths.
                    let flat_row: &[F] = opened_rows
                        .get(&row_idx)
                        .and_then(|rows| rows.iter_rows().next()?.get(..base_width))
                        .ok_or(FriError::InvalidOpening {
                            tree_index: row_idx,
                            round: round_idx,
                        })?;
                    // Reinterpret base-field elements as extension field for folding.
                    let row: Vec<EF> = EF::reconstitute_from_base(flat_row.to_vec());

                    if row.get(position) != Some(&eval) {
                        return Err(FriError::EvaluationMismatch {
                            round: round_idx,
                            tree_index: row_idx,
                            position,
                        });
                    }

                    // s⁻¹ = (g^{bitrev(row_idx)})⁻¹, needed for iFFT over <s>.
                    let s_pow = reverse_bits_len(row_idx, log_folded_domain_size);
                    let s_inv = g_inv.exp_u64(s_pow as u64);
                    let folded = params.fold.fold_evals(&row, s_inv, round.beta);
                    Ok((row_idx, folded))
                })
                .collect::<Result<_, _>>()?;

            log_domain_size = log_folded_domain_size;
            g_inv = g_inv.exp_power_of_2(log_arity);
        }

        // After all folding rounds, the polynomial has been reduced to degree < final_degree.
        // The prover sent this final polynomial's coefficients; we evaluate it at each
        // folded query point on the final domain and check consistency with the folded
        // values. This closes the FRI proximity argument: if the original codeword was
        // far from low-degree, at least one query fails with high probability.
        //
        // `final_poly` is in descending degree order [cₙ, ..., c₁, c₀], which is
        // the native order for Horner evaluation.
        let generator = F::two_adic_generator(log_domain_size);
        for (idx, eval) in evals {
            let exp = reverse_bits_len(idx, log_domain_size);
            let x = generator.exp_u64(exp as u64);
            let final_eval: EF = horner(x, self.final_poly.iter().copied());

            if final_eval != eval {
                return Err(FriError::FinalPolyMismatch { tree_index: idx });
            }
        }

        Ok(())
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during FRI verification.
#[derive(Debug, Error)]
pub enum FriError {
    #[error("LMCS verification failed at round {round}: {source}")]
    LmcsError { source: LmcsError, round: usize },
    #[error("invalid opening for tree index {tree_index} at round {round}")]
    InvalidOpening { tree_index: usize, round: usize },
    #[error(
        "evaluation mismatch at round {round}, tree index {tree_index}, coset position {position}"
    )]
    EvaluationMismatch {
        round: usize,
        tree_index: usize,
        position: usize,
    },
    #[error("final polynomial mismatch at tree index {tree_index}")]
    FinalPolyMismatch { tree_index: usize },
    #[error("transcript error: {0}")]
    TranscriptError(#[from] TranscriptError),
}
