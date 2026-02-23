use alloc::vec::Vec;

use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_miden_lmcs::RowList;
use p3_miden_transcript::{TranscriptError, VerifierChannel};

/// Out-of-domain (OOD) evaluations in point-major layout.
///
/// In the DEEP technique, the prover evaluates committed polynomials at random
/// challenge points z₀, z₁, ... outside the evaluation domain. These claimed
/// evaluations let the verifier reduce polynomial identity checks to a single
/// low-degree test (FRI), avoiding re-evaluation over the whole domain.
///
/// `points[point_idx]` is a `RowList<EF>` with one row per matrix (across all groups).
/// Point-major layout matches the verifier's access pattern: it reduces all column
/// evaluations for one point at a time via Horner, so `point(idx).iter_values()`
/// yields exactly the values needed for each reduction step.
#[derive(Clone, Debug)]
pub struct DeepEvals<EF> {
    points: Vec<RowList<EF>>,
}

impl<EF: Field> DeepEvals<EF> {
    /// Parse out-of-domain evaluations from a verifier channel.
    ///
    /// Reads in point-major order: for each evaluation point, then for each matrix
    /// width, we receive that many extension field elements.
    /// Widths must include any alignment padding applied during commitment.
    pub(crate) fn read_from_channel<F, Ch>(
        widths: &[&[usize]],
        num_points: usize,
        channel: &mut Ch,
    ) -> Result<Self, TranscriptError>
    where
        F: TwoAdicField,
        EF: ExtensionField<F>,
        Ch: VerifierChannel<F = F>,
    {
        let flat_widths: Vec<usize> = widths.iter().flat_map(|s| s.iter()).copied().collect();
        let total: usize = flat_widths.iter().sum();

        let mut points = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            // All column evaluations for this point; RowList recovers per-matrix structure.
            let elems = channel.receive_algebra_slice::<EF>(total)?;
            points.push(RowList::new(elems, &flat_widths));
        }

        Ok(Self { points })
    }

    /// Get the `RowList` for a specific evaluation point.
    pub fn point(&self, idx: usize) -> &RowList<EF> {
        &self.points[idx]
    }

    /// Number of evaluation points.
    pub fn num_points(&self) -> usize {
        self.points.len()
    }
}
