//! MMCS trait implementation for LMCS configurations.
//!
//! These implementations adapt LMCS trees/proofs to the standard [`Mmcs`] interface
//! used by FRI and other PCS code. The proof format mirrors [`crate::Proof`]:
//! opened rows (and optional salt) plus sibling hashes from leaf to root.
//!
//! Nuances:
//! - `open_batch` does not read transcript hints; it builds a single-leaf opening
//!   directly from the in-memory [`LiftedMerkleTree`].
//! - `verify_batch` recomputes the leaf hash from rows+salt, derives widths and
//!   the expected authentication-path length from [`Dimensions`], and checks the root.
//!   The caller must supply dimensions in the same height order used to build the tree,
//!   with widths matching the committed rows (including any alignment padding if
//!   `build_aligned_tree` was used). `Dimensions` are trusted
//!   statement data; verification does not re-check height ordering or power-of-two
//!   constraints beyond row-length checks.
//! - The hiding configuration delegates to the inner `LmcsConfig` implementation so
//!   proof shape and validation stay identical; only tree construction consumes randomness.

mod hiding;

#[cfg(test)]
mod tests;

use alloc::vec::Vec;
use core::iter::zip;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_stateful_hasher::{Alignable, StatefulHasher};
use p3_symmetric::{Hash, PseudoCompressionFunction};
use p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};

use crate::LmcsTree;
use crate::lifted_tree::LiftedMerkleTree;
use crate::{Lmcs, LmcsConfig, LmcsError};

// ============================================================================
// Mmcs implementation for LmcsConfig (non-hiding)
// ============================================================================

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Mmcs<PF::Value> for LmcsConfig<PF, PD, H, C, WIDTH, DIGEST_ELEMS, SALT_ELEMS>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    PF::Value: PartialEq,
    H: StatefulHasher<PF, [PD; DIGEST_ELEMS], State = [PD; WIDTH]>
        + StatefulHasher<PF::Value, [PD::Value; DIGEST_ELEMS], State = [PD::Value; WIDTH]>
        + Alignable<PF::Value, PD::Value>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
        + Sync,
    [PF::Value; SALT_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    [PD::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
{
    type ProverData<M> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST_ELEMS, SALT_ELEMS>;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST_ELEMS>;
    /// Proof includes salt and siblings: `([F; SALT_ELEMS], Vec<Self::Commitment>)`
    type Proof = ([PF::Value; SALT_ELEMS], Vec<Self::Commitment>);
    type Error = LmcsError;

    fn commit<M: Matrix<PF::Value>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        let tree = self.build_tree(inputs);
        (tree.root(), tree)
    }

    fn open_batch<M: Matrix<PF::Value>>(
        &self,
        index: usize,
        tree: &Self::ProverData<M>,
    ) -> BatchOpening<PF::Value, Self> {
        let final_height = tree.height();
        assert!(
            index < final_height,
            "index {index} out of range {final_height}"
        );

        let crate::Proof {
            rows,
            salt,
            siblings,
        } = tree.single_proof(index);

        // Convert RowList to Vec<Vec<F>> at the Mmcs trait boundary.
        let rows_vec: Vec<Vec<PF::Value>> = rows.iter_rows().map(|r| r.to_vec()).collect();
        BatchOpening::new(rows_vec, (salt, siblings))
    }

    fn get_matrices<'a, M: Matrix<PF::Value>>(&self, tree: &'a Self::ProverData<M>) -> Vec<&'a M> {
        tree.leaves.iter().collect()
    }

    /// Verify a single-leaf opening against `commitment`.
    ///
    /// Security notes:
    /// - `dimensions.width` is interpreted as the committed row length (including any
    ///   alignment padding if `build_aligned_tree` was used); LMCS does not enforce
    ///   that padded values are zero.
    /// - `dimensions` must match the commitment; out-of-range `index` returns
    ///   `InvalidProof`.
    /// - Returns `InvalidProof` for malformed rows/siblings, `RootMismatch` for a
    ///   well-formed proof to a different root.
    fn verify_batch(
        &self,
        commitment: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, PF::Value, Self>,
    ) -> Result<(), Self::Error> {
        let (rows, (salt, siblings)) = batch_opening.unpack();

        // Convert dimensions to widths + log_max_height for internal use.
        let widths: Vec<usize> = dimensions.iter().map(|d| d.width).collect();
        if batch_opening.opened_values.len() != widths.len() {
            return Err(LmcsError::InvalidProof);
        }
        for (row, &width) in zip(batch_opening.opened_values, &widths) {
            if row.len() != width {
                return Err(LmcsError::InvalidProof);
            }
        }

        let leaf_hash = self.hash(
            rows.iter()
                .map(|row| row.as_slice())
                .chain(core::iter::once(salt.as_slice())),
        );

        let max_height = dimensions
            .iter()
            .map(|d| d.height)
            .max()
            .ok_or(LmcsError::InvalidProof)?;
        let log_max_height = log2_ceil_usize(max_height);
        if siblings.len() != log_max_height {
            return Err(LmcsError::InvalidProof);
        }

        if index >= max_height {
            return Err(LmcsError::InvalidProof);
        }

        let computed_commitment = {
            let mut current = leaf_hash;
            let mut pos = index;

            for sibling_hash in siblings {
                let is_left = pos & 1 == 0;
                current = if is_left {
                    self.compress(current, *sibling_hash)
                } else {
                    self.compress(*sibling_hash, current)
                };
                pos >>= 1;
            }

            current
        };

        if computed_commitment != *commitment {
            return Err(LmcsError::RootMismatch);
        }

        Ok(())
    }
}
