//! Single-opening proof structures and transcript parsing helpers.
//!
//! - [`Proof`]: Single-opening proof with rows, optional salt, and authentication path.
//! - [`BatchProof`]: Parsed batch opening containing rows/salt plus hinted siblings.
//! - [`LeafOpening::receive_from_hints`]: Read one leaf’s rows + salt from hint fields.
//!
//! For batched openings via transcript hints in this crate, see
//! [`LmcsConfig`](crate::LmcsConfig) and [`LiftedMerkleTree`](crate::LiftedMerkleTree).
//! [`BatchProof`] parses hints without hashing, and can be turned into per-index
//! [`Proof`] objects once the hashing context is available.

use alloc::{collections::BTreeMap, vec::Vec};

use p3_miden_transcript::VerifierChannel;
use serde::{Deserialize, Serialize};

use crate::{Lmcs, LmcsError, SortedTreeIndices, utils::RowList};

/// Single-opening Merkle proof with rows and authentication path.
///
/// Contains the opening (rows + salt) and siblings (bottom-to-top) for a single leaf.
///
/// # Type Parameters
///
/// - `F`: Field element type.
/// - `C`: Hash type (also used for commitments).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, C: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, C: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct Proof<F, C, const SALT_ELEMS: usize = 0> {
    /// Opened rows for this query.
    pub rows: RowList<F>,
    /// Salt for this leaf (zero-sized when the configuration is non-hiding).
    pub salt: [F; SALT_ELEMS],
    /// Sibling hashes from leaf level to root (bottom-to-top).
    pub siblings: Vec<C>,
}

/// Opened rows and optional salt for a single leaf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct LeafOpening<F, const SALT_ELEMS: usize = 0> {
    /// Opened rows for this query.
    pub rows: RowList<F>,
    /// Salt for this leaf (zero-sized when the configuration is non-hiding).
    pub salt: [F; SALT_ELEMS],
}

impl<F: Copy, const SALT_ELEMS: usize> LeafOpening<F, SALT_ELEMS> {
    /// Read one leaf opening from transcript hints:
    /// concatenated row field elements then `SALT_ELEMS` salt elements.
    ///
    /// When `SALT_ELEMS == 0`, the salt read is a no-op.
    pub fn receive_from_hints<Ch>(channel: &mut Ch, widths: &[usize]) -> Result<Self, LmcsError>
    where
        Ch: VerifierChannel<F = F>,
    {
        let total_width: usize = widths.iter().sum();
        let elems = channel.receive_hint_field_slice(total_width)?.to_vec();
        let rows = RowList::new(elems, widths);
        let salt: [F; SALT_ELEMS] = channel.receive_hint_field_array()?;
        Ok(Self { rows, salt })
    }
}

/// Batch opening parsed from transcript hints, without hashing.
///
/// Stores per-index openings plus the hinted siblings needed to reconstruct
/// authentication paths. Siblings are indexed by `(depth, index)` where depth 0
/// is the leaf level.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, C: Serialize, [F; SALT_ELEMS]: Serialize",
    deserialize = "F: Deserialize<'de>, C: Deserialize<'de>, [F; SALT_ELEMS]: Deserialize<'de>"
))]
pub struct BatchProof<F, C, const SALT_ELEMS: usize = 0> {
    /// Openings keyed by leaf index.
    pub openings: BTreeMap<usize, LeafOpening<F, SALT_ELEMS>>,
    /// Hinted sibling hashes keyed by `(depth, index)`.
    pub siblings: BTreeMap<(usize, usize), C>,
}

impl<F, C, const SALT_ELEMS: usize> BatchProof<F, C, SALT_ELEMS> {
    /// Parse a batch opening from a transcript channel without validation.
    ///
    /// This is a parse-only function: it reads rows, salts, and sibling hashes from
    /// the channel but does **not** hash leaves or verify against a commitment.
    /// The returned proof may be invalid if the inputs (indices, widths, or channel
    /// contents) are themselves invalid — validation happens in
    /// [`open_batch`](crate::Lmcs::open_batch).
    ///
    /// Reads unique queried indices in sorted (ascending) order, matching the order
    /// in which [`LmcsTree::prove_batch`](crate::LmcsTree::prove_batch) writes them.
    /// Allocations are O(n · log_max_height) regardless of index values, where n is
    /// the number of unique indices.
    ///
    /// Returns [`LmcsError::InvalidProof`] if any index is `>= 2^log_max_height` or if
    /// `log_max_height` is too large for the platform word size.
    pub fn read_from_channel<Ch>(
        widths: &[usize],
        log_max_height: u8,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Self, LmcsError>
    where
        F: Copy,
        C: Clone + PartialEq,
        Ch: VerifierChannel<F = F, Commitment = C>,
    {
        let sorted = SortedTreeIndices::try_new(indices.iter().copied(), log_max_height)?;
        // Read openings in sorted order, matching prove_batch's write order.
        let openings: BTreeMap<usize, LeafOpening<F, SALT_ELEMS>> = sorted
            .indices()
            .iter()
            .copied()
            .map(|index| Ok((index, LeafOpening::receive_from_hints(channel, widths)?)))
            .collect::<Result<_, LmcsError>>()?;

        // Consume sibling hints in the same canonical order the prover emits them.
        let siblings: BTreeMap<(usize, usize), C> = sorted
            .missing_sibling_positions()
            .into_iter()
            .map(|key| Ok((key, channel.receive_hint_commitment()?.clone())))
            .collect::<Result<_, LmcsError>>()?;

        Ok(Self { openings, siblings })
    }

    /// Reconstruct per-index proofs by hashing rows/salt and rebuilding paths.
    ///
    /// Returns a map keyed by leaf index; duplicate indices are coalesced.
    /// Does not verify against a commitment.
    ///
    /// Assumes all indices are in `0..2^log_max_height`; returns `None` if
    /// widths mismatch or sibling reconstruction fails.
    pub fn single_proofs<L>(
        &self,
        lmcs: &L,
        widths: &[usize],
        log_max_height: u8,
    ) -> Option<BTreeMap<usize, Proof<F, C, SALT_ELEMS>>>
    where
        F: Copy,
        C: Clone + PartialEq,
        L: Lmcs<F = F, Commitment = C>,
    {
        let mut proofs: BTreeMap<usize, Proof<F, C, SALT_ELEMS>> = BTreeMap::new();
        // Track known nodes by (depth, index) to reconstruct sibling paths deterministically.
        let mut tree: BTreeMap<(usize, usize), C> = BTreeMap::new();

        for (&index, opening) in self.openings.iter() {
            if opening.rows.num_rows() != widths.len() {
                return None;
            }
            for (row, &width) in opening.rows.iter_rows().zip(widths.iter()) {
                if row.len() != width {
                    return None;
                }
            }

            let rows_iter = opening.rows.iter_rows();
            let leaf_hash = if SALT_ELEMS > 0 {
                lmcs.hash(rows_iter.chain([opening.salt.as_slice()]))
            } else {
                lmcs.hash(rows_iter)
            };

            proofs.entry(index).or_insert_with(|| Proof {
                rows: opening.rows.clone(),
                salt: opening.salt,
                siblings: Vec::with_capacity(log_max_height as usize),
            });

            // Reject if two openings claim different data for the same leaf.
            if tree
                .insert((0, index), leaf_hash.clone())
                .is_some_and(|existing_hash| existing_hash != leaf_hash)
            {
                return None;
            }
        }

        // Preload hinted siblings so combining pairs can assume adjacency.
        let tree_depth = log_max_height as usize;
        let sorted =
            SortedTreeIndices::try_new(self.openings.keys().copied(), log_max_height).ok()?;
        for (depth, index) in sorted.missing_sibling_positions() {
            tree.insert((depth, index), self.siblings.get(&(depth, index))?.clone());
        }

        for current_depth in 0..tree_depth {
            // BTreeMap ordering yields left-to-right pairing at this depth.
            let nodes_at_depth: Vec<(usize, C)> = tree
                .range((current_depth, 0)..=(current_depth, usize::MAX))
                .map(|(&(_, idx), hash)| (idx, hash.clone()))
                .collect();

            let mut nodes_iter = nodes_at_depth.into_iter().peekable();
            while let Some((index, hash)) = nodes_iter.next() {
                let sibling_index = index ^ 1;
                let sibling_hash =
                    match nodes_iter.next_if(|(next_index, _)| *next_index == sibling_index) {
                        Some((_, hash)) => hash,
                        None => return None,
                    };

                let is_left_child = index & 1 == 0;
                let (left, right) = if is_left_child {
                    (hash, sibling_hash)
                } else {
                    (sibling_hash, hash)
                };

                let parent_depth = current_depth + 1;
                let parent_index = index / 2;
                let parent_hash = lmcs.compress(left, right);
                tree.insert((parent_depth, parent_index), parent_hash);
            }
        }

        // Add authentication paths from the reconstructed tree.
        for (&index, proof) in proofs.iter_mut() {
            let mut current_index = index;
            for current_depth in 0..tree_depth {
                let sibling_index = current_index ^ 1;
                let sibling_hash = tree.get(&(current_depth, sibling_index)).cloned()?;
                proof.siblings.push(sibling_hash);
                current_index >>= 1;
            }
        }

        Some(proofs)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use p3_matrix::dense::RowMajorMatrix;
    use p3_miden_transcript::{VerifierChannel, VerifierTranscript};
    use p3_symmetric::Hash;
    use rand::{SeedableRng, rngs::SmallRng};

    use crate::{
        BatchProof, Lmcs, LmcsTree, log2_strict_u8,
        tests::{DIGEST, F, lmcs, roundtrip_open_batch},
    };

    #[test]
    fn batch_proof_consistent_with_open_batch() {
        let lmcs = lmcs();

        let test = |seed: u64, shapes: &[(usize, usize)], indices: &[usize]| {
            let mut rng = SmallRng::seed_from_u64(seed);
            let matrices: Vec<_> = shapes
                .iter()
                .map(|&(h, w)| RowMajorMatrix::rand(&mut rng, h, w))
                .collect();
            let tree = lmcs.build_tree(matrices);
            let widths = tree.widths();
            let log_max_height = log2_strict_u8(tree.height());

            // Path A: open_batch (verification)
            let (transcript, opened_rows) =
                roundtrip_open_batch(&lmcs, &tree, indices).expect("open_batch should verify");

            // Path B: BatchProof::read_from_channel (parse-only)
            let mut verifier_channel = VerifierTranscript::from_data(
                p3_miden_dev_utils::configs::baby_bear_poseidon2::test_challenger(),
                &transcript,
            );
            let batch = BatchProof::<F, Hash<F, F, DIGEST>>::read_from_channel(
                &widths,
                log_max_height,
                indices,
                &mut verifier_channel,
            )
            .expect("batch proof should parse");
            assert!(
                verifier_channel.is_empty(),
                "parse path should fully consume transcript"
            );

            // Same number of unique openings
            assert_eq!(opened_rows.len(), batch.openings.len());

            // Row data must match between the two paths
            for (&idx, verified_rows) in &opened_rows {
                let parsed = batch.openings.get(&idx).expect("parsed opening for index");
                assert_eq!(
                    *verified_rows, parsed.rows,
                    "row mismatch between open_batch and BatchProof at index {idx}"
                );
            }

            // Reconstructed single proofs must match tree's own single_proof
            let proofs = batch
                .single_proofs(&lmcs, &widths, log_max_height)
                .expect("single_proofs should reconstruct");
            for &idx in indices {
                let proof = proofs.get(&idx).expect("proof for index");
                let expected = tree.single_proof(idx);
                assert_eq!(proof, &expected, "single_proof mismatch at index {idx}");
            }
        };

        test(1, &[(8, 4)], &[0, 3, 7]);
        test(42, &[(4, 3), (8, 5), (16, 7)], &[0, 5, 10, 15]);
        test(99, &[(4, 2), (8, 6)], &[3, 1, 3, 0, 1]); // duplicates
    }
}
