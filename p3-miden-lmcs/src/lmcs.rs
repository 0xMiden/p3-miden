//! LMCS configuration types.

use alloc::{collections::BTreeMap, vec::Vec};
use core::marker::PhantomData;

use p3_field::PackedValue;
use p3_matrix::Matrix;
use p3_miden_stateful_hasher::{Alignable, StatefulHasher};
use p3_miden_transcript::VerifierChannel;
use p3_symmetric::{Hash, PseudoCompressionFunction};

use crate::{
    LeafOpening, LiftedMerkleTree, Lmcs, LmcsError, OpenedRows, PrunedTree, SortedTreeIndices,
    fold_pruned_opening_root, utils::RowList,
};

type Opening<F, C> = (RowList<F>, C);

/// LMCS configuration holding cryptographic primitives (sponge + compression).
///
/// This implementation defines the transcript hint layout used by
/// [`LmcsTree::prove_batch`](crate::LmcsTree::prove_batch) and consumed by
/// `open_batch` and [`PrunedTree::read_from_channel`](crate::PrunedTree::read_from_channel):
/// - For each *distinct* query index (in caller order, skipping duplicates): one row per
///   matrix (in leaf order), then `SALT_ELEMS` field elements of salt.
/// - After all indices: missing sibling hashes, level-by-level, left-to-right, bottom-to-top.
///
/// Hints are not observed into the Fiat-Shamir challenger.
///
/// `open_batch` expects `widths` and `log_max_height` to match the committed tree,
/// rejects empty `indices`, and ignores extra hint data. Widths must match the
/// committed row lengths (including any alignment padding if `build_aligned_tree`
/// was used). Duplicate indices are coalesced in the returned openings.
/// [`PrunedTree::read_from_channel`](crate::PrunedTree::read_from_channel) parses
/// the same hint stream without hashing, and [`PrunedTree::single_proofs`](crate::PrunedTree::single_proofs)
/// can reconstruct per-index proofs (keyed by index) without verifying against a commitment. Empty indices
/// yield an empty [`PrunedTree`], and out-of-range indices return [`LmcsError::InvalidProof`] at parse time.
///
/// Padding note:
/// - LMCS does not enforce that aligned padding values are zero. Verifiers cannot
///   distinguish zero padding from arbitrary values unless they check those columns
///   in the opened rows or constrain them elsewhere.
///
/// For hiding commitments with salt, use [`crate::HidingLmcsConfig`] instead.
#[derive(Clone, Debug)]
pub struct LmcsConfig<
    PF,
    PD,
    H,
    C,
    const WIDTH: usize,
    const DIGEST: usize,
    const SALT_ELEMS: usize = 0,
> {
    /// Stateful sponge for hashing matrix rows into leaf hashes.
    pub sponge: H,
    /// 2-to-1 compression function for building internal tree nodes.
    pub compress: C,
    pub(crate) _phantom: PhantomData<(PF, PD)>,
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST: usize, const SALT_ELEMS: usize>
    LmcsConfig<PF, PD, H, C, WIDTH, DIGEST, SALT_ELEMS>
{
    /// Create a new LMCS configuration.
    #[inline]
    pub const fn new(sponge: H, compress: C) -> Self {
        Self {
            sponge,
            compress,
            _phantom: PhantomData,
        }
    }
}

impl<PF, PD, H, C, const WIDTH: usize, const DIGEST: usize, const SALT_ELEMS: usize> Lmcs
    for LmcsConfig<PF, PD, H, C, WIDTH, DIGEST, SALT_ELEMS>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    H: StatefulHasher<PF::Value, [PD::Value; DIGEST], State = [PD::Value; WIDTH]>
        + StatefulHasher<PF, [PD; DIGEST], State = [PD; WIDTH]>
        + Alignable<PF::Value, PD::Value>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST], 2>
        + PseudoCompressionFunction<[PD; DIGEST], 2>
        + Sync,
{
    type F = PF::Value;
    type Commitment = Hash<PF::Value, PD::Value, DIGEST>;
    type PrunedTree = crate::proof::PrunedTree<PF::Value, Self::Commitment, SALT_ELEMS>;
    type Tree<M: Matrix<PF::Value>> = LiftedMerkleTree<PF::Value, PD::Value, M, DIGEST, SALT_ELEMS>;

    /// Build a tree from matrices with no transcript padding (alignment = 1).
    ///
    /// Preconditions:
    /// - `leaves` is non-empty.
    /// - Matrix heights are powers of two and sorted by height (shortest to tallest).
    ///
    /// Panics if `leaves` is empty. Incorrect height order commits to a different
    /// lifted matrix than intended.
    fn build_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M> {
        const { assert!(SALT_ELEMS == 0) }
        LiftedMerkleTree::build_with_alignment::<PF, PD, H, C, WIDTH>(
            &self.sponge,
            &self.compress,
            leaves,
            None,
            1,
        )
    }

    /// Build a tree from matrices using the hasher alignment for transcript padding.
    ///
    /// Preconditions:
    /// - `leaves` is non-empty.
    /// - Matrix heights are powers of two and sorted by height (shortest to tallest).
    ///
    /// Panics if `leaves` is empty. Incorrect height order commits to a different
    /// lifted matrix than intended.
    fn build_aligned_tree<M: Matrix<Self::F>>(&self, leaves: Vec<M>) -> Self::Tree<M> {
        const { assert!(SALT_ELEMS == 0) }
        LiftedMerkleTree::build_with_alignment::<PF, PD, H, C, WIDTH>(
            &self.sponge,
            &self.compress,
            leaves,
            None,
            <H as Alignable<PF::Value, PD::Value>>::ALIGNMENT,
        )
    }

    fn hash<'a, I>(&self, rows: I) -> Self::Commitment
    where
        I: IntoIterator<Item = &'a [Self::F]>,
        Self::F: 'a,
    {
        let mut state = [PD::Value::default(); WIDTH];
        for row in rows {
            self.sponge.absorb_into(&mut state, row.iter().cloned());
        }
        let digest: [PD::Value; DIGEST] = self.sponge.squeeze(&state);
        Hash::from(digest)
    }

    fn compress(&self, left: Self::Commitment, right: Self::Commitment) -> Self::Commitment {
        let left_digest = *left.as_ref();
        let right_digest = *right.as_ref();
        Hash::from(self.compress.compress([left_digest, right_digest]))
    }

    /// Verify a batch opening from transcript hints.
    ///
    /// Security notes:
    /// - `widths` and `log_max_height` must describe the committed tree; they are not checked.
    /// - `widths` must match the committed row lengths (including any alignment padding
    ///   if `build_aligned_tree` was used); LMCS does not enforce that padded values are
    ///   zero. Verifiers cannot distinguish zero padding from arbitrary values unless
    ///   they check the opened rows or constrain them elsewhere.
    /// - Empty `indices` returns `InvalidProof`.
    /// - Duplicate indices are coalesced in the returned map (unique keys only).
    /// - Out-of-range indices (>= 2^log_max_height) return `InvalidProof`.
    /// - Missing siblings or malformed hints return `InvalidProof`.
    /// - Extra hints are ignored and left unread.
    /// - Returns `RootMismatch` only after a well-formed proof yields a different root.
    ///
    /// Leaf openings are read in **sorted tree index order** (ascending, deduplicated).
    fn open_batch<Ch>(
        &self,
        commitment: &Self::Commitment,
        widths: &[usize],
        log_max_height: u8,
        indices: impl IntoIterator<Item = usize>,
        channel: &mut Ch,
    ) -> Result<OpenedRows<Self::F>, LmcsError>
    where
        Ch: VerifierChannel<F = Self::F, Commitment = Self::Commitment>,
    {
        let sorted = SortedTreeIndices::try_new(indices, log_max_height)?;
        if sorted.is_empty() {
            return Err(LmcsError::InvalidProof);
        }

        // Map index -> (rows, leaf_hash), filled in sorted order.
        let mut openings_by_index: BTreeMap<usize, Opening<Self::F, Self::Commitment>> =
            BTreeMap::new();

        // Read openings in sorted tree index order.
        for index in sorted.indices().iter().copied() {
            let LeafOpening { rows, salt } =
                LeafOpening::<Self::F, SALT_ELEMS>::receive_from_hints(channel, widths)?;

            // Recompute leaf hash from opened data to verify against the Merkle commitment.
            let leaf_hash = if SALT_ELEMS > 0 {
                self.hash(rows.iter_rows().chain([salt.as_slice()]))
            } else {
                self.hash(rows.iter_rows())
            };

            openings_by_index.insert(index, (rows, leaf_hash));
        }

        // Recompute root via [`fold_pruned_opening_root`]; hinted siblings are consumed in the
        // same order as [`SortedTreeIndices::missing_sibling_positions`].
        let children: Vec<(usize, Self::Commitment)> = openings_by_index
            .iter()
            .map(|(&index, (_, hash))| (index, *hash))
            .collect();
        let computed_commitment = fold_pruned_opening_root(
            log_max_height,
            children,
            |_level, _sibling_pos| Ok(*channel.receive_hint_commitment()?),
            |left, right| self.compress(left, right),
        )?;

        // Compare against commitment.
        if computed_commitment != *commitment {
            return Err(LmcsError::RootMismatch);
        }

        // Return deduplicated openings keyed by index.
        Ok(openings_by_index
            .into_iter()
            .map(|(idx, (rows, _hash))| (idx, rows))
            .collect())
    }

    /// Parse batch hints without hashing.
    ///
    /// Notes:
    /// - `widths` and `log_max_height` are trusted parameters.
    /// - `widths` must match the committed row lengths (including any alignment padding
    ///   if `build_aligned_tree` was used).
    /// - Out-of-range indices return [`LmcsError::InvalidProof`]. Empty `indices` yield an
    ///   empty [`PrunedTree`](crate::PrunedTree).
    fn read_pruned_tree_from_channel<Ch>(
        &self,
        widths: &[usize],
        log_max_height: u8,
        indices: &[usize],
        channel: &mut Ch,
    ) -> Result<Self::PrunedTree, LmcsError>
    where
        Ch: VerifierChannel<F = Self::F, Commitment = Self::Commitment>,
    {
        PrunedTree::read_from_channel(widths, log_max_height, indices, channel)
    }

    fn alignment(&self) -> usize {
        <H as Alignable<PF::Value, PD::Value>>::ALIGNMENT
    }
}
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use alloc::vec;

    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
    use p3_miden_transcript::{ProverTranscript, TranscriptData, VerifierTranscript};

    use crate::{Lmcs, LmcsConfig, LmcsError, LmcsTree, log2_strict_u8};

    type TestLmcs =
        LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;

    fn small_matrix(height: usize, width: usize, seed: u64) -> RowMajorMatrix<bb::F> {
        let values = (0..height * width)
            .map(|i| bb::F::from_u64(seed + i as u64))
            .collect();
        RowMajorMatrix::new(values, width)
    }

    #[test]
    fn open_batch_cases() {
        let (_, sponge, compress) = bb::test_components();
        let lmcs: TestLmcs = LmcsConfig::new(sponge, compress);
        let matrices = vec![small_matrix(4, 2, 0), small_matrix(4, 3, 100)];
        let tree = lmcs.build_tree(matrices);
        let widths = tree.widths();
        let log_max_height = log2_strict_u8(tree.height());
        let commitment = tree.root();

        let make_transcript = |indices: &[usize]| {
            let mut prover_channel = ProverTranscript::new(bb::test_challenger());
            tree.prove_batch(indices.iter().copied(), &mut prover_channel);
            prover_channel.finalize()
        };

        let assert_open = |indices: &[usize]| {
            let (prover_digest, transcript) = make_transcript(indices);
            let mut verifier_channel =
                VerifierTranscript::from_data(bb::test_challenger(), &transcript);
            let opened = lmcs
                .open_batch(
                    &commitment,
                    &widths,
                    log_max_height,
                    indices.iter().copied(),
                    &mut verifier_channel,
                )
                .unwrap();
            for &idx in indices {
                assert_eq!(opened[&idx], tree.rows(idx));
            }
            let verifier_digest = verifier_channel
                .finalize()
                .expect("transcript should finalize cleanly");
            assert_eq!(prover_digest, verifier_digest);
        };

        assert_open(&[0]);
        assert_open(&[0, 1]);
        assert_open(&[0, 2]);
        assert_open(&[0, 1, 2, 3]);
        assert_open(&[2, 2]);

        let tiny_tree = lmcs.build_tree(vec![small_matrix(1, 1, 7)]);
        let widths_tiny = tiny_tree.widths();
        let log_tiny = log2_strict_u8(tiny_tree.height());
        let mut prover_channel = ProverTranscript::new(bb::test_challenger());
        tiny_tree.prove_batch([0], &mut prover_channel);
        let (prover_digest, transcript) = prover_channel.finalize();
        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        let opened = lmcs
            .open_batch(
                &tiny_tree.root(),
                &widths_tiny,
                log_tiny,
                [0],
                &mut verifier_channel,
            )
            .unwrap();
        assert_eq!(opened[&0], tiny_tree.rows(0));
        let verifier_digest = verifier_channel
            .finalize()
            .expect("transcript should finalize cleanly");
        assert_eq!(prover_digest, verifier_digest);

        // oob index
        let (_, transcript) = ProverTranscript::new(bb::test_challenger()).finalize();
        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        assert_eq!(
            lmcs.open_batch(
                &commitment,
                &widths,
                log_max_height,
                [tree.height()],
                &mut verifier_channel,
            ),
            Err(LmcsError::InvalidProof)
        );

        // wrong tree
        let (_, transcript) = make_transcript(&[0]);
        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        let wrong_tree = lmcs.build_tree(vec![small_matrix(4, 2, 999)]);
        assert_eq!(
            lmcs.open_batch(
                &wrong_tree.root(),
                &widths,
                log_max_height,
                [0],
                &mut verifier_channel,
            ),
            Err(LmcsError::RootMismatch)
        );

        // missing item from transcript
        let indices = [0usize];
        let (_, transcript) = make_transcript(&indices);
        let (fields, mut commitments) = transcript.into_parts();
        commitments.pop();
        let truncated = TranscriptData::new(fields, commitments);
        let mut verifier_channel = VerifierTranscript::from_data(bb::test_challenger(), &truncated);
        assert_eq!(
            lmcs.open_batch(
                &commitment,
                &widths,
                log_max_height,
                indices,
                &mut verifier_channel,
            ),
            Err(LmcsError::TranscriptError(
                p3_miden_transcript::TranscriptError::NoMoreCommitments
            ))
        );

        // empty indices
        let (_, transcript) = ProverTranscript::new(bb::test_challenger()).finalize();
        let mut verifier_channel =
            VerifierTranscript::from_data(bb::test_challenger(), &transcript);
        assert_eq!(
            lmcs.open_batch(
                &commitment,
                &widths,
                log_max_height,
                [],
                &mut verifier_channel,
            ),
            Err(LmcsError::InvalidProof)
        );
    }

    /// Reproduces the "root mismatch" bug when using Goldilocks + Blake3 (byte-based hash).
    ///
    /// The lifted STARK only tests with field-based Poseidon2, never with byte-based hashes.
    /// This test isolates the LMCS layer to confirm that ChainingHasher<Blake3> +
    /// CompressionFunctionFromHasher<Blake3> work correctly for commit-then-open.
    #[test]
    fn goldilocks_blake3_roundtrip() {
        use alloc::{vec, vec::Vec};

        use p3_blake3::Blake3;
        use p3_challenger::{HashChallenger, SerializingChallenger64};
        use p3_goldilocks::Goldilocks;
        use p3_miden_stateful_hasher::ChainingHasher;
        use p3_symmetric::CompressionFunctionFromHasher;

        type F = Goldilocks;
        type Sponge = ChainingHasher<Blake3>;
        type Compress = CompressionFunctionFromHasher<Blake3, 2, 32>;
        const WIDTH: usize = 32;
        const DIGEST: usize = 32;
        type Blake3Lmcs = LmcsConfig<F, u8, Sponge, Compress, WIDTH, DIGEST>;
        type Challenger = SerializingChallenger64<F, HashChallenger<u8, Blake3, 32>>;

        fn challenger() -> Challenger {
            SerializingChallenger64::from_hasher(vec![], Blake3)
        }

        let sponge = ChainingHasher::new(Blake3);
        let compress = CompressionFunctionFromHasher::new(Blake3);
        let lmcs: Blake3Lmcs = LmcsConfig::new(sponge, compress);

        // Single 4x2 matrix of constant values.
        let values: Vec<F> = (0..4 * 2).map(|i| F::from_u64(i as u64)).collect();
        let matrix = RowMajorMatrix::new(values, 2);

        let tree = lmcs.build_tree(vec![matrix]);
        let widths = tree.widths();
        let log_max_height = log2_strict_u8(tree.height());
        let commitment = tree.root();

        // Prove then verify a single index.
        let mut prover_channel = ProverTranscript::new(challenger());
        tree.prove_batch([0usize], &mut prover_channel);
        let (prover_digest, transcript) = prover_channel.finalize();

        let mut verifier_channel = VerifierTranscript::from_data(challenger(), &transcript);
        let opened = lmcs
            .open_batch(
                &commitment,
                &widths,
                log_max_height,
                [0usize],
                &mut verifier_channel,
            )
            .expect("Goldilocks+Blake3 LMCS roundtrip should verify");

        assert_eq!(opened[&0], tree.rows(0));
        let verifier_digest = verifier_channel
            .finalize()
            .expect("transcript should finalize cleanly");
        assert_eq!(prover_digest, verifier_digest);
    }
}
