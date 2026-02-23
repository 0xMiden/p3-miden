//! MMCS implementation for `HidingLmcsConfig`.
//!
//! This wrapper only changes how trees are built (salted leaves). Once a tree
//! exists, MMCS openings are identical to the non-hiding case, so `open_batch`,
//! `get_matrices`, and `verify_batch` delegate to the inner `LmcsConfig`.
//! Delegation keeps proof shape and validation consistent and avoids consulting
//! the RNG during opening or verification.

use alloc::vec::Vec;

use p3_commit::{BatchOpening, BatchOpeningRef, Mmcs};
use p3_field::PackedValue;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_stateful_hasher::{Alignable, StatefulHasher};
use p3_symmetric::{Hash, PseudoCompressionFunction};
use rand::Rng;
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use crate::lifted_tree::LiftedMerkleTree;
use crate::{HidingLmcsConfig, Lmcs, LmcsError, LmcsTree};

// ============================================================================
// Mmcs implementation for HidingLmcsConfig
// ============================================================================

impl<PF, PD, H, C, R, const WIDTH: usize, const DIGEST_ELEMS: usize, const SALT_ELEMS: usize>
    Mmcs<PF::Value> for HidingLmcsConfig<PF, PD, H, C, R, WIDTH, DIGEST_ELEMS, SALT_ELEMS>
where
    PF: PackedValue + Default,
    PD: PackedValue + Default,
    PF::Value: PartialEq,
    R: Rng + Clone,
    StandardUniform: Distribution<PF::Value>,
    H: StatefulHasher<PF, [PD; DIGEST_ELEMS], State = [PD; WIDTH]>
        + StatefulHasher<PF::Value, [PD::Value; DIGEST_ELEMS], State = [PD::Value; WIDTH]>
        + Alignable<PF::Value, PD::Value>
        + Sync,
    C: PseudoCompressionFunction<[PD::Value; DIGEST_ELEMS], 2>
        + PseudoCompressionFunction<[PD; DIGEST_ELEMS], 2>
        + Sync,
    [PD::Value; DIGEST_ELEMS]: Serialize + for<'de> Deserialize<'de>,
    [PF::Value; SALT_ELEMS]: Serialize + for<'de> Deserialize<'de>,
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
        let BatchOpening {
            opened_values,
            opening_proof,
        } = Mmcs::open_batch(&self.inner, index, tree);
        BatchOpening {
            opened_values,
            opening_proof,
        }
    }

    fn get_matrices<'a, M: Matrix<PF::Value>>(&self, tree: &'a Self::ProverData<M>) -> Vec<&'a M> {
        self.inner.get_matrices(tree)
    }

    fn verify_batch(
        &self,
        commitment: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        batch_opening: BatchOpeningRef<'_, PF::Value, Self>,
    ) -> Result<(), Self::Error> {
        let batch_opening = BatchOpeningRef {
            opened_values: batch_opening.opened_values,
            opening_proof: batch_opening.opening_proof,
        };
        self.inner
            .verify_batch(commitment, dimensions, index, batch_opening)
    }
}
