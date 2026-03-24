//! Sorted, deduplicated leaf indices for LMCS batch openings with a fixed tree height.

use alloc::{collections::BTreeSet, vec::Vec};

use crate::LmcsError;

/// Unique query indices in ascending order, all strictly less than `2^log_max_height`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortedTreeIndices {
    log_max_height: u8,
    indices: BTreeSet<usize>,
}

impl SortedTreeIndices {
    /// Dedupe, sort, and validate `index < 2^log_max_height` for every index.
    ///
    /// Empty iterators are allowed (for parse-only batch proofs with no openings).
    pub fn try_new(
        indices: impl IntoIterator<Item = usize>,
        log_max_height: u8,
    ) -> Result<Self, LmcsError> {
        let max_height = 1usize
            .checked_shl(log_max_height as u32)
            .ok_or(LmcsError::InvalidProof)?;
        let indices: BTreeSet<usize> = indices.into_iter().collect();
        if indices.iter().any(|&i| i >= max_height) {
            return Err(LmcsError::InvalidProof);
        }

        Ok(Self {
            log_max_height,
            indices,
        })
    }

    #[inline]
    pub fn log_max_height(&self) -> u8 {
        self.log_max_height
    }

    #[inline]
    pub fn max_height(&self) -> usize {
        1usize << self.log_max_height as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    pub fn indices(&self) -> &BTreeSet<usize> {
        &self.indices
    }

    /// Visit each `(depth, sibling_index)` in canonical hint order (level-by-level,
    /// left-to-right, bottom-to-top).
    ///
    /// Matches the order in which [`LmcsTree::prove_batch`](crate::LmcsTree::prove_batch) writes
    /// sibling hints and [`fold_pruned_opening_root`](crate::fold_pruned_opening_root) requests
    /// them when recomputing the root.
    pub fn try_for_each_missing_sibling<E>(
        &self,
        mut f: impl FnMut(usize, usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let log_max_height = self.log_max_height as usize;
        let mut known: BTreeSet<usize> = self.indices.clone();

        for current_depth in 0..log_max_height {
            let mut parents = BTreeSet::new();

            for &pos in &known {
                let parent_pos = pos / 2;
                if !parents.insert(parent_pos) {
                    continue;
                }

                let left_pos = parent_pos * 2;
                let right_pos = left_pos + 1;
                let have_left = known.contains(&left_pos);
                let have_right = known.contains(&right_pos);

                let missing_pos = match (have_left, have_right) {
                    (true, false) => right_pos,
                    (false, true) => left_pos,
                    _ => continue,
                };

                f(current_depth, missing_pos)?;
            }

            known = parents;
        }

        Ok(())
    }

    /// Infallible variant of [`Self::try_for_each_missing_sibling`].
    pub fn for_each_missing_sibling(&self, mut f: impl FnMut(usize, usize)) {
        let _ = self.try_for_each_missing_sibling(|d, i| {
            f(d, i);
            Ok::<_, core::convert::Infallible>(())
        });
    }

    /// Sibling positions that must be supplied as transcript hints, in canonical order:
    /// level-by-level, left-to-right, bottom-to-top.
    ///
    /// Matches the order used by [`LmcsTree::prove_batch`](crate::LmcsTree::prove_batch) and
    /// consumed by [`Lmcs::open_batch`](crate::Lmcs::open_batch).
    pub fn missing_sibling_positions(&self) -> Vec<(usize, usize)> {
        let mut missing = Vec::new();
        self.for_each_missing_sibling(|d, i| missing.push((d, i)));
        missing
    }
}
