//! Upward fold for LMCS-style pruned Merkle trees (batch openings).

use alloc::vec::Vec;

use crate::LmcsError;

/// Recompute the Merkle root from sorted leaf `(position, digest)` pairs and hinted siblings.
///
/// `children` must be sorted by leaf index ascending (unique indices). At each level, when
/// a node's sibling digest is not the next entry in the list, `missing_sibling` supplies it.
/// Calls to `missing_sibling(level, sibling_index)` occur in the same order as
/// [`SortedTreeIndices::missing_sibling_positions`](crate::SortedTreeIndices::missing_sibling_positions).
pub fn fold_pruned_opening_root<C: Copy>(
    log_max_height: u8,
    mut children: Vec<(usize, C)>,
    mut missing_sibling: impl FnMut(usize, usize) -> Result<C, LmcsError>,
    mut compress: impl FnMut(C, C) -> C,
) -> Result<C, LmcsError> {
    for level in 0..log_max_height as usize {
        let mut parents = Vec::with_capacity(children.len());
        let mut children_iter = children.iter().peekable();

        while let Some((child_position, child_hash)) = children_iter.next() {
            let sibling_position = child_position ^ 1;
            let sibling_hash = match children_iter.next_if(|(pos, _)| *pos == sibling_position) {
                Some((_, hash)) => *hash,
                None => missing_sibling(level, sibling_position)?,
            };

            let child_is_left = child_position & 1 == 0;
            let (left_hash, right_hash) = if child_is_left {
                (*child_hash, sibling_hash)
            } else {
                (sibling_hash, *child_hash)
            };

            let parent_hash = compress(left_hash, right_hash);
            let parent_position = child_position >> 1;
            parents.push((parent_position, parent_hash));
        }

        children = parents;
    }

    match children.as_slice() {
        [(0, root)] => Ok(*root),
        _ => Err(LmcsError::InvalidProof),
    }
}
