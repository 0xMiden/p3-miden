# p3-miden-lmcs

Lifted Matrix Commitment Scheme (LMCS) for committing to multiple evaluation
matrices with power-of-two heights, using a uniform lifted view in bit-reversed
order.

## Overview

- Commit to matrices of different heights via virtual upsampling to a shared maximal height.
- Hash rows with a configurable stateful hasher; internal nodes use a 2-to-1
  compression function.
- Support batch openings with canonical sibling ordering in proofs.

## Motivation

Standard STARK proofs involve multiple evaluation matrices with different heights,
particularly in multi-AIR designs where each AIR may have a different trace length.
This heterogeneity complicates
PCS integration: the verifier must track per-matrix heights, the DEEP quotient must
handle different lifting factors, and recursive verifiers need height-dependent code paths.

LMCS provides a **uniform-height view** by virtually upsampling shorter matrices to
the maximum height. This simplifies downstream logic:
- The PCS treats all matrices as having the same height.
- DEEP batching applies the same domain points to all matrices.
- Recursive verifiers avoid height-dependent branching.
- Switching to alternative polynomial commitment schemes (e.g., STIR or WHIR) becomes easier.

## Upsampling & Commitment Sketch

Let a polynomial f be evaluated over a coset gK with |K| = n, stored in
bit-reversed order. Let N = r · n be the max height over all matrices in the same commitment group. **Upsampling** repeats each
row r times in the bit-reversed layout, which corresponds to evaluating the
**lifted** polynomial f'(X) = f(X^r) over a larger coset of size N. The
commitment is the Merkle root of leaf hashes. Each leaf is computed by
sequentially absorbing the upsampled rows of all matrices into a
`StatefulHasher` instance, then squeezing:

```
leaf(i) = squeeze(absorb(row_0(i) || row_1(i) || ... || row_{t-1}(i) || salt?))
```

where row_j(i) is the upsampled row for matrix j at leaf index i.

**Absorption order**: Matrices are absorbed in their declared order (shortest to
tallest). Within each matrix, the row's field elements are absorbed left-to-right.
When alignment is enabled, each row is zero-padded to the aligned width before
absorbing the next matrix's row. An optional salt is absorbed after all rows.
This ordering is binding: reordering matrices or columns produces a different
commitment.

## Alignment

When using sponge-based hashers, leaf hashing absorbs field elements up to the
sponge rate before each permutation. **Alignment** ensures that each row
contributes a fixed number of permutations, with implicit zero-padding for any
remainder.

`build_aligned_tree` records the aligned widths so that verifiers know how many
elements to read per row. Benefits:
- Fixed permutation count per leaf simplifies parsing.
- No special end-of-row handling; padding is implicit in the sponge.
- Easier to reason about hashing costs.

For non-aligned trees (via `build_tree`), rows are absorbed directly without
padding; verifiers must use the original widths.

## Protocol Contract (High Level)

**What LMCS proves**:
- Opened rows are consistent with some committed leaf preimages under the
  configured hash and compression functions.
- Openings refer to the lifted, uniform-height view of the input matrices.

**What LMCS does not prove**:
- It does not prove a matrix "really had height n" beyond the supplied
  dimensions; lifting is indistinguishable from explicit repetition.
- It does not enforce periodicity or other structure; upstream protocols must.

**Trusted inputs / statement data**:
- `widths` (matrix widths, in commitment order).
- `log_max_height` (tree height / max domain size).
- Matrix ordering; permutation changes the commitment.

## Entry Points

| Item | Purpose |
|------|---------|
| `LmcsConfig` / `HidingLmcsConfig` | Build non-hiding or salted commitments |
| `Lmcs::build_tree` | Build a commitment tree with no transcript padding |
| `Lmcs::build_aligned_tree` | Build a tree using the hasher alignment for transcript padding |
| `LmcsTree::prove_batch` | Prove openings at multiple indices |
| `Lmcs::open_batch` | Verify batch openings |
| `BatchProof` / `Proof` | Parsing helpers for export (see below) |

### A Note on Proof Types

`BatchProof` and `Proof` are **parsing helpers** for exporting and debugging
proofs. They reconstruct the Merkle path structure from raw channel data but
do **not** validate proofs themselves.

Production verification uses `LmcsConfig::open_batch`, which parses and verifies
directly from the channel without constructing intermediate `BatchProof` objects.

## Assumptions & Invariants

- Matrices are sorted by height (shortest to tallest).
- `widths` and `log_max_height` are statement data chosen by the protocol.
- Evaluation rows are in bit-reversed order.
- Query indices are in range of the max height.
- Duplicate indices are coalesced in the transcript (sorted tree index order); verifiers
  return the same opening for each occurrence.
- If `build_aligned_tree` is used, alignment padding (as recorded on the tree) must be
  included in absorbed rows; LMCS does not enforce zero-padding.
- Hash and compression functions are collision-resistant.

## Code Map

| Path | Purpose |
|------|---------|
| `src/lmcs.rs` | Public API and verification |
| `src/lifted_tree.rs` | Tree construction and lifting |
| `src/utils.rs` | Leaf hashing helpers |

## Security

Audits should start with `SECURITY.md` at the workspace root for canonical proof
parsing, lifting correctness, and critical paths.

## License

Dual-licensed under MIT and Apache-2.0 at the workspace root.
