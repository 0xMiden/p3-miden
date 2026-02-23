# Security Review Guide

This document provides a focused guide for security review of the Lifted FRI PCS implementation,
with an explicit path toward a composed STARK protocol.

## Protocol Hierarchy

LMCS -> {DEEP, FRI} -> PCS -> STARK (TBD)

- LMCS provides commitments and batch openings.
- DEEP + FRI implement the low-degree check.
- PCS orchestrates DEEP + FRI over LMCS commitments.
- STARK (planned) will bind statement data, enforce canonicality, and define AIR constraints.

## Quick Map

- Threat model: statement metadata is validated and Fiat-Shamir-bound by the outer STARK; attacker controls transcript bytes.
- APIs: `open_with_channel`, `verify_with_channel`, `LmcsConfig::open_batch`.
- Data ownership: params/widths/heights are trusted statement data; transcript streams are attacker-controlled.
- Composition boundary: this layer may leave unread transcript data; outer protocol enforces canonicality.

## What to Review First

1) `p3-miden-lifted-fri/src/verifier.rs::verify_with_channel`
2) `p3-miden-lmcs/src/lmcs.rs::LmcsConfig::open_batch` + `p3-miden-lmcs/src/proof.rs`
3) DEEP verifier reduction and domain reconstruction
4) FRI verifier round loop (grind/challenge, folding, index shifting)

## Scope Overview

| Area | Priority | Notes |
|------|----------|-------|
| `p3-miden-lmcs` (LMCS) | High | Commitment scheme used by the protocol. |
| `p3-miden-lmcs/src/mmcs/*` (MMCS wrapper) | Low | Compatibility layer for potential upstreaming; not used by our protocol today. |
| `p3-miden-lifted-fri` | High | DEEP quotient + FRI + PCS orchestration. |
| `p3-miden-transcript` | Medium | Transcript channels, parsing shape, PoW witnesses. |
| `p3-miden-stateful-hasher` | Medium | Stateful hashers and alignment semantics. |
| `p3-miden-stark` (TBD) | High | Will bind statement data and define protocol-level constraints. |

## Attack Surface and Data Ownership

### Externally used APIs (current trust boundary)
- `p3-miden-lifted-fri/src/prover.rs::open_with_channel`
- `p3-miden-lifted-fri/src/verifier.rs::verify_with_channel`
- `p3-miden-lmcs/src/lmcs.rs::LmcsConfig::open_batch`

These are the APIs expected to be called by the outer STARK.

### Key internal codepaths (not entrypoints)
- `p3-miden-lifted-fri/src/deep/verifier.rs::DeepOracle::{new,open_batch}`
- `p3-miden-lifted-fri/src/fri/verifier.rs::FriOracle::{new,test_low_degree}`
- Shape-only parsing helpers: `p3-miden-lifted-fri/src/proof.rs::PcsTranscript::from_verifier_channel`,
  `p3-miden-lmcs/src/proof.rs::BatchProof::read_from_channel`, `...::single_proofs`

### Trusted statement data (caller-supplied)
- Trace commitments (roots) and their matrix widths (including any alignment padding).
- `log_lde_height` / `log_max_height`.
- `PcsParams` (domain sizes, blowup, arities, query count, PoW bits).
- LMCS hash/compress configuration; for MMCS, `Dimensions` and matrix ordering.

Statement metadata (params/widths/heights/commitment roots) is not automatically
bound into this layer's transcript; the outer STARK must bind it to prevent statement rebinding.

### Attacker-controlled proof bytes
- Transcript streams (field elements + prover-serialized commitment objects) and LMCS hint data (rows, salts, siblings).
- PoW witnesses (consumed by `grind`) are part of the transcript field stream.
- Indices and challenges are derived by the verifier via Fiat-Shamir; they are not
  provided directly by the prover (but depend on prior transcript data).

## Transcript Model (p3-miden-transcript + typed views)

The transcript crate provides simple channels with explicit structure and parsing:

- `ProverTranscript` stores two streams (fields and commitments). `send_*` records and observes
  into the challenger; `hint_*` records only (used for LMCS proofs).
- `VerifierTranscript` consumes slices with explicit lengths and observes them into the challenger.
  `grind` reads a PoW witness from the field stream and checks it; `is_empty()` is available
  for canonical-consumption checks.

Because verification parses the transcript as it goes, we avoid redundant structural checks.
This keeps proofs compact but means malformed transcripts typically surface as parse failures
or evaluation mismatches rather than bespoke "shape" errors.

### Structured Types (Export Only)

We provide structured transcript views (`PcsTranscript`, `DeepTranscript`, `FriTranscript`,
`BatchProof`, etc.). These are **not used in production verification**; they exist for:
- Exporting proofs in a structured format for debugging or analysis.
- Reconstructing the full prover-verifier interaction (including challenges and PoW witnesses)
  without re-running verification.
- Enabling recursive verifiers that need structured proof data.

These types validate parsing shape only and do **not** guarantee proof validity.
Examples include `LmcsConfig::read_batch_proof_from_channel` and `BatchProof::single_proofs`.

### Future: Transcript Fingerprint

A transcript fingerprint (hash squeezed from the challenger after full consumption) may be
added to verify implementation equivalence: two verifiers processing the same valid proof
should produce identical fingerprints, providing a lightweight check that transcript
handling is consistent across implementations.

## Alignment and Padding Contract (LMCS + PCS)

- Hasher absorption padding: sponge-based hashers implicitly zero-pad to their alignment.
  Honest provers can rely on this instead of materializing zeros.
- Transcript alignment padding: rows may include extra elements for fixed-width parsing.
  LMCS does not constrain these values; a malicious prover can materialize non-zero padding.
- PCS/FRI: padded columns are treated as additional polynomials and are only constrained
  to be low-degree here. AIRs explicitly ignore them (unused variables); there is no need
  to check they are zero in this layer.
- Verifier behavior: alignment only affects the expected widths and how many elements are read
  from transcript hints. If prover and verifier disagree on alignment, transcript parsing fails.

## Protocol Contracts (High Level)

### LMCS (p3-miden-lmcs)

| What LMCS proves | What LMCS does not prove |
|---|---|
| Opened rows are consistent with a commitment under the configured hash/compress functions. | It does not prove the original matrix heights beyond the supplied dimensions. |
| Openings refer to the lifted, uniform-height view of the input matrices. | It does not enforce periodicity or semantics of columns. |
| Matrix order is binding in the commitment. | It does not assign meaning to matrix positions; the caller must bind ordering. |

**Trusted statement data**:
- `widths` (matrix widths, in commitment order).
- `log_max_height` (Merkle tree height / max domain size).
- The ordered list of matrices; permutation changes the commitment.
- For MMCS verification, `Dimensions` (widths already aligned, heights in commitment order) are
  trusted statement data; verification does not re-check ordering or power-of-two constraints.

**Commitment preimage (StatefulHasher model)**:
- LMCS leaf hashing is defined in terms of the `StatefulHasher` abstraction:
  initialize a hasher state, absorb lifted rows in matrix order, optionally absorb
  salt, then squeeze a digest.
- We support three hashing modes (see `p3-miden-stateful-hasher`):
  1) `StatefulSponge`: field-native sponge with proper sponge padding (alignment = rate).
  2) `SerializingStatefulSponge`: serializes field elements to u8/u32/u64 then sponges; alignment
     is derived from field size and inner rate.
  3) `ChainingHasher`: chaining mode `H(state || input)`; no padding (alignment = 1).

### Lifted FRI PCS (p3-miden-lifted-fri)

| What the PCS proves | What the PCS does not prove |
|---|---|
| Claimed evaluations are consistent with a low-degree polynomial matching commitments. | It does not validate parameter security levels or domain sizes. |
| DEEP + FRI enforce consistency of evaluation claims. | It does not reinterpret LMCS lifting; smaller matrices are treated as lifted. |

**Trusted statement data**:
- Domain sizes, folding arities, and other `PcsParams`.
- Transcript ordering and challenge sampling must follow the prescribed flow.

## Error Handling Policy

Many prover-side routines assume well-formed inputs (sorted heights, valid params, non-empty
matrices) and will `panic!` on violations. This is intentional for development and should be
interpreted as programmer error, not verifier-controlled input. Verification paths return
`Result`/`Option` on malformed proof bytes; panics should only be reachable via inconsistent
caller-supplied statement data. If a panic becomes reachable from attacker-controlled proof
bytes, treat it as a DoS bug.

## Verification Invariants

### 1. LMCS Merkle verification (`p3-miden-lmcs`)

**File**: `p3-miden-lmcs/src/lmcs.rs`

| Function | What to Check |
|----------|---------------|
| `LmcsConfig::open_batch` | Root recomputation, sibling consumption order, duplicate indices |
| `LmcsConfig::read_batch_proof_from_channel` | Transcript parsing shape |

**File**: `p3-miden-lmcs/src/lifted_tree.rs`

| Function | What to Check |
|----------|---------------|
| `LiftedMerkleTree::prove_batch` | Canonical sibling emission order |

**File**: `p3-miden-lmcs/src/proof.rs`

| Function | What to Check |
|----------|---------------|
| `BatchProof::read_from_channel` | Canonical sibling parsing; no out-of-range indices |
| `BatchProof::single_proofs` | Deterministic path reconstruction |

**Key invariants to verify**:
- [ ] Siblings are consumed in canonical order (left-to-right, bottom-to-top)
- [ ] Missing siblings return `InvalidProof`
- [ ] Duplicate indices may be coalesced as an optimization; verification should not rely on duplicates being preserved
- [ ] Out-of-range indices return `InvalidProof`
- [ ] Extra hints are ignored and left unread (callers can enforce transcript exhaustion)

### 2. LMCS leaf digest computation (`p3-miden-lmcs`)

**File**: `p3-miden-lmcs/src/lmcs.rs`

| Function | What to Check |
|----------|---------------|
| `Lmcs::hash` | Width validation (caller), salt absorption order |

**File**: `p3-miden-lmcs/src/lifted_tree.rs`

| Function | What to Check |
|----------|---------------|
| `build_leaf_states_upsampled` | Upsampling correctness, state maintenance |
| `absorb_matrix` | Correct row absorption into states |
| `validate_heights` | Height ordering enforcement |

**Key invariants to verify**:
- [ ] Matrices absorbed in height-sorted order (enforced by panic)
- [ ] Upsampling duplicates states correctly when height increases
- [ ] Salt absorbed after all matrix rows
- [ ] SIMD path produces identical results to scalar path

### 3. DEEP quotient invariants (`p3-miden-lifted-fri/deep`)

**Files**: `p3-miden-lifted-fri/src/deep/prover.rs`, `.../deep/verifier.rs`,
`.../deep/interpolate.rs`, `.../proof.rs`, `.../utils.rs`

The DEEP quotient is a protocol invariant shared by prover and verifier: the constructed
quotient must be low-degree iff all evaluation claims are correct. Review the symmetry of
the reduction and the mapping from evaluation points to domain points.

**Key invariants to verify**:
- [ ] Evaluations are observed into the transcript before grinding and challenge sampling.
- [ ] Challenge sampling order (grind -> sample) is consistent and uses the same transcript view.
- [ ] Coefficients `alpha` and `beta` are applied in the same Horner order on both sides.
- [ ] Domain points are reconstructed identically (bit-reversal + coset shift).
- [ ] Alignment/padding: evaluation vectors are serialized with aligned widths; padded columns are
      treated as extra polynomials. For sponge-based hashers, explicit zero padding is equivalent
      to implicit sponge padding; for chaining hashers, alignment is 1.
- [ ] Lifting semantics: smaller matrices are evaluated via `z^r` with the correct `r`, and the
      barycentric weight folding matches that lifted view.
- [ ] Evaluation points are outside the LDE domain (no division by zero).

### 4. FRI protocol invariants (`p3-miden-lifted-fri/fri`)

**Files**: `p3-miden-lifted-fri/src/fri/mod.rs`, `.../fri/prover.rs`,
`.../fri/verifier.rs`, `.../fri/fold/*`

Treat FRI as a single protocol: check that the prover/verification logic implements the same
sequence of folds, indices, and checks.

**Key invariants to verify**:
- [ ] Round count and domain sizes are derived correctly from params (loop termination is correct).
- [ ] Per-round commitment -> grind -> challenge order is preserved.
- [ ] Query indices are shifted correctly each round (`log_arity * (round + 1)`).
- [ ] `s_inv` values are computed identically (bit-reversed positions).
- [ ] Folding formulas for arity 2/4/8 match the intended interpolation.
- [ ] Final polynomial evaluation uses the correct point and degree bounds.

### 5. PCS orchestration invariants (`p3-miden-lifted-fri`)

- [ ] Transcript operations occur in the correct order.
- [ ] All proofs are verified (none skipped).
- [ ] Error types are propagated correctly.
- [ ] Statement data is bound into the transcript by the caller before sampling challenges.
- [ ] Canonicality is enforced at composition boundaries (e.g., `is_empty()` if required).

## STARK Protocol (TBD)

Assumptions for the composed STARK layer:
- Statement metadata is validated and Fiat-Shamir-bound to prevent statement rebinding.
- Proof bytes are serialized/deserialized with size caps.
- Canonicality is enforced at protocol boundaries (transcript exhaustion).
- AIR constraints define whether padded columns are ignored or must be zero.

Placeholders for STARK responsibilities:
- Bind statement data (commitment roots, widths, heights, params) into Fiat-Shamir.
- Define serialization/deserialization of proof bytes and enforce size caps.
- Enforce transcript exhaustion and proof canonicality at protocol boundaries.
- Specify AIR constraints, including whether padded columns must be zero.
- Define the full proof layout and interaction flow around `open_with_channel` and
  `verify_with_channel`.

## Dependencies

We rely on upstream Plonky3 primitives (`p3-field`, `p3-symmetric`, `p3-challenger`, `p3-dft`).
Assume these are correct for the purposes of this review.

## Appendix

### DoS / size-bound considerations

Proof parsing allocates based on statement data (widths, number of matrices, `num_queries`)
and transcript data (LMCS hints, FRI commitments). This layer does not deserialize raw bytes;
size caps and transcript length limits belong to the outer protocol. Verification cost and
allocations scale with transcript lengths provided via the channel; upstream must cap them.

### Tests and reference points

- LMCS lifting and equivalence: `p3-miden-lmcs/src/lifted_tree.rs` (`upsampled_equivalence`).
- LMCS batch openings / duplicate indices: `p3-miden-lmcs/src/tests.rs`.
- PCS end-to-end and FRI arities: `p3-miden-lifted-fri/src/tests.rs`, `.../fri/tests.rs`.
- DEEP tests: `p3-miden-lifted-fri/src/deep/tests.rs`.
- Sponge alignment semantics: `p3-miden-stateful-hasher/src/field_sponge.rs`,
  `p3-miden-stateful-hasher/src/serializing_sponge.rs`.

## Soundness Analysis

### DEEP Quotient Soundness

TODO

### FRI Soundness

TODO

### Upsampling Soundness

TODO

Key argument: the verifier sees only equal-height matrices (via upsampling). A cheating
prover cannot exploit height differences because all openings occur at the uniform height.
The upsampling operation preserves polynomial evaluations under the lifting map f(X) → f(X^r).

### Overall PCS Soundness

TODO

## Security Parameters

TODO: To be filled with the composed STARK protocol.

Placeholder for guidance on:
- Query count selection
- `log_blowup` recommendations
- `log_final_degree` bounds
- PoW bits for grinding
