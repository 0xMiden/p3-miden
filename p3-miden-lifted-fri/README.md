# p3-miden-lifted-fri

Lifted Polynomial Commitment Scheme (PCS) built from the FRI LDT on the DEEP quotient polynomial,
using LMCS for commitments over evaluation matrices.

## Protocol Summary

This is a polynomial commitment scheme where the verifier requests evaluations
of all committed matrices at the same N points (derived via Fiat-Shamir).
The protocol and implementation are fully generic over N; the typical
STARK instantiation uses N = 2 (z and gz for next-row constraints).
All matrices appear at a uniform height via LMCS upsampling, which:
- Eliminates height-dependent code paths, enabling uniform recursive verifier implementations.
- Simplifies DEEP quotient construction (same domain points for all matrices).

**Protocol flow**:
1. **Commit**: LMCS commits to (possibly upsampled) evaluation matrices.
2. **Open**: DEEP batches evaluation claims into one low-degree polynomial.
3. **Verify**: FRI checks low degree; LMCS verifies consistency with commitments.

FRI folding is configurable (arity 2, 4, or 8).

## Math Sketch

Given evaluation claims f_i(z_j), DEEP forms a quotient polynomial

```
Q(X) = ∑_i ∑_j α^i · β^j · (f_i(X) - f_i(z_j)) / (X - z_j)
```

so Q is low-degree iff all claims are consistent. Evaluations are stored in
bit-reversed order. **Lifting** treats shorter matrices as evaluations of
f_i(X^r) on the LDE domain (r = lde_height / height_i), achieved via
**upsampling** (row repetition in bit-reversed order). FRI then repeatedly
folds evaluations by the chosen arity to test low degree.

## Optimizations

- **Aligned openings**: Rows are padded to a multiple of the sponge rate,
  ensuring a fixed number of permutations per leaf hash. Simplifies verifier
  parsing.
- **Uniform opening points**: All matrices (regardless of original height) are
  opened at the same domain points. DEEP batching uses a single set of
  z-values; no per-matrix domain reconstruction.
- **Recursive verifier benefits**: Uniform height eliminates branching on
  matrix dimensions. The verifier logic is a single code path.

## Entry Points

| Path | Purpose |
|------|---------|
| `src/prover.rs` | Prover flow and transcript emission |
| `src/verifier.rs` | Verifier flow and transcript consumption |
| `params.rs` | `PcsParams` configuration |
| `proof.rs` | `PcsTranscript` typed view of proof data |

## Modules

| Module | Purpose |
|--------|---------|
| `deep/` | DEEP quotient batching and interpolation |
| `fri/` | FRI folding and query logic |
| `utils.rs` | Domain mapping and bit-reversed helpers |

## Conventions & Assumptions

- Evaluation rows are in bit-reversed order.
- Domain sizes and folding factors come from `PcsParams`.
- The PCS APIs use `log_lde_height` to denote the LDE evaluation domain height (the
  height of committed LDE matrices). When a trace degree is known, the typical mapping is:
  `log_lde_height = log_trace_height + log_blowup` (plus any caller-chosen extension).
- **Upsampling** (row repetition in bit-reversed order) semantics follow `p3-miden-lmcs`.
- **Lifted-trace soundness**: This implementation assumes that the AIR's transition
  constraints do not depend on the wrap-around row (i.e., they are not periodic over the
  original small domain). The PCS does not enforce periodicity constraints on lifted
  columns; the caller/AIR must ensure that transition constraints are compatible with
  the lifted evaluation domain.
- Alignment padding is a transcript-formatting convenience for trace commitments built with
  `build_aligned_tree` (alignment is recorded on the tree). FRI round commitments are unaligned. Padding keeps verifier parsing
  uniform because the underlying sponge pads with zeros after absorbing a list of elements.
  The padded columns are treated as additional polynomials and are checked for low degree by
  the PCS, but they are not forced to be zero unless the caller/AIR enforces that.
- Transcript ordering (observe → grind → sample) is security-critical; see
  `SECURITY.md`.

## Transcript Model

Channels produce compact proof buffers suitable for transmission:
- `ProverTranscript` records field elements, commitments, and LMCS hints.
- `VerifierTranscript` consumes the buffer while verifying.

Structured types (`PcsTranscript`, `DeepTranscript`, `FriTranscript`) exist for
**export and debugging**. They reconstruct the full prover-verifier interaction
(including challenges and PoW witnesses) without re-running verification. These
types validate parsing shape only—they do **not** guarantee proof validity.

**Future**: A transcript fingerprint (hash of the fully consumed transcript) may
be added to check implementation equivalence across verifier implementations.

## Tests to Start With

- `src/fri/fold/mod.rs`
- `src/deep/interpolate.rs`
- `src/tests.rs`

## Security

Audits should start with `SECURITY.md` at the workspace root for transcript
ordering, DEEP/FRI invariants, and critical paths.

## License

Dual-licensed under MIT and Apache-2.0 at the workspace root.
