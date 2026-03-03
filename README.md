# Miden Plonky3

Miden-specific [Plonky3](https://github.com/Plonky3/Plonky3) crates.

The current focus of this workspace is a *lifted STARK* prover/verifier stack:
multi-trace proofs where traces of different heights are presented to the PCS
and verifier as a single uniform-height object via virtual lifting.

## Lifted STARK Stack

```text
p3-miden-lifted-{prover,verifier}
└── p3-miden-lifted-stark
    ├── p3-miden-lifted-fri   (PCS: DEEP + FRI)
    │   └── p3-miden-lmcs     (Merkle commitments with lifting)
    ├── p3-miden-lifted-air   (AIR traits + symbolic analysis)
    └── p3-miden-transcript   (Fiat-Shamir channels)
```

## Workspace Crates

| Crate | Purpose |
|------|---------|
| `p3-miden-lifted-prover` | Lifted STARK prover (`prove_single`, `prove_multi`) |
| `p3-miden-lifted-verifier` | Lifted STARK verifier (`verify_single`, `verify_multi`) |
| `p3-miden-lifted-stark` | Shared types (`StarkConfig`, `LiftedCoset`, `Selectors`, instances/witnesses) |
| `p3-miden-lifted-air` | Lifted AIR traits and symbolic constraint analysis |
| `p3-miden-lifted-fri` | PCS: DEEP quotient + FRI over LMCS commitments |
| `p3-miden-lmcs` | Lifted Matrix Commitment Scheme (uniform-height view) |
| `p3-miden-transcript` | Transcript channels (`ProverTranscript`, `VerifierTranscript`) |
| `p3-miden-stateful-hasher` | Stateful hashers used by LMCS |
| `p3-miden-lifted-examples` | Example AIRs + benchmark binaries |
| `p3-miden-dev-utils` | Test harnesses and config builders |

## Docs

- `docs/faq.md` (architecture Q&A)
- `docs/lifting.md` (math background for lifting)
- `SECURITY.md` (audit/review guide; transcript and composition notes)

## Where To Start (Code)

- Protocol flow: `p3-miden-lifted-prover/src/prover.rs` and `p3-miden-lifted-verifier/src/verifier.rs`
- PCS layer: `p3-miden-lifted-fri/src/prover.rs` and `p3-miden-lifted-fri/src/verifier.rs`
- Commitment layer: `p3-miden-lmcs/src/lmcs.rs` and `p3-miden-lmcs/src/lifted_tree.rs`
- Math background: `docs/lifting.md`

## Build / Test

```bash
make check
make test
make test-parallel
make lint
make doc
```

## Run An Example

```bash
cargo run -p p3-miden-lifted-examples --release --bin lifted_keccak
```

## Security Disclaimer

This code is research/prototype quality and has not been independently audited.
Do not treat any default parameters as production-ready.

## License

Dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE).
