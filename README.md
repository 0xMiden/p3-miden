# Miden Plonky3

Miden-specific [Plonky3](https://github.com/Plonky3/Plonky3) crates for the Miden VM STARK prover.

## Crates

| Crate | Based On | Purpose |
|-------|----------|---------|
| `p3-miden-air` | `p3-air` | AIR traits supporting auxiliary and periodic columns |
| `p3-miden-lmcs` | - | Lifted Matrix Commitment Scheme for uniform-height commitments |
| `p3-miden-lifted-fri` | - | Lifted FRI PCS (DEEP quotient + FRI over LMCS) |
| `p3-miden-transcript` | - | Transcript channels for Fiat-Shamir protocols |
| `p3-miden-stateful-hasher` | `p3-symmetric` | Stateful sponge-like hashers for incremental hashing |
| `p3-miden-uni-stark` | `p3-uni-stark` | Extended `Entry` enum with `Aux` and `Periodic` variants |
| `p3-miden-fri` | `p3-fri` | Miden FRI implementation with configurable folding factors |
| `p3-miden-prover` | - | Miden STARK prover combining the above crates |

## Modifications

### p3-miden-air & p3-miden-uni-stark
- Extends `Entry` enum with `Aux` (auxiliary trace columns) and `Periodic` (periodic columns) variants
- Required for Miden's permutation arguments and periodic column constraints

### p3-miden-fri
- Supports higher folding factors for Miden's FRI implementation
- Configurable folding strategy for future transition to lifted FRI

### p3-miden-prover
- Orchestrates proof generation with auxiliary trace support
- Includes LogUp argument implementation for permutation checks
- Constraint folding for auxiliary constraints

### p3-miden-stateful-hasher
- `StatefulHasher` trait for incremental hashing with serializable state
- Field-native sponge and serializing sponge implementations
- Re-exports upstream `p3-symmetric` types for convenience

### p3-miden-lmcs
- Lifted Matrix Commitment Scheme for matrices of varying heights
- Virtual upsampling to uniform height via row repetition in bit-reversed order
- Batch openings with canonical sibling emission

### p3-miden-transcript
- `ProverTranscript` and `VerifierTranscript` for Fiat-Shamir channels
- Separate streams for field elements, commitments, and hints
- Proof-of-work witness handling

### p3-miden-lifted-fri
- Lifted Merkle tree commitments for matrices of varying heights via upsampling
- DEEP quotient construction for batching polynomial evaluation claims
- FRI protocol with configurable arity-2, arity-4, and arity-8 folding
- Complete PCS (Polynomial Commitment Scheme) combining DEEP and FRI

## Upstream Compatibility

Core Plonky3 crates remain unchanged from upstream:
`p3-field`, `p3-matrix`, `p3-commit`, `p3-challenger`, `p3-symmetric`, `p3-merkle-tree`, `p3-dft`, `p3-interpolation`, `p3-util`

## License

This project is dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE).
