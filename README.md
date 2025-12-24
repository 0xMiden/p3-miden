# Miden Plonky3

Miden-specific [Plonky3](https://github.com/Plonky3/Plonky3) crates for the Miden VM STARK prover.

## Crates

| Crate | Based On | Purpose |
|-------|----------|---------|
| `p3-miden-air` | `p3-air` | AIR traits supporting auxiliary and periodic columns |
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

## Upstream Compatibility

Core Plonky3 crates remain unchanged from upstream:
`p3-field`, `p3-matrix`, `p3-commit`, `p3-challenger`, `p3-symmetric`, `p3-merkle-tree`, `p3-dft`, `p3-interpolation`, `p3-util`

## License

This project is dual-licensed under [MIT](LICENSE-MIT) and [Apache-2.0](LICENSE-APACHE).
