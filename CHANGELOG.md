## 0.4.3 (TBD)

- Fix periodic column evaluation on LDE/quotient domains.
- [BREAKING] Remove forced conversion of periodic values from F to EF.

## 0.4.2 (2025-01-14)

- [BREAKING] Removed `p3-miden-goldilocks` crate, now uses upstream `p3-goldilocks` (#3).
- Updated `Pcs` trait implementation for Plonky3 v0.4.2 compatibility (#3).
- Updated Plonky3 dependencies to v0.4.2 (#3).
- Handle aux boundary values constraints in prover and verifier (#7).
- Fix panics in verifier (#19).

## 0.4.0 (2025-12-23)

- Initial release on crates.io containing Miden-specific Plonky3 crates.
- [BREAKING] Consolidated crates and removed duplicate symbolic modules to use base Plonky3 (#1).
- Added workspace release automation with dry-run and publish workflows.
- Migrated Plonky3 dependencies from git to crates.io v0.4.1 (#1).
- Added README documenting the five Miden-specific Plonky3 crates.
- Added dual MIT/Apache-2.0 license.
- Added CI workflows and Makefile for build automation.
- Fixed debug constraint checking to be gated behind `cfg(debug_assertions)`.

### Crates included

- `p3-miden-air`: Miden-specific AIR abstractions.
- `p3-miden-fri`: Miden FRI implementation with hiding commitments.
- `p3-miden-prover`: Miden prover with constraint checking.
- `p3-miden-uni-stark`: Miden uni-STARK implementation.
