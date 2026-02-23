//! Shared utilities for lifted benchmarks.

use p3_commit::Mmcs;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_dev_utils::configs::goldilocks_poseidon2 as gl;
use p3_miden_dev_utils::{BabyBearPoseidon2, GoldilocksPoseidon2, PcsScenario};
use p3_miden_lmcs::LmcsConfig;

// =============================================================================
// LMCS types (re-exported from dev-utils)
// =============================================================================

pub type BabyBearLmcs =
    LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;
pub type GoldilocksLmcs =
    LmcsConfig<gl::P, gl::P, gl::Sponge, gl::Compress, { gl::WIDTH }, { gl::DIGEST }>;

// =============================================================================
// LmcsScenario trait
// =============================================================================

/// Trait for creating LMCS instances from PcsScenario.
///
/// Extends `PcsScenario` to provide access to `RATE` and `challenger()` needed
/// for PCS benchmarks, plus the LMCS-specific type and constructor.
///
/// The LMCS commitment type must match the MMCS commitment type so that the
/// challenger (which is bound to observe MMCS commitments) can also observe
/// LMCS commitments.
pub trait LmcsScenario: PcsScenario {
    /// LMCS type with commitment compatible with the MMCS commitment type.
    type Lmcs: Mmcs<Self::F, Commitment = <Self::Mmcs as Mmcs<Self::F>>::Commitment>;

    /// Create a new LMCS instance.
    fn lmcs() -> Self::Lmcs;
}

impl LmcsScenario for BabyBearPoseidon2 {
    type Lmcs = BabyBearLmcs;

    fn lmcs() -> Self::Lmcs {
        let (_, sponge, compress) = bb::test_components();
        LmcsConfig::new(sponge, compress)
    }
}

impl LmcsScenario for GoldilocksPoseidon2 {
    type Lmcs = GoldilocksLmcs;

    fn lmcs() -> Self::Lmcs {
        let (_, sponge, compress) = gl::test_components();
        LmcsConfig::new(sponge, compress)
    }
}
