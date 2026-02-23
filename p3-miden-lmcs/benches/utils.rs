//! Shared utilities for LMCS benchmarks.

use p3_keccak::{KeccakF, VECTOR_LEN};
use p3_miden_dev_utils::configs::baby_bear_keccak as bb_keccak;
use p3_miden_dev_utils::configs::baby_bear_poseidon2 as bb;
use p3_miden_dev_utils::configs::goldilocks_keccak as gl_keccak;
use p3_miden_dev_utils::configs::goldilocks_poseidon2 as gl;
use p3_miden_dev_utils::{
    BabyBearKeccak, BabyBearPoseidon2, BenchScenario, GoldilocksKeccak, GoldilocksPoseidon2,
};
use p3_miden_lmcs::{Lmcs, LmcsConfig};
use p3_miden_stateful_hasher::{SerializingStatefulSponge, StatefulSponge};

// =============================================================================
// Poseidon2 LMCS types
// =============================================================================

pub type BabyBearLmcs =
    LmcsConfig<bb::P, bb::P, bb::Sponge, bb::Compress, { bb::WIDTH }, { bb::DIGEST }>;
pub type GoldilocksLmcs =
    LmcsConfig<gl::P, gl::P, gl::Sponge, gl::Compress, { gl::WIDTH }, { gl::DIGEST }>;

// =============================================================================
// Keccak LMCS types
// =============================================================================

type KeccakStatefulSponge = StatefulSponge<KeccakF, 25, 17, 4>;

/// Keccak LMCS for BabyBear field.
///
/// Uses `[F; VECTOR_LEN]` for field elements and `[u64; VECTOR_LEN]` for digest elements,
/// enabling SIMD parallelization where `VECTOR_LEN` is platform-specific (1, 2, 4, or 8).
pub type BabyBearKeccakLmcs = LmcsConfig<
    [bb_keccak::F; VECTOR_LEN],
    [u64; VECTOR_LEN],
    SerializingStatefulSponge<KeccakStatefulSponge>,
    bb_keccak::KeccakCompress,
    25,
    4,
>;

/// Keccak LMCS for Goldilocks field.
///
/// Uses `[F; VECTOR_LEN]` for field elements and `[u64; VECTOR_LEN]` for digest elements,
/// enabling SIMD parallelization where `VECTOR_LEN` is platform-specific (1, 2, 4, or 8).
pub type GoldilocksKeccakLmcs = LmcsConfig<
    [gl_keccak::F; VECTOR_LEN],
    [u64; VECTOR_LEN],
    SerializingStatefulSponge<KeccakStatefulSponge>,
    gl_keccak::KeccakCompress,
    25,
    4,
>;

// =============================================================================
// LmcsScenario trait
// =============================================================================

/// Trait for creating LMCS instances from BenchScenario.
pub trait LmcsScenario: BenchScenario {
    type Lmcs: Lmcs<F = Self::F>;

    fn lmcs() -> Self::Lmcs;
}

// =============================================================================
// Poseidon2 implementations
// =============================================================================

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

// =============================================================================
// Keccak implementations
// =============================================================================

fn keccak_sponge() -> SerializingStatefulSponge<KeccakStatefulSponge> {
    SerializingStatefulSponge::new(StatefulSponge::new(KeccakF {}))
}

impl LmcsScenario for BabyBearKeccak {
    type Lmcs = BabyBearKeccakLmcs;

    fn lmcs() -> Self::Lmcs {
        let inner = bb_keccak::KeccakMmcsSponge::new(KeccakF {});
        let compress = bb_keccak::KeccakCompress::new(inner);
        LmcsConfig::new(keccak_sponge(), compress)
    }
}

impl LmcsScenario for GoldilocksKeccak {
    type Lmcs = GoldilocksKeccakLmcs;

    fn lmcs() -> Self::Lmcs {
        let inner = gl_keccak::KeccakMmcsSponge::new(KeccakF {});
        let compress = gl_keccak::KeccakCompress::new(inner);
        LmcsConfig::new(keccak_sponge(), compress)
    }
}
