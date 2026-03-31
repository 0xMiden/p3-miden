//! Goldilocks + Keccak test configuration.

use alloc::vec;

use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_field::{Field, extension::BinomialExtensionField};
use p3_goldilocks::Goldilocks;
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_miden_stateful_hasher::{SerializingStatefulSponge, StatefulSponge};
use p3_symmetric::{CompressionFunctionFromHasher, Hash, PaddingFreeSponge};

/// Keccak permutation width (fixed).
pub const WIDTH: usize = 25;

/// Sponge rate (fixed for Keccak).
pub const RATE: usize = 17;

/// Digest size in u64 elements (fixed for Keccak).
pub const DIGEST: usize = 4;

/// Base field.
pub type F = Goldilocks;

/// Packed base field for SIMD operations.
pub type P = <F as Field>::Packing;

/// Extension field.
pub type EF = BinomialExtensionField<F, 2>;

/// Sponge for MMCS-style hashing (field-agnostic, operates on `u64` lanes).
pub type KeccakMmcsSponge = PaddingFreeSponge<KeccakF, WIDTH, RATE, DIGEST>;

/// 2-to-1 compression for Merkle trees.
pub type Compress = CompressionFunctionFromHasher<KeccakMmcsSponge, 2, DIGEST>;

/// Inner stateful sponge operating on `u64` lanes.
pub type InnerSponge = StatefulSponge<KeccakF, WIDTH, RATE, DIGEST>;

/// Serializing sponge that converts field elements to `u64` before absorption.
pub type Sponge = SerializingStatefulSponge<InnerSponge>;

/// Commitment type (hash of `u64` digest elements).
pub type Commitment = Hash<F, u64, DIGEST>;

/// Fiat-Shamir challenger (serializing, byte-based via Keccak-256).
pub type Challenger = SerializingChallenger64<F, HashChallenger<u8, Keccak256Hash, 32>>;

/// Create standard test components for Merkle tree construction.
pub fn test_components() -> (Sponge, Compress) {
    let sponge = Sponge::new(InnerSponge::new(KeccakF));
    let compress = Compress::new(KeccakMmcsSponge::new(KeccakF));
    (sponge, compress)
}

/// Create a standard challenger for Fiat-Shamir.
pub fn test_challenger() -> Challenger {
    Challenger::from_hasher(vec![], Keccak256Hash)
}

// =============================================================================
// LMCS layer
// =============================================================================

/// LMCS configured with Goldilocks + Keccak (SIMD-parallel).
pub type Lmcs = crate::lmcs::config::LmcsConfig<
    [F; p3_keccak::VECTOR_LEN],
    [u64; p3_keccak::VECTOR_LEN],
    Sponge,
    Compress,
    WIDTH,
    DIGEST,
>;

crate::testing::define_lmcs_test_helpers!();

/// Create a test LMCS instance.
pub fn test_lmcs() -> Lmcs {
    let (sponge, compress) = test_components();
    crate::lmcs::config::LmcsConfig::new(sponge, compress)
}
