//! Goldilocks + BLAKE3-192 (24-byte digest) test configuration.

use alloc::vec;

use p3_blake3::Blake3;
use p3_challenger::{HashChallenger, SerializingChallenger64};
use p3_field::{Field, extension::BinomialExtensionField};
use p3_goldilocks::Goldilocks;
use p3_miden_stateful_hasher::{ChainingHasher, TruncatingHasher};
use p3_symmetric::CompressionFunctionFromHasher;

pub type Blake3_192 = TruncatingHasher<Blake3, 32, 24>;

/// Chaining state / digest width in bytes (matches digest size).
pub const WIDTH: usize = 24;

/// Digest size in bytes.
pub const DIGEST: usize = 24;

/// Base field.
pub type F = Goldilocks;

/// Packed base field for SIMD (not used by byte LMCS, but kept for parity with Poseidon2 configs).
pub type P = <F as Field>::Packing;

/// Extension field.
pub type EF = BinomialExtensionField<F, 2>;

/// Chaining sponge over BLAKE3-192 on serialized field elements (LMCS `StatefulHasher`).
pub type Sponge = ChainingHasher<Blake3_192>;

/// 2-to-1 compression via BLAKE3-192.
pub type Compress = CompressionFunctionFromHasher<Blake3_192, 2, DIGEST>;

/// Fiat-Shamir challenger over serialized field elements.
pub type Challenger = SerializingChallenger64<F, HashChallenger<u8, Blake3_192, DIGEST>>;

/// Sponge + compressor for Merkle construction.
pub fn test_components() -> (Sponge, Compress) {
    let h = Blake3_192::new(Blake3);
    (
        ChainingHasher::new(h),
        CompressionFunctionFromHasher::new(h),
    )
}

/// Fresh hash challenger (empty initial state).
pub fn test_challenger() -> Challenger {
    SerializingChallenger64::new(HashChallenger::<u8, Blake3_192, DIGEST>::new(
        vec![],
        Blake3_192::new(Blake3),
    ))
}

// =============================================================================
// LMCS layer
// =============================================================================

/// LMCS configured with Goldilocks + Blake3-192.
pub type Lmcs = crate::lmcs::config::LmcsConfig<F, u8, Sponge, Compress, WIDTH, DIGEST>;

crate::testing::define_lmcs_test_helpers!();

/// Create a test LMCS instance.
pub fn test_lmcs() -> Lmcs {
    let (sponge, compress) = test_components();
    crate::lmcs::config::LmcsConfig::new(sponge, compress)
}
