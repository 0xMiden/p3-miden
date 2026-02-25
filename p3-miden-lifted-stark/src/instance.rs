//! Re-export instance types from `p3-miden-lifted-air`.
//!
//! These types were moved to `p3-miden-lifted-air` to live closer to the AIR trait
//! definitions. Re-exported here for backward compatibility.

pub use p3_miden_lifted_air::{AirInstance, AirWitness, ValidationError, validate_instances};
