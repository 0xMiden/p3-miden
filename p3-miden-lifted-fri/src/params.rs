//! PCS parameters.

use crate::deep::DeepParams;
use crate::fri::FriParams;

/// Complete PCS parameters combining DEEP and FRI parameters.
///
/// Groups all parameters needed for `open` and `verify` into a single struct,
/// reducing the number of function arguments and ensuring consistent configuration.
#[derive(Clone, Copy, Debug)]
pub struct PcsParams {
    /// DEEP quotient parameters.
    pub deep: DeepParams,

    /// FRI protocol parameters.
    pub fri: FriParams,

    /// Number of query repetitions.
    pub num_queries: usize,

    /// Grinding bits before query index sampling.
    pub query_pow_bits: usize,
}
