use crate::{MidenAirBuilder, RowMajorMatrix};
use p3_air::BaseAir;
use p3_field::Field;

pub enum BusType {
    /// A multiset bus
    Multiset,
    /// A logup bus
    Logup,
}

/// Variable-length public inputs for a single bus.
pub type BusPublicInputs<'a, F> = &'a [&'a [F]];

/// Variable-length public inputs grouped per bus.
///
/// Structure: `&[bus0_rows, bus1_rows, ...]` where each `bus*_rows` is a slice
/// of rows, and each row is a slice of field elements.
pub type VarLenPublicInputs<'a, F> = &'a [BusPublicInputs<'a, F>];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuxFinalsError {
    InvalidAuxFinalsLength { expected: usize, got: usize },
    MissingBusPublicInputs { bus_index: usize },
    InvalidBoundary { bus_index: usize },
    Custom(&'static str),
}

/// Contribution to the global bus boundary check.
///
/// The verifier combines these by multiplying `multiset` and summing `logup`,
/// and expects the folded result to be `(1, 0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuxFinalsContribution<EF> {
    pub multiset: EF,
    pub logup: EF,
}

impl<EF: Field> AuxFinalsContribution<EF> {
    pub fn identity() -> Self {
        Self {
            multiset: EF::ONE,
            logup: EF::ZERO,
        }
    }

    pub fn combine(self, other: Self) -> Self {
        Self {
            multiset: self.multiset * other.multiset,
            logup: self.logup + other.logup,
        }
    }

    pub fn combine_in_place(&mut self, other: Self) {
        self.multiset *= other.multiset;
        self.logup += other.logup;
    }

    pub fn combine_from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self>,
    {
        iter.into_iter().reduce(Self::combine).unwrap_or_default()
    }

    pub fn is_identity(&self) -> bool {
        self.multiset == EF::ONE && self.logup == EF::ZERO
    }
}

impl<EF: Field> Default for AuxFinalsContribution<EF> {
    fn default() -> Self {
        Self::identity()
    }
}

/// An extension of `BaseAir` that includes support for auxiliary traces.
///
/// This trait is defined in p3-miden-air (not upstream p3-air) because auxiliary
/// trace support is Miden-specific functionality.
pub trait BaseAirWithAuxTrace<F, EF>: BaseAir<F> {
    /// Number of challenges (extension fields) that is required to compute the aux trace
    fn num_randomness(&self) -> usize {
        0
    }

    /// Number of columns (in extension field) that is required for aux trace
    fn aux_width(&self) -> usize {
        0
    }

    /// Types of buses
    fn bus_types(&self) -> &[BusType] {
        &[]
    }

    /// Build an aux trace (EF-based) given the main trace and EF challenges.
    /// Return None to indicate no aux or to fall back to legacy behavior.
    fn build_aux_trace(
        &self,
        _main: &RowMajorMatrix<F>,
        _challenges: &[EF],
    ) -> Option<RowMajorMatrix<F>> {
        // default: do nothing
        None
    }

    /// Load an aux builder.
    ///
    /// An aux builder takes in a main matrix and a randomness, and generate a aux matrix.
    fn with_aux_builder<Builder>(&mut self, _builder: Builder)
    where
        Builder: Fn(&RowMajorMatrix<F>, &[EF]) -> RowMajorMatrix<F> + Send + Sync + 'static,
    {
        // default: do nothing
    }
}

/// Super trait for all AIR definitions in the Miden VM ecosystem.
///
/// This trait contains all methods from `BaseAir`, `BaseAirWithPublicValues`,
/// `BaseAirWithAuxTrace`, and `Air`. Implementers only need to implement this
/// single trait.
///
/// To use your AIR with the STARK prover/verifier, you'll need to also implement
/// the p3-air traits using the `impl_p3_air_traits!` macro.
///
/// # Type Parameters
///
/// - `F`: The base field type
/// - `EF`: The extension field type (used for auxiliary traces like LogUp)
///
/// # Required Methods
///
/// - [`width`](MidenAir::width) - Number of columns in the main trace
/// - [`eval`](MidenAir::eval) - Constraint evaluation logic
///
/// # Optional Methods (with default implementations)
///
/// All other methods have default implementations that can be overridden as needed.
pub trait MidenAir<F, EF>: Sync {
    // ==================== BaseAir Methods ====================

    /// The number of columns (a.k.a. registers) in this AIR.
    fn width(&self) -> usize;

    /// Return an optional preprocessed trace matrix to be included in the prover's trace.
    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        None
    }

    // ==================== BaseAirWithPublicValues Methods ====================

    /// Return the number of expected public values.
    fn num_public_values(&self) -> usize {
        0
    }

    // ==================== BaseAirWithPeriodicTables Methods ====================

    /// Return the periodic table data.
    fn periodic_table(&self) -> Vec<Vec<F>> {
        vec![]
    }

    // ==================== BaseAirWithAuxTrace Methods ====================

    /// Number of challenges (extension fields) that is required to compute the aux trace
    fn num_randomness(&self) -> usize {
        0
    }

    /// Number of columns (in based field) that is required for aux trace
    fn aux_width(&self) -> usize {
        0
    }

    /// Types of buses
    fn bus_types(&self) -> &[BusType] {
        &[]
    }

    /// Build an aux trace (EF-based) given the main trace and EF challenges.
    /// Return None to indicate no aux or to fall back to legacy behavior.
    /// The output is a matrix of EF elements, flattened to a matrix of F elements.
    fn build_aux_trace(
        &self,
        _main: &RowMajorMatrix<F>,
        _challenges: &[EF],
    ) -> Option<RowMajorMatrix<F>> {
        None
    }

    /// Computes this AIR's contribution to the global bus boundary check.
    ///
    /// This hook should combine aux-final values with public inputs (including
    /// variable-length public inputs) to return a contribution for multiset and
    /// logup buses. The verifier will fold these contributions by multiplying
    /// `multiset` and summing `logup`, and expects the global result to be `(1, 0)`.
    ///
    /// For standard multiset/logup buses, consider using
    /// `crate::reduce_multiset_bus_boundary_varlen` /
    /// `crate::reduce_logup_bus_boundary_varlen` to compute expected boundary values.
    ///
    /// `var_len_public_inputs` is grouped per bus and per row. Example:
    ///
    /// ```rust,ignore
    /// // Two buses: the first has two rows of length 2, the second has one row of length 1.
    /// let bus0_rows: Vec<&[F]> = vec![&[a0, a1], &[b0, b1]];
    /// let bus1_rows: Vec<&[F]> = vec![&[c0]];
    /// let var_len_public_inputs = vec![bus0_rows.as_slice(), bus1_rows.as_slice()];
    /// ```
    fn verify_aux_finals(
        &self,
        randomness: &[EF],
        aux_finals: &[EF],
        public_values: &[F],
        var_len_public_inputs: VarLenPublicInputs<'_, F>,
    ) -> Result<AuxFinalsContribution<EF>, AuxFinalsError>;

    /// Load an aux builder.
    ///
    /// An aux builder takes in a main matrix and a randomness, and generate a aux matrix.
    fn with_aux_builder<Builder>(&mut self, _builder: Builder)
    where
        Builder: Fn(&RowMajorMatrix<F>, &[EF]) -> RowMajorMatrix<F> + Send + Sync + 'static,
    {
        // default: do nothing
    }

    // ==================== Air Methods ====================

    /// Evaluate all AIR constraints using the provided builder.
    ///
    /// The builder provides both the trace on which the constraints
    /// are evaluated on as well as the method of accumulating the
    /// constraint evaluations.
    ///
    /// # Arguments
    /// - `builder`: Mutable reference to a `MidenAirBuilder` for defining constraints.
    fn eval<AB: MidenAirBuilder<F = F>>(&self, builder: &mut AB);
}
