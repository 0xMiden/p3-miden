use core::marker::PhantomData;

use alloc::vec::Vec;
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_miden_air::{BusType, MidenAir, MidenAirBuilder};

use crate::{StarkGenericConfig, Val};

pub struct AirWithBoundaryConstraints<'a, SC, A>
where
    SC: StarkGenericConfig + core::marker::Sync,
    A: MidenAir<Val<SC>, SC::Challenge>,
    Val<SC>: TwoAdicField,
{
    pub inner: &'a A,
    pub phantom: PhantomData<SC>,
}

impl<'a, SC, A> MidenAir<Val<SC>, SC::Challenge> for AirWithBoundaryConstraints<'a, SC, A>
where
    SC: StarkGenericConfig + core::marker::Sync,
    A: MidenAir<Val<SC>, SC::Challenge>,
    Val<SC>: TwoAdicField,
{
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val<SC>>> {
        self.inner.preprocessed_trace()
    }

    fn num_public_values(&self) -> usize {
        self.inner.num_public_values()
    }

    fn periodic_table(&self) -> Vec<Vec<Val<SC>>> {
        self.inner.periodic_table()
    }

    fn num_randomness(&self) -> usize {
        self.inner.num_randomness()
    }

    fn aux_width(&self) -> usize {
        self.inner.aux_width()
    }

    /// Types of buses
    fn bus_types(&self) -> &[BusType] {
        self.inner.bus_types()
    }

    fn build_aux_trace(
        &self,
        _main: &RowMajorMatrix<Val<SC>>,
        _challenges: &[SC::Challenge],
    ) -> Option<RowMajorMatrix<Val<SC>>> {
        self.inner.build_aux_trace(_main, _challenges)
    }

    fn eval<AB: MidenAirBuilder<F = Val<SC>>>(&self, builder: &mut AB) {
        // First, apply the inner AIR's constraints
        self.inner.eval(builder);

        if self.inner.num_randomness() > 0 {
            // Then, apply any additional boundary constraints as needed
            let aux = builder.permutation();
            let aux_current = aux.row_slice(0).unwrap();
            let aux_width = aux_current.len();
            let aux_bus_boundary_values = builder.aux_bus_boundary_values().to_vec();
            let bus_types = self.inner.bus_types();

            assert_eq!(
                aux_width,
                self.inner.aux_width(),
                "aux trace width does not match MidenAir::aux_width()"
            );
            assert_eq!(
                aux_bus_boundary_values.len(),
                aux_width,
                "aux bus boundary values length does not match MidenAir::aux_width()"
            );
            assert!(
                bus_types.len() <= aux_width,
                "bus_types length exceeds aux width"
            );

            for (idx, bus_type) in bus_types.iter().enumerate() {
                match bus_type {
                    BusType::Multiset => {
                        builder
                            .when_first_row()
                            .assert_zero_ext(aux_current[idx].into() - AB::ExprEF::ONE);
                    }
                    BusType::Logup => {
                        builder
                            .when_first_row()
                            .assert_zero_ext(aux_current[idx].into());
                    }
                }
            }

            for idx in 0..aux_width {
                builder
                    .when_last_row()
                    .assert_zero_ext(aux_current[idx].into() - aux_bus_boundary_values[idx].into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;

    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
    use p3_merkle_tree::MerkleTreeMmcs;
    use p3_miden_air::{BusType, MidenAir};
    use p3_miden_fri::TwoAdicFriPcs;
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};

    use crate::{AirWithBoundaryConstraints, Entry, SymbolicExpression, get_symbolic_constraints};

    type Val = Goldilocks;
    type Perm = Poseidon2Goldilocks<16>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = MerkleTreeMmcs<
        <Val as p3_field::Field>::Packing,
        <Val as p3_field::Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    type Challenge = BinomialExtensionField<Val, 2>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Dft = Radix2DitParallel<Val>;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    type TestConfig = crate::StarkConfig<Pcs, Challenge, Challenger>;

    struct AuxBoundaryAir {
        aux_width: usize,
        bus_types: &'static [BusType],
    }

    impl MidenAir<Goldilocks, Challenge> for AuxBoundaryAir {
        fn width(&self) -> usize {
            1
        }

        fn num_randomness(&self) -> usize {
            1
        }

        fn aux_width(&self) -> usize {
            self.aux_width
        }

        fn bus_types(&self) -> &[BusType] {
            self.bus_types
        }

        fn eval<AB: p3_miden_air::MidenAirBuilder<F = Goldilocks>>(&self, _builder: &mut AB) {}
    }

    fn expr_contains_aux_boundary(expr: &SymbolicExpression<Goldilocks>, idx: usize) -> bool {
        use SymbolicExpression::*;

        match expr {
            Variable(v) => v.entry == Entry::AuxBusBoundary && v.index == idx,
            Add { x, y, .. } | Sub { x, y, .. } | Mul { x, y, .. } => {
                expr_contains_aux_boundary(x, idx) || expr_contains_aux_boundary(y, idx)
            }
            Neg { x, .. } => expr_contains_aux_boundary(x, idx),
            IsFirstRow | IsLastRow | IsTransition | Constant(_) => false,
        }
    }

    #[test]
    fn test_boundary_constraints_cover_all_aux_columns() {
        let inner = AuxBoundaryAir {
            aux_width: 3,
            bus_types: &[BusType::Logup],
        };
        let air = AirWithBoundaryConstraints::<TestConfig, _> {
            inner: &inner,
            phantom: PhantomData,
        };

        let constraints = get_symbolic_constraints::<Goldilocks, Challenge, _>(
            &air,
            0,
            0,
            inner.aux_width(),
            inner.num_randomness(),
        );

        for idx in 0..inner.aux_width() {
            assert!(
                constraints
                    .iter()
                    .any(|expr| expr_contains_aux_boundary(expr, idx)),
                "missing aux bus boundary constraint for column {idx}"
            );
        }
    }
}
