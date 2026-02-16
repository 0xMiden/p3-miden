use p3_field::{ExtensionField, Field};

/// Computes the final value for a multiset bus given variable-length public inputs.
///
/// Each row is encoded as a linear combination `r0 + sum(p_i * r_{i+1})` and then
/// multiplied across rows.
///
/// # Panics
/// Panics if `randomness` is empty or any row has length greater than `randomness.len() - 1`.
pub fn reduce_multiset_bus_boundary_varlen<'a, F, EF, I>(randomness: &[EF], public_inputs: I) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
    I: IntoIterator<Item = &'a [F]>,
{
    let (r0, r_tail) = randomness
        .split_first()
        .expect("randomness must have at least one element");
    let mut bus_p_last = EF::ONE;
    for row in public_inputs {
        bus_p_last *= row_linear_combination(*r0, r_tail, row);
    }
    bus_p_last
}

/// Computes the final value for a logup bus boundary constraint given variable-length public inputs.
///
/// Each row is encoded as a linear combination `r0 + sum(p_i * r_{i+1})` and then the
/// sum of inverses across rows is returned.
///
/// # Panics
/// Panics if `randomness` is empty or any row has length greater than `randomness.len() - 1`.
pub fn reduce_logup_bus_boundary_varlen<'a, F, EF, I>(randomness: &[EF], public_inputs: I) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
    I: IntoIterator<Item = &'a [F]>,
{
    let (r0, r_tail) = randomness
        .split_first()
        .expect("randomness must have at least one element");
    let mut bus_q_last = EF::ZERO;
    for row in public_inputs {
        let q_last = row_linear_combination(*r0, r_tail, row);
        bus_q_last += q_last.inverse();
    }
    bus_q_last
}

#[inline]
fn row_linear_combination<F, EF>(r0: EF, r_tail: &[EF], row: &[F]) -> EF
where
    F: Field,
    EF: ExtensionField<F>,
{
    assert!(
        row.len() <= r_tail.len(),
        "randomness must have at least row_len + 1 elements"
    );
    row.iter()
        .zip(r_tail.iter())
        .fold(r0, |acc, (p_i, r_i)| acc + *r_i * *p_i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 2>;

    #[test]
    fn test_multiset_boundary_varlen() {
        let randomness = vec![EF::from_u64(3), EF::from_u64(5), EF::from_u64(7)];
        let rows = [
            vec![F::from_u64(2), F::from_u64(4)],
            vec![F::from_u64(1), F::from_u64(6)],
        ];

        let r0 = randomness[0];
        let r1 = randomness[1];
        let r2 = randomness[2];
        let expected =
            (r0 + r1 * rows[0][0] + r2 * rows[0][1]) * (r0 + r1 * rows[1][0] + r2 * rows[1][1]);

        let got = reduce_multiset_bus_boundary_varlen::<F, EF, _>(
            &randomness,
            rows.iter().map(|r| r.as_slice()),
        );

        assert_eq!(got, expected);
    }

    #[test]
    fn test_logup_boundary_varlen() {
        let randomness = vec![EF::from_u64(7), EF::from_u64(11), EF::from_u64(13)];
        let rows = [
            vec![F::from_u64(3), F::from_u64(5)],
            vec![F::from_u64(2), F::from_u64(9)],
        ];

        let r0 = randomness[0];
        let r1 = randomness[1];
        let r2 = randomness[2];
        let q0 = r0 + r1 * rows[0][0] + r2 * rows[0][1];
        let q1 = r0 + r1 * rows[1][0] + r2 * rows[1][1];
        assert!(!q0.is_zero());
        assert!(!q1.is_zero());
        let expected = q0.inverse() + q1.inverse();

        let got = reduce_logup_bus_boundary_varlen::<F, EF, _>(
            &randomness,
            rows.iter().map(|r| r.as_slice()),
        );

        assert_eq!(got, expected);
    }
}
