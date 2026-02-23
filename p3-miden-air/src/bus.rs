use crate::AuxFinalsError;
use p3_field::{ExtensionField, Field};

/// Computes the final value for a multiset bus given variable-length public inputs.
///
/// Each row is encoded as a linear combination `r0 + sum(p_i * r_{i+1})` and then
/// multiplied across rows.
///
/// # Errors
/// Returns an error if `randomness` is empty or any row has length greater than
/// `randomness.len() - 1`.
pub fn reduce_multiset_bus_boundary_varlen<'a, F, EF, I>(
    randomness: &[EF],
    public_inputs: I,
) -> Result<EF, AuxFinalsError>
where
    F: Field,
    EF: ExtensionField<F>,
    I: IntoIterator<Item = &'a [F]>,
{
    if randomness.is_empty() {
        return Err(AuxFinalsError::Custom(
            "randomness must have at least one element",
        ));
    }
    let mut bus_p_last = EF::ONE;
    for row in public_inputs {
        bus_p_last *= row_linear_combination(randomness, row)?;
    }
    Ok(bus_p_last)
}

/// Computes the final value for a logup bus boundary constraint given variable-length public inputs.
///
/// Each row is encoded as a linear combination `r0 + sum(p_i * r_{i+1})` and then the
/// sum of inverses across rows is returned.
///
/// # Errors
/// Returns an error if `randomness` is empty or any row has length greater than
/// `randomness.len() - 1`, or if a row's linear combination is zero.
pub fn reduce_logup_bus_boundary_varlen<'a, F, EF, I>(
    randomness: &[EF],
    public_inputs: I,
) -> Result<EF, AuxFinalsError>
where
    F: Field,
    EF: ExtensionField<F>,
    I: IntoIterator<Item = &'a [F]>,
{
    if randomness.is_empty() {
        return Err(AuxFinalsError::Custom(
            "randomness must have at least one element",
        ));
    }
    let mut bus_q_last = EF::ZERO;
    for row in public_inputs {
        let q_last = row_linear_combination(randomness, row)?;
        if q_last.is_zero() {
            return Err(AuxFinalsError::Custom(
                "logup bus row linear combination evaluated to zero",
            ));
        }
        bus_q_last += q_last.inverse();
    }
    Ok(bus_q_last)
}

/// Computes the multiset bus contribution for a given actual/expected boundary value.
///
/// Returns `actual * expected^{-1}` so that the global verifier can fold contributions
/// by multiplication and check the final result equals 1.
pub fn multiset_contribution<EF: Field>(actual: EF, expected: EF) -> Result<EF, AuxFinalsError> {
    if expected.is_zero() {
        return Err(AuxFinalsError::Custom(
            "multiset expected boundary evaluated to zero",
        ));
    }
    Ok(actual * expected.inverse())
}

/// Computes the logup bus contribution for a given actual/expected boundary value.
///
/// Returns `actual - expected` so that the global verifier can fold contributions
/// by summation and check the final result equals 0.
pub fn logup_contribution<EF: Field>(actual: EF, expected: EF) -> EF {
    actual - expected
}

#[inline]
fn row_linear_combination<F, EF>(randomness: &[EF], row: &[F]) -> Result<EF, AuxFinalsError>
where
    F: Field,
    EF: ExtensionField<F>,
{
    let (r0, r_tail) = randomness.split_first().ok_or(AuxFinalsError::Custom(
        "randomness must have at least one element",
    ))?;
    if row.len() > r_tail.len() {
        return Err(AuxFinalsError::Custom(
            "randomness must have at least row_len + 1 elements",
        ));
    }
    Ok(row
        .iter()
        .zip(r_tail.iter())
        .fold(*r0, |acc, (p_i, r_i)| acc + *r_i * *p_i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AuxFinalsError;
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
        )
        .expect("unexpected multiset boundary error");

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
        )
        .expect("unexpected logup boundary error");

        assert_eq!(got, expected);
    }

    #[test]
    fn test_multiset_boundary_varlen_empty_randomness() {
        let rows: [&[F]; 0] = [];
        let err = reduce_multiset_bus_boundary_varlen::<F, EF, _>(&[], rows).unwrap_err();
        assert!(matches!(
            err,
            AuxFinalsError::Custom("randomness must have at least one element")
        ));
    }

    #[test]
    fn test_multiset_boundary_varlen_row_too_long() {
        let randomness = vec![EF::from_u64(1), EF::from_u64(2)];
        let rows = [vec![F::from_u64(3), F::from_u64(4)]];
        let err = reduce_multiset_bus_boundary_varlen::<F, EF, _>(
            &randomness,
            rows.iter().map(|r| r.as_slice()),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AuxFinalsError::Custom("randomness must have at least row_len + 1 elements")
        ));
    }

    #[test]
    fn test_logup_boundary_varlen_empty_randomness() {
        let rows: [&[F]; 0] = [];
        let err = reduce_logup_bus_boundary_varlen::<F, EF, _>(&[], rows).unwrap_err();
        assert!(matches!(
            err,
            AuxFinalsError::Custom("randomness must have at least one element")
        ));
    }

    #[test]
    fn test_logup_boundary_varlen_row_too_long() {
        let randomness = vec![EF::from_u64(1), EF::from_u64(2)];
        let rows = [vec![F::from_u64(3), F::from_u64(4)]];
        let err = reduce_logup_bus_boundary_varlen::<F, EF, _>(
            &randomness,
            rows.iter().map(|r| r.as_slice()),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AuxFinalsError::Custom("randomness must have at least row_len + 1 elements")
        ));
    }

    #[test]
    fn test_logup_boundary_varlen_zero_linear_combination() {
        let randomness = vec![EF::ZERO, EF::ONE];
        let rows = [vec![F::ZERO]];
        let err = reduce_logup_bus_boundary_varlen::<F, EF, _>(
            &randomness,
            rows.iter().map(|r| r.as_slice()),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            AuxFinalsError::Custom("logup bus row linear combination evaluated to zero")
        ));
    }

    #[test]
    fn test_multiset_contribution_expected_zero() {
        let err = multiset_contribution(EF::ONE, EF::ZERO).unwrap_err();
        assert!(matches!(
            err,
            AuxFinalsError::Custom("multiset expected boundary evaluated to zero")
        ));
    }

    #[test]
    fn test_logup_contribution_basic() {
        let actual = EF::from_u64(7);
        let expected = EF::from_u64(3);
        let got = logup_contribution(actual, expected);
        assert_eq!(got, actual - expected);
    }
}
