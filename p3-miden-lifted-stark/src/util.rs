use p3_field::{ExtensionField, TwoAdicField};
use p3_miden_transcript::Channel;

use crate::LiftedCoset;

/// Sample an out-of-domain point.
pub fn sample_ood_point<F, EF, Ch>(channel: &mut Ch, coset: &LiftedCoset) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
    Ch: Channel<F = F>,
{
    loop {
        let z: EF = channel.sample_algebra_element();
        if !coset.is_in_trace_domain::<F, _>(z) && !coset.is_in_lde_coset::<F, _>(z) {
            return z;
        }
    }
}
