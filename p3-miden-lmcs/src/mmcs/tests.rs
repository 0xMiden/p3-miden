//! MMCS integration tests.

use alloc::vec::Vec;

use p3_commit::{BatchOpeningRef, Mmcs};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_miden_dev_utils::configs::baby_bear_poseidon2::{
    Compress, DIGEST, F, P, Sponge, WIDTH, test_challenger, test_components,
};
use p3_miden_transcript::{ProverTranscript, VerifierTranscript};
use p3_symmetric::Hash;
use p3_util::log2_strict_usize;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::{BatchProof, HidingLmcsConfig, Lmcs, LmcsConfig, LmcsError, LmcsTree};

type BaseMmcs = LmcsConfig<P, P, Sponge, Compress, WIDTH, DIGEST>;
type RowMatrix = RowMajorMatrix<F>;
const SALT: usize = 4;
type HidingMmcs = HidingLmcsConfig<P, P, Sponge, Compress, SmallRng, WIDTH, DIGEST, SALT>;
const BASE_SHAPES: &[(usize, usize)] = &[(4, 5), (8, 3)];
type OpeningProof = <BaseMmcs as Mmcs<F>>::Proof;

fn mmcs() -> BaseMmcs {
    let (_, sponge, compress) = test_components();
    LmcsConfig::new(sponge, compress)
}

fn hiding_mmcs(rng: SmallRng) -> HidingMmcs {
    let (_, sponge, compress) = test_components();
    HidingLmcsConfig::new(sponge, compress, rng)
}

fn tree_context<C, T>(tree: &T) -> (C, Vec<Dimensions>, usize)
where
    T: LmcsTree<F, C, RowMatrix>,
{
    let commitment = tree.root();
    let dimensions = tree
        .leaves()
        .iter()
        .zip(tree.widths())
        .map(|(m, width)| Dimensions {
            width,
            height: m.height(),
        })
        .collect();
    let index = tree.height() / 2;
    (commitment, dimensions, index)
}

fn build_tree_with_alignment<C>(
    mmcs: &C,
    seed: u64,
    shapes: &[(usize, usize)],
    aligned: bool,
) -> C::Tree<RowMatrix>
where
    C: Lmcs<F = F>,
{
    let mut rng = SmallRng::seed_from_u64(seed);
    // Generate deterministic random matrices for the requested shapes.
    let matrices = shapes
        .iter()
        .map(|&(h, w)| RowMajorMatrix::rand(&mut rng, h, w))
        .collect();
    if aligned {
        mmcs.build_aligned_tree(matrices)
    } else {
        mmcs.build_tree(matrices)
    }
}

#[test]
fn extract_proofs_roundtrip() {
    let mmcs = mmcs();

    let test = |seed: u64, shapes: &[(usize, usize)], indices: &[usize]| {
        let tree = build_tree_with_alignment(&mmcs, seed, shapes, false);
        let widths = tree.widths();
        let log_max_height = log2_strict_usize(tree.height());
        let (commitment, dimensions, _) = tree_context(&tree);

        let mut prover_channel = ProverTranscript::new(test_challenger());
        tree.prove_batch(indices.iter().copied(), &mut prover_channel);
        let transcript = prover_channel.into_data();

        let mut verifier_channel = VerifierTranscript::from_data(test_challenger(), &transcript);
        let batch = BatchProof::<F, Hash<F, F, DIGEST>>::read_from_channel(
            &widths,
            log_max_height,
            indices,
            &mut verifier_channel,
        )
        .expect("batch proof should parse from transcript");
        let proofs = batch
            .single_proofs(&mmcs, &widths, log_max_height)
            .expect("batch proof should reconstruct proofs");
        assert_eq!(proofs.len(), indices.len());

        for (pos, &idx) in indices.iter().enumerate() {
            let proof = proofs.get(&idx).expect("proof for index");
            let proof_expected = tree.single_proof(idx);
            assert_eq!(
                proof, &proof_expected,
                "path mismatch for index {idx} at position {pos}"
            );

            let opening_proof = (proof.salt, proof.siblings.clone());
            let rows_vec: Vec<Vec<F>> = proof.rows.iter_rows().map(|r| r.to_vec()).collect();
            let batch_opening = BatchOpeningRef {
                opened_values: &rows_vec,
                opening_proof: &opening_proof,
            };
            Mmcs::verify_batch(&mmcs, &commitment, &dimensions, idx, batch_opening)
                .expect("proof should verify");

            assert_eq!(proof.rows, tree.rows(idx));
        }
    };

    test(42, &[(4, 3), (8, 5)], &[0, 1, 5]); // adjacent + non-adjacent
    test(55, &[(4, 2), (16, 3)], &[0, 5, 10, 15]); // larger tree
}

#[test]
fn mmcs_roundtrip_non_hiding() {
    let mmcs = mmcs();
    let tree = build_tree_with_alignment(&mmcs, 10, BASE_SHAPES, false);
    let (commitment, dimensions, index) = tree_context(&tree);

    let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);
    Mmcs::verify_batch(
        &mmcs,
        &commitment,
        &dimensions,
        index,
        (&batch_opening).into(),
    )
    .expect("mmcs verify should succeed");

    let expected_rows = tree.rows(index);
    for (row, expected_row) in batch_opening
        .opened_values
        .iter()
        .zip(expected_rows.iter_rows())
    {
        assert_eq!(row.as_slice(), expected_row);
    }
}

#[test]
fn mmcs_roundtrip_hiding() {
    let mmcs = hiding_mmcs(SmallRng::seed_from_u64(12));
    let tree = build_tree_with_alignment(&mmcs, 11, BASE_SHAPES, false);
    let (commitment, dimensions, index) = tree_context(&tree);

    let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);
    Mmcs::verify_batch(
        &mmcs,
        &commitment,
        &dimensions,
        index,
        (&batch_opening).into(),
    )
    .expect("mmcs verify should succeed");
}

#[test]
fn mmcs_verify_rejects_invalid_openings() {
    let mmcs = mmcs();
    type MutateFn = fn(
        usize,
        usize,
        &[Dimensions],
        &mut Vec<Dimensions>,
        &mut usize,
        &mut Vec<Vec<F>>,
        &mut OpeningProof,
    ) -> Result<(), LmcsError>;

    struct Case {
        name: &'static str,
        seed: u64,
        aligned: bool,
        mutate: MutateFn,
    }

    let cases: &[Case] = &[
        Case {
            name: "wrong_row_count",
            seed: 20,
            aligned: false,
            mutate: |_height, _alignment, _unaligned, _dims, _index, opened_values, _proof| {
                opened_values.pop();
                Err(LmcsError::InvalidProof)
            },
        },
        Case {
            name: "wrong_row_width",
            seed: 21,
            aligned: false,
            mutate: |_height, _alignment, _unaligned, _dims, _index, opened_values, _proof| {
                if let Some(row) = opened_values.first_mut() {
                    row.pop();
                }
                Err(LmcsError::InvalidProof)
            },
        },
        Case {
            name: "wrong_siblings_len",
            seed: 22,
            aligned: false,
            mutate: |_height, _alignment, _unaligned, _dims, _index, _opened_values, proof| {
                proof.1.pop();
                Err(LmcsError::InvalidProof)
            },
        },
        Case {
            name: "out_of_range_index",
            seed: 24,
            aligned: false,
            mutate: |height, _alignment, _unaligned, _dims, index, _opened_values, _proof| {
                *index = height;
                Err(LmcsError::InvalidProof)
            },
        },
        Case {
            name: "misordered_dimensions",
            seed: 25,
            aligned: false,
            mutate: |_height, _alignment, _unaligned, dims, _index, _opened_values, _proof| {
                if dims.len() > 1 {
                    dims.swap(0, 1);
                }
                Err(LmcsError::InvalidProof)
            },
        },
        Case {
            name: "unaligned_dimensions",
            seed: 26,
            aligned: true,
            mutate: |_height, alignment, unaligned, dims, _index, _opened_values, _proof| {
                dims.clone_from_slice(unaligned);
                if alignment > 1 {
                    Err(LmcsError::InvalidProof)
                } else {
                    Ok(())
                }
            },
        },
        Case {
            name: "root_mismatch",
            seed: 23,
            aligned: false,
            mutate: |_height, _alignment, _unaligned, _dims, _index, opened_values, _proof| {
                if let Some(cell) = opened_values.first_mut().and_then(|row| row.first_mut()) {
                    *cell += F::ONE;
                }
                Err(LmcsError::RootMismatch)
            },
        },
    ];

    for case in cases {
        let tree = build_tree_with_alignment(&mmcs, case.seed, BASE_SHAPES, case.aligned);
        let (commitment, dimensions, index) = tree_context(&tree);
        let unaligned_dimensions: Vec<Dimensions> = tree
            .leaves()
            .iter()
            .map(|m| Dimensions {
                width: m.width(),
                height: m.height(),
            })
            .collect();
        let height = tree.height();
        let alignment = tree.alignment();
        let batch_opening = Mmcs::open_batch(&mmcs, index, &tree);

        Mmcs::verify_batch(
            &mmcs,
            &commitment,
            &dimensions,
            index,
            (&batch_opening).into(),
        )
        .expect("baseline opening should verify");

        let mut dims = dimensions.clone();
        let mut idx = index;
        let mut opened_values = batch_opening.opened_values.clone();
        let mut opening_proof = batch_opening.opening_proof.clone();

        let expected = (case.mutate)(
            height,
            alignment,
            &unaligned_dimensions,
            &mut dims,
            &mut idx,
            &mut opened_values,
            &mut opening_proof,
        );
        let result = Mmcs::verify_batch(
            &mmcs,
            &commitment,
            &dims,
            idx,
            BatchOpeningRef::new(&opened_values, &opening_proof),
        );

        match expected {
            Ok(()) => result.expect(case.name),
            Err(err) => assert_eq!(result, Err(err), "{}", case.name),
        }
    }
}
