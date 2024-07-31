use rand::{rngs::SmallRng, SeedableRng};
use structured_gen_rust::util::{ConstsLogitsModel, DeterministicModel, LangModel, MaskingAlgo};

#[test]
fn unmasked_output() {
    let tokens = ["A", "3", ".", "42", "B", ".2", "1"];

    // DeterministicModel
    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();
    let mut determ = DeterministicModel::new(vocabulary);

    let mut previous_samples = String::new();

    let pattern = None;
    let out = determ.sample_n_tokens(15, &mut previous_samples, MaskingAlgo::Naive { pattern });

    insta::assert_snapshot!(out, @"A3.42B.21A3.42B.21A");

    // ConstsLogitsModel
    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = ConstsLogitsModel::new(vocabulary, rng);

    let mut previous_samples = String::new();

    let pattern = None;
    let out2 =
        const_logits.sample_n_tokens(15, &mut previous_samples, MaskingAlgo::Naive { pattern });

    insta::assert_snapshot!(out2, @"3...2423.42A33A.1.2");
}

#[test]
fn masked_output() {
    let tokens = ["A", "3", ".", "42", "B", ".2", "1"];

    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();
    let mut determ = DeterministicModel::new(vocabulary);

    let mut previous_samples2 = String::new();

    let binding = &String::from(r"^([0-9]*)?\.?[0-9]*$");
    let pattern = Some(binding);
    let out = determ.sample_n_tokens(15, &mut previous_samples2, MaskingAlgo::Naive { pattern });

    insta::assert_snapshot!(out, @"33.421113342421113");

    // ConstsLogitsModel
    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();
    let rng = SmallRng::seed_from_u64(42);
    let weights = vec![1.0; vocabulary.len()];
    let mut const_logits = ConstsLogitsModel::new_with_weights(vocabulary, &weights[..], rng);

    let mut previous_samples = String::new();

    let binding = &String::from(r"^([0-9]*)?\.?[0-9]*$");
    let pattern = Some(binding);
    let out2 =
        const_logits.sample_n_tokens(15, &mut previous_samples, MaskingAlgo::Naive { pattern });

    insta::assert_snapshot!(out2, @"3.3142334233334211");
}
