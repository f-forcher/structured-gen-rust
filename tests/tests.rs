use rand::{rngs::SmallRng, SeedableRng};
use structured_gen_rust::{
    sample_model,
    util::{DeterministicModel, RandomSampleModel},
    MaskingAlgorithmConfig,
};

fn small_default_setup() -> (Vec<String>, usize, &'static str) {
    let tokens = vec!["A", "3", ".", "42", "B", ".2", "1"];
    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();
    let max_samples = 15;
    let pattern = r"^([0-9]*)?\.?[0-9]*$";
    (vocabulary, max_samples, pattern)
}

#[test]
fn unmasked() {
    let (vocabulary, max_samples, _) = small_default_setup();

    // DeterministicModel
    let mut determ = DeterministicModel::new(vocabulary.clone());

    let out = sample_model(
        &mut determ,
        max_samples,
        "",
        &MaskingAlgorithmConfig::NoMasking,
    )
    .unwrap();

    insta::assert_snapshot!(out, @"A3.42B.21A3.42B.21A");

    // RandomSampleModel
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = RandomSampleModel::new(vocabulary, rng);

    let out_rng = sample_model(
        &mut const_logits,
        max_samples,
        "",
        &MaskingAlgorithmConfig::NoMasking,
    )
    .unwrap();

    insta::assert_snapshot!(out_rng, @"3...2423.42A33A.1.2");
}

#[test]
fn naive_mask() {
    // DeterministicModel
    let (vocabulary, max_samples, pattern) = small_default_setup();

    let mut determ = DeterministicModel::new(vocabulary.clone());

    let out = sample_model(
        &mut determ,
        max_samples,
        "",
        &MaskingAlgorithmConfig::Naive(pattern),
    )
    .unwrap();

    insta::assert_snapshot!(out, @"33.421113342421113");

    // RandomSampleModel
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = RandomSampleModel::new(vocabulary, rng);

    let out_rng = sample_model(
        &mut const_logits,
        max_samples,
        "",
        &MaskingAlgorithmConfig::Naive(pattern),
    )
    .unwrap();

    insta::assert_snapshot!(out_rng, @"3.3142334233334211");
}

#[test]
fn indexed_fsm_mask() {
    let (vocabulary, max_samples, pattern) = small_default_setup();

    // DeterministicModel
    let mut determ = DeterministicModel::new(vocabulary.clone());

    let out = sample_model(
        &mut determ,
        max_samples,
        "",
        &MaskingAlgorithmConfig::IndexedFSM(pattern),
    )
    .unwrap();

    insta::assert_snapshot!(out, @"33.421113342421113");

    // RandomSampleModel
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = RandomSampleModel::new(vocabulary.clone(), rng);

    let out_rng = sample_model(
        &mut const_logits,
        max_samples,
        "",
        &MaskingAlgorithmConfig::IndexedFSM(pattern),
    )
    .unwrap();

    insta::assert_snapshot!(out_rng, @"3.3142334233334211");
}

/// Test that the previous input is processed properly. Moreover, check
/// that once no more token are allowed, the model return less than max tokens.
#[test]
fn fsm_with_input_shorter() {
    let (vocabulary, _, _) = small_default_setup();

    // We can have at most 5 As and 5 Bs, including the ones in the preexisting prompt.
    let max_samples = 50;
    let pattern = r"^A{3,5}B{0,5}$";
    let input_prompt = "AAAA";

    // DeterministicModel
    let mut determ = DeterministicModel::new(vocabulary.clone());

    let out = sample_model(
        &mut determ,
        max_samples,
        &String::from(input_prompt),
        &MaskingAlgorithmConfig::IndexedFSM(pattern),
    )
    .unwrap();

    assert!(out.len() < max_samples);
    insta::assert_snapshot!(out, @"AAAAABBBBB");

    // RandomSampleModel
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = RandomSampleModel::new(vocabulary.clone(), rng);

    let out_rng = sample_model(
        &mut const_logits,
        max_samples,
        &String::from(input_prompt),
        &MaskingAlgorithmConfig::IndexedFSM(pattern),
    )
    .unwrap();

    assert!(out_rng.len() < max_samples);
    insta::assert_snapshot!(out_rng, @"AAAAABBBBB");
}
