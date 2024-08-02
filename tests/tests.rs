use rand::{rngs::SmallRng, SeedableRng};
use structured_gen_rust::{
    sample_model,
    util::{generate_dict, DeterministicModel, RandomSampleModel},
    MaskingAlgorithmConfig,
};

fn small_default_setup() -> (Vec<String>, usize, &'static str) {
    let tokens = vec!["A", "3", ".", "42", "B", ".2", "1"];
    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();
    let max_samples = 15;
    let pattern = r"^([0-9]*)?\.?[0-9]*$";
    (vocabulary, max_samples, pattern)
}

// Function to pretty print long dicts
fn print_combinations(combinations: Vec<String>, values_per_row: usize) -> String {
    let mut out = String::new();
    let mut count = 0;
    for combination in combinations {
        out.push_str(&format!("{}, ", combination));
        count += 1;
        if count % values_per_row == 0 {
            out.push('\n');
        }
    }

    if count % values_per_row != 0 {
        out.push('\n');
    }
    out
}

#[test]
fn unmasked() {
    let (vocabulary, max_samples, _) = small_default_setup();

    // DeterministicModel
    let mut determ = DeterministicModel::new(&vocabulary);

    let out = sample_model(
        &mut determ,
        max_samples,
        "",
        &MaskingAlgorithmConfig::NoMasking,
    )
    .unwrap();

    insta::assert_snapshot!(out, @"3.42B.21A3.42B.21A3");

    // RandomSampleModel
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = RandomSampleModel::new(&vocabulary, rng);

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

    let mut determ = DeterministicModel::new(&vocabulary);

    let out = sample_model(
        &mut determ,
        max_samples,
        "",
        &MaskingAlgorithmConfig::Naive(pattern),
    )
    .unwrap();

    insta::assert_snapshot!(out, @"3.421342134213421342");

    // RandomSampleModel
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = RandomSampleModel::new(&vocabulary, rng);

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
    let mut determ = DeterministicModel::new(&vocabulary);

    let out = sample_model(
        &mut determ,
        max_samples,
        "",
        &MaskingAlgorithmConfig::IndexedFSM(pattern),
    )
    .unwrap();

    insta::assert_snapshot!(out, @"3.421342134213421342");

    // RandomSampleModel
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = RandomSampleModel::new(&vocabulary, rng);

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
    let mut determ = DeterministicModel::new(&vocabulary);

    let out = sample_model(
        &mut determ,
        max_samples,
        &String::from(input_prompt),
        &MaskingAlgorithmConfig::IndexedFSM(pattern),
    )
    .unwrap();

    assert!(out.len() < max_samples);
    insta::assert_snapshot!(out, @"AAAABBBBB");

    // RandomSampleModel
    let rng = SmallRng::seed_from_u64(42);
    let mut const_logits = RandomSampleModel::new(&vocabulary, rng);

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

#[test]
fn test_generate_dict() {
    let max_tokens = 200;

    let tokens = generate_dict(max_tokens);

    let out = print_combinations(tokens, 20);
    insta::assert_snapshot!(out, @r###"
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, 
    u, v, w, x, y, z, A, B, C, D, E, F, G, H, I, J, K, L, M, N, 
    O, P, Q, R, S, T, U, V, W, X, Y, Z, 0, 1, 2, 3, 4, 5, 6, 7, 
    8, 9, ., ;, ,, !, ?, aa, ab, ac, ad, ae, af, ag, ah, ai, aj, ak, al, am, 
    an, ao, ap, aq, ar, as, at, au, av, aw, ax, ay, az, aA, aB, aC, aD, aE, aF, aG, 
    aH, aI, aJ, aK, aL, aM, aN, aO, aP, aQ, aR, aS, aT, aU, aV, aW, aX, aY, aZ, a0, 
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a., a;, a,, a!, a?, ba, bb, bc, bd, be, bf, 
    bg, bh, bi, bj, bk, bl, bm, bn, bo, bp, bq, br, bs, bt, bu, bv, bw, bx, by, bz, 
    bA, bB, bC, bD, bE, bF, bG, bH, bI, bJ, bK, bL, bM, bN, bO, bP, bQ, bR, bS, bT, 
    bU, bV, bW, bX, bY, bZ, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b., b;, b,, b!, 
    "###);
}
