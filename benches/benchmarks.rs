use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use structured_gen_rust::{
    sample_model,
    util::{generate_dict, DeterministicModel},
    LangModel, MaskingAlgorithmConfig,
};

fn _bench_setup_small_dict() -> (
    Vec<String>,
    &'static str,
    impl LangModel,
    String,
    MaskingAlgorithmConfig<'static>,
) {
    let tokens = vec!["A", "3", ".", "42", "B", ".2", "1"];
    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();

    let pattern = r"^([0-9]*)?\.?[0-9]*$";
    let prompt = String::from("");
    let algo = MaskingAlgorithmConfig::IndexedFSM(pattern);

    let model = DeterministicModel::new(&vocabulary);

    (vocabulary, pattern, model, prompt, algo)
}

fn bench_setup_gen_dict(
    vocab_size: usize,
) -> (
    Vec<String>,
    &'static str,
    impl LangModel,
    String,
    MaskingAlgorithmConfig<'static>,
) {
    let vocabulary: Vec<String> = generate_dict(vocab_size);
    let pattern = r"^([0-9]*)?\.?[0-9]*$";
    let prompt = String::from("");
    let algo = MaskingAlgorithmConfig::IndexedFSM(pattern);

    let model = DeterministicModel::new(&vocabulary);

    (vocabulary, pattern, model, prompt, algo)
}

pub fn simple_benchmark(c: &mut Criterion) {
    let vocab_size = 500;
    let (_, _, mut model, prompt, algo) = bench_setup_gen_dict(vocab_size);
    let max_tokens: usize = 3000;

    c.bench_function(
        &format!("sample determ indexed-fsm v={vocab_size} n={max_tokens}"),
        |b| b.iter(|| sample_model(&mut model, black_box(max_tokens), &prompt, &algo).unwrap()),
    );
}

fn naive_vs_fsm(c: &mut Criterion) {
    let vocab_size = 500;
    let mut group = c.benchmark_group("Sampling algo comparison");
    group.sample_size(10);

    let (_, pattern, mut model, prompt, _) = bench_setup_gen_dict(vocab_size);
    let naive_algo = MaskingAlgorithmConfig::Naive(pattern);
    let fsm_algo = MaskingAlgorithmConfig::IndexedFSM(pattern);

    for max_tokens in (10..50).step_by(10) {
        group.bench_with_input(
            BenchmarkId::new("Naive", max_tokens),
            &max_tokens,
            |b, max_tokens| {
                b.iter(|| {
                    sample_model(&mut model, black_box(*max_tokens), &prompt, &naive_algo).unwrap()
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Indexed FSM", max_tokens),
            &max_tokens,
            |b, max_tokens: &usize| {
                b.iter(|| {
                    sample_model(&mut model, black_box(*max_tokens), &prompt, &fsm_algo).unwrap()
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, simple_benchmark, naive_vs_fsm);
criterion_main!(benches);
