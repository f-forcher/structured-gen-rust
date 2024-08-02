use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_setup() -> (Vec<String>, usize, &'static str) {
    let tokens = vec!["A", "3", ".", "42", "B", ".2", "1"];
    let vocabulary: Vec<String> = tokens.into_iter().map(|s| s.to_owned()).collect();
    let max_samples = 15;
    let pattern = r"^([0-9]*)?\.?[0-9]*$";
    (vocabulary, max_samples, pattern)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("sample determ d=500 n=10000", |b| b.iter(|| fibonacci(black_box(20))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
