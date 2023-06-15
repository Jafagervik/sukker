use sukker::Matrix;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Benchmark for matrix multiplication
fn matmul_bench(c: &mut Criterion) {
    let x = black_box(Matrix::<f32>::randomize((258, 100)));
    let y = black_box(Matrix::<f32>::randomize((100, 148)));

    c.bench_function("matmul transpose", |b| b.iter(|| x.matmul(&y).unwrap()));
}

criterion_group!(benches, matmul_bench);
criterion_main!(benches);
