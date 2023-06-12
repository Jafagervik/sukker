use sukker::{Matrix, MatrixLinAlg};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark for matrix multiplication
fn matmul_bench(c: &mut Criterion) {
    let x = black_box(Matrix::<f32>::randomize((258, 1000)));
    let y = black_box(Matrix::<f32>::randomize((1000, 148)));

    c.bench_function("matmul transpose", |b| b.iter(|| x.matmul(&y)));
}

criterion_group!(benches, matmul_bench);
criterion_main!(benches);
