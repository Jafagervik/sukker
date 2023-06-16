use sukker::Matrix;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Benchmark for matrix multiplication
fn matmul_bench(c: &mut Criterion) {
    let x = black_box(Matrix::<f64>::randomize((999, 1000)));
    let y = black_box(Matrix::<f64>::randomize((1000, 999)));

    c.bench_function("MxN @ NxP dense matmul with transpose", |b| {
        b.iter(|| x.matmul(&y).unwrap())
    });
}

criterion_group!(benches, matmul_bench);
criterion_main!(benches);
