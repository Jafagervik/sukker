use sukker::SparseMatrix;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark for matrix multiplication
// fn matmul_bench(c: &mut Criterion) {
//     let x = black_box(SparseMatrix::<f32>::randomize((258, 1000)));
//     let y = black_box(SparseMatrix::<f32>::randomize((1000, 148)));
//
//     c.bench_function("sparse matmul", |b| b.iter(|| x.matmul(&y).unwrap()));
// }
//
// criterion_group!(benches, matmul_bench);
// criterion_main!(benches);
