use std::collections::HashMap;
use sukker::{SparseMatrix, SparseMatrixData};

use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Benchmark for matrix multiplication
fn sparse_matmul_bench(c: &mut Criterion) {
    let mut indexes: SparseMatrixData<f64> = HashMap::new();

    indexes.insert((0, 1), 2.0);
    indexes.insert((1, 9), 4.0);
    indexes.insert((1, 8), 6.0);
    indexes.insert((9, 9), 8.0);
    indexes.insert((6, 2), 8.0);
    indexes.insert((2, 2), 8.0);
    indexes.insert((1, 3), 8.0);
    indexes.insert((8, 6), 8.0);
    indexes.insert((9, 1), 8.0);
    indexes.insert((2, 3), 8.0);
    indexes.insert((7, 9), 8.0);
    indexes.insert((7, 8), 8.0);
    indexes.insert((3, 8), 8.0);
    indexes.insert((0, 9), 8.0);

    let x = black_box(SparseMatrix::<f64>::init(indexes, (10, 10)));

    let mut indexes2: SparseMatrixData<f64> = HashMap::new();

    indexes2.insert((0, 0), 2.0);
    indexes2.insert((1, 0), 4.0);
    indexes2.insert((1, 1), 8.0);
    indexes2.insert((2, 1), 6.0);
    indexes2.insert((8, 1), 3.0);
    indexes2.insert((2, 5), 1.0);
    indexes2.insert((8, 1), 12.0);
    indexes2.insert((9, 8), 4.0);
    indexes2.insert((8, 2), 2.0);
    indexes2.insert((2, 4), 6.0);
    indexes2.insert((3, 9), 1.0);
    indexes2.insert((7, 0), 6.0);
    indexes2.insert((7, 0), 8.0);
    indexes2.insert((9, 7), 6.0);

    let y = black_box(SparseMatrix::<f64>::init(indexes2, (10, 10)));

    c.bench_function("sparse matmul", |b| b.iter(|| x.matmul_sparse(&y).unwrap()));
}

criterion_group!(benches, sparse_matmul_bench);
criterion_main!(benches);
