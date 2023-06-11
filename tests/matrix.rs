use sukker::{Matrix, MatrixLinAlg, MatrixPredicates};

#[test]
fn basic() {
    let a = Matrix::init(3f32, (2, 3));
    let b = Matrix::init(5f32, (2, 3));

    let mut c = a.sub(&b);
    assert_eq!(c.size(), 6);

    c.add_val_self(2.32);

    let c = c;

    let a = Matrix::randomize((3, 2));

    let x = c.matmul(&a);

    assert_eq!(x.shape, (2, 2));

    assert!(x.any(|&e| e >= 0.05));
}
