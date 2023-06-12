use sukker::{Matrix, MatrixLinAlg};

fn main() {
    let a: Matrix<f32> = Matrix::randomize((3, 1024));
    let b: Matrix<f32> = Matrix::randomize((1024, 3));

    let c = a.matmul(&b);

    c.print();

    assert!(c.size() == 9);
}
