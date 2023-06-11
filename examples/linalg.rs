use kaffe::{Matrix, MatrixLinAlg};

fn main() {
    let a = Matrix::randomize((3, 1024));
    let b = Matrix::randomize((1024, 3));

    let c = a.matmul(&b);

    c.print();

    assert!(c.size() == 9);
}
