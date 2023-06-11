use kaffe::{Matrix, MatrixLinAlg};

fn main() {
    let a = Matrix::randomize((2, 3));
    let b = Matrix::randomize((2, 3));

    let c = a.add(&b);

    c.print();
}
