use sukker::{Matrix, MatrixLinAlg};

fn main() {
    let a: Matrix<f32> = Matrix::randomize((3, 1024));
    let b: Matrix<f32> = Matrix::randomize((1024, 3));

    let c = a.matmul(&b);

    println!("{}", c);
    println!("{:?}", c);

    // Print with however many decimals you want
    c.print(4);

    assert!(c.size() == 9);
}
