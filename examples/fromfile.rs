use sukker::{Matrix, MatrixLinAlg};

fn main() {
    let path = "path/file.txt";

    let a: Matrix<f32> = Matrix::from_file(path);

    a.print();
}
