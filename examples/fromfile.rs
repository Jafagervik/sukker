use kaffe::{Matrix, MatrixLinAlg};

fn main() {
    let path = "path/file.txt";

    let a = Matrix::from_file(path);

    a.print();
}
