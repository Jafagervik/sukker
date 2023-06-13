use sukker::Matrix;

fn main() {
    let path = "path/file.txt";

    let a: Matrix<f32> = Matrix::from_file(path);

    a.print(Some(3));
}
