use sukker::SparseMatrix;

fn main() {
    let matrix = SparseMatrix::<i32>::eye(100);

    println!("sparsity: {}", matrix.sparsity());

    matrix.print(3);
}
