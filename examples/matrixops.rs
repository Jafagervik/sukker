use sukker::Matrix;

fn main() {
    let a = Matrix::randomize((2, 3));
    let b = Matrix::randomize((2, 3));

    let mut c = a.add(&b).unwrap();

    // Will not multiply if dimensions does not line up
    c.mul_self(&b);

    let d = c.add_val(42f32);

    d.print(5);
}
