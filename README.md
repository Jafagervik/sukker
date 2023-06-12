# Sukker - Matrix library written in rust

![Build Status](https://github.com/Jafagervik/sukker/actions/workflows/test.yml/badge.svg)
[![Documentation](https://docs.rs/sukker/badge.svg)](https://docs.rs/sukker/)
[![Crates.io](https://img.shields.io/crates/v/sukker.svg)](https://crates.io/crates/sukker)
[![Coverage Status](https://codecov.io/gh/Jafagervik/sukker/branch/master/graph/badge.svg)](https://codecov.io/gh/Jafagervik/sukker)
![Maintenance](https://img.shields.io/badge/maintenance-experimental-blue.svg)


Linear algebra in Rust!
In version 1.1.0, there is now added support for several datatypes,
with the tradeof being no rayon added just yet.
This and more linear algebra functions to be added

## Examples


```rust 
use sukker::Matrix;

fn main() {
    let a = Matrix::init(2f32, (2,3));
    let b = Matrix::init(4f32, (2,3));

    let c = a.add(&b);

    // To print this beautiful matrix:
    c.print();
}
```

More examples can be found [here](/examples/)


## Documentation
Full API documentation can be found [here](https://docs.rs/sukker/latest/sukker/).

## Features 
- [X] Blazingly fast
- [X] Common matrix operations exists under matrix module
- [X] Support for f32, f64, i32, i64 and even i8

