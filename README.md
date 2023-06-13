# Sukker - Matrix library written in rust

![Build Status](https://github.com/Jafagervik/sukker/actions/workflows/test.yml/badge.svg)
[![Documentation](https://docs.rs/sukker/badge.svg)](https://docs.rs/sukker/)
[![Crates.io](https://img.shields.io/crates/v/sukker.svg)](https://crates.io/crates/sukker)
[![Coverage Status](https://codecov.io/gh/Jafagervik/sukker/branch/master/graph/badge.svg)](https://codecov.io/gh/Jafagervik/sukker)
![Maintenance](https://img.shields.io/badge/maintenance-experimental-blue.svg)


Linear algebra in Rust!
Parallelized using rayon with support for many common datatypes,
sukker tries to make matrix operations easier for the user, 
while still giving you as the user the performance you deserve

## Examples


```rust 
use sukker::{Matrix, MatrixLinAlg};

fn main() {
    let a = Matrix::<f32>::randomize((7,56));
    let b = Matrix::<f32>::randomize((56,8));

    let c = a.matmul(&b);

    // To print this beautiful matrix:
    c.print(Some(5));
}
```

More examples can be found [here](/examples/)


## Documentation
Full API documentation can be found [here](https://docs.rs/sukker/latest/sukker/).

## Features 
- [X] Blazingly fast
- [X] Common matrix operations exists under matrix module
- [X] Support for f32, f64, i32, i64 and even i8

