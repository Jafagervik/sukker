# Sukker - Linear Algebra library written in rust

![Build Status](https://github.com/Jafagervik/sukker/actions/workflows/test.yml/badge.svg)
[![Documentation](https://docs.rs/sukker/badge.svg)](https://docs.rs/sukker/)
[![Crates.io](https://img.shields.io/crates/v/sukker.svg)](https://crates.io/crates/sukker)
[![Coverage Status](https://codecov.io/gh/Jafagervik/sukker/branch/master/graph/badge.svg)](https://codecov.io/gh/Jafagervik/sukker)
![Maintenance](https://img.shields.io/badge/maintenance-experimental-blue.svg)
![License](https://img.shields.io/crates/l/sukker)


Linear algebra in Rust!

Parallelized using rayon with support for many common datatypes,
sukker tries to make matrix operations easier for the user, 
while still giving you as the user the performance you deserve.

Basic operations on sparse matrices are also supported now

Need a feature? Please let me/us know!


## Why V2 already?

With added error handling and a good amount of rewriting, a major version was due 
to avoid any confusion.

## Examples


### Dens Matrices 

```rust 
use sukker::Matrix;

fn main() {
    let a = Matrix::<f32>::randomize((7,56));
    let b = Matrix::<f32>::randomize((56,8));

    let c = a.matmul(&b).unwrap();

    // To print this beautiful matrix:
    c.print(5);
}
```

### Sparse Matrices 


```rust 
use std::collections::HashMap;
use sukker::{SparseMatrix, SparseMatrixData};

fn main() {
    let mut indexes: SparseMatrixData<f64> = HashMap::new();

    indexes.insert((0, 1), 2.0);
    indexes.insert((1, 0), 4.0);
    indexes.insert((2, 3), 6.0);
    indexes.insert((3, 3), 8.0);

    let sparse = SparseMatrix::<f64>::init(indexes, (4, 4));

    sparse.print(3);
}
```

More examples can be found [here](/examples/)


## Documentation
Full API documentation can be found [here](https://docs.rs/sukker/latest/sukker/).

## Features 
- [X] Easy to use!
- [X] Blazingly fast
- [X] Multiple features on dense matrices
- [X] Serde support 
- [X] Support for all signed numeric datatypes 
- [X] Can be sent over threads
- [X] Sparse matrices with matrix multiplications

