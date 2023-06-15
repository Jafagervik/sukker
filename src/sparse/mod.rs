//!  Module for defining sparse matrices.
//!
//! # What are sparse matrices
//!
//! Generally speaking, matrices with a lot of 0s
//!
//! # How are they represented
//!
//! Since storing large sparse matrices in memory is expensive
//!
//!
//! # What datastructure does sukker use
#![warn(missing_docs)]

mod helper;

use helper::*;

use std::fmt::Display;
use std::{collections::HashMap, error::Error, marker::PhantomData, str::FromStr};

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{Matrix, MatrixElement, MatrixError, Shape};

macro_rules! at {
    ($row:expr, $col:expr, $ncols:expr) => {
        ($row * $ncols + $col) as usize
    };
}

/// SparseMatrixData represents the datatype used to store information
/// about non-zero values in a general matrix.
///
/// The keys are the index to the position in data,
/// while the value is the value to be stored inside the matrix
pub type SparseMatrixData<'a, T> = HashMap<Shape, T>;

#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
/// Represents a sparse matrix and its data
pub struct SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Vector containing all data
    data: SparseMatrixData<'a, T>,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    _lifetime: PhantomData<&'a T>,
}

impl<'a, T> Display for SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                let elem = match self.data.get(&(i, j)) {
                    Some(&val) => val,
                    None => T::zero(),
                };

                write!(f, "{elem} ");
            }
            write!(f, "\n");
        }
        writeln!(f, "\ndtype = {}", std::any::type_name::<T>())
    }
}

impl<'a, T> Default for SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Returns a sparse 3x3 identity matrix
    fn default() -> Self {
        Self::eye(3)
    }
}

impl<'a, T> SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Constructs a new sparse matrix based on a shape
    ///
    /// All elements are set to 0 initially
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::new(3,3);
    ///
    /// assert_eq!(sparse.ncols, 3);
    /// assert_eq!(sparse.nrows, 3);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: HashMap::new(),
            nrows: rows,
            ncols: cols,
            _lifetime: PhantomData::default(),
        }
    }

    /// Constructs a new sparse matrix based on a hashmap
    /// containing the indices where value is not 0
    ///
    /// This function does not check whether or not the
    /// indices are valid and according to shape. Use `reshape`
    /// to fix this issue.
    ///
    /// Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use sukker::{SparseMatrix, SparseMatrixData};
    ///
    /// let mut indexes: SparseMatrixData<f64> = HashMap::new();
    ///
    /// indexes.insert((0,0), 2.0);
    /// indexes.insert((0,3), 4.0);
    /// indexes.insert((4,5), 6.0);
    /// indexes.insert((2,7), 8.0);
    ///
    /// let sparse = SparseMatrix::<f64>::init(indexes, (3,3));
    ///
    /// assert_eq!(sparse.shape(), (3,3));
    /// assert_eq!(sparse.get(4,5), None);
    /// assert_eq!(sparse.get(0,1), Some(0.0));
    /// ```
    pub fn init(data: SparseMatrixData<'a, T>, shape: Shape) -> Self {
        Self {
            data,
            nrows: shape.0,
            ncols: shape.1,
            _lifetime: PhantomData::default(),
        }
    }

    /// Returns a sparse eye matrix
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.ncols, 3);
    /// assert_eq!(sparse.nrows, 3);
    /// ```
    pub fn eye(size: usize) -> Self {
        let data: SparseMatrixData<'a, T> = (0..size)
            .into_par_iter()
            .map(|i| ((i, i), T::one()))
            .collect();

        Self::init(data, (size, size))
    }

    /// Same as eye
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f64>::identity(3);
    ///
    /// assert_eq!(sparse.ncols, 3);
    /// assert_eq!(sparse.nrows, 3);
    /// ```
    pub fn identity(size: usize) -> Self {
        Self::eye(size)
    }

    /// Reshapes a sparse matrix
    ///
    /// Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse = SparseMatrix::<f64>::identity(3);
    ///
    /// sparse.reshape(5,5);
    ///
    /// assert_eq!(sparse.ncols, 5);
    /// assert_eq!(sparse.nrows, 5);
    /// ```
    pub fn reshape(&mut self, nrows: usize, ncols: usize) {
        self.nrows = nrows;
        self.ncols = ncols;
    }

    /// Creates a sparse matrix from a already existent
    /// dense one.
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::{SparseMatrix, Matrix};
    ///
    /// let dense = Matrix::<i32>::eye(4);
    ///
    /// let sparse = SparseMatrix::from_dense(dense);
    ///
    /// assert_eq!(sparse.get(0,0), Some(1));
    /// assert_eq!(sparse.get(1,0), Some(0));
    /// assert_eq!(sparse.shape(), (4,4));
    /// ```
    pub fn from_dense(matrix: Matrix<'a, T>) -> Self {
        let mut data: SparseMatrixData<'a, T> = HashMap::new();

        for i in 0..matrix.nrows {
            for j in 0..matrix.ncols {
                let val = matrix.get(i, j).unwrap();
                if val != T::zero() {
                    data.insert((i, j), val);
                }
            }
        }

        Self::init(data, matrix.shape())
    }

    /// Gets an element from the sparse matrix.
    ///
    /// Returns None if index is out of bounds.
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.get(0,0), Some(1));
    /// assert_eq!(sparse.get(1,0), Some(0));
    /// assert_eq!(sparse.get(4,0), None);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        let idx = at!(i, j, self.ncols);

        if idx >= self.size() {
            eprintln!("Error, index out of bounds. Not setting value");
            return None;
        }

        match self.data.get(&(i, j)) {
            None => Some(T::zero()),
            val => val.copied(),
        }
    }

    /// Same as `get`, but will panic if indexes are out of bounds
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.at(0,0), 1);
    /// assert_eq!(sparse.at(1,0), 0);
    /// ```
    pub fn at(&self, i: usize, j: usize) -> T {
        match self.data.get(&(i, j)) {
            None => T::zero(),
            Some(val) => val.clone(),
        }
    }

    /// Sets an element
    ///
    /// Mutates or inserts a value based on indeces given
    pub fn set(&mut self, idx: Shape, value: T) {
        let i = at!(idx.0, idx.1, self.ncols);

        if i >= self.size() {
            eprintln!("Error, index out of bounds. Not setting value");
            return;
        }

        self.data
            .entry(idx)
            .and_modify(|val| *val = value)
            .or_insert(value);
    }

    /// Prints out the sparse matrix data
    ///
    /// Only prints out the hashmap with a set amount of decimals
    pub fn print(&self, decimals: usize) {
        self.data
            .iter()
            .for_each(|((i, j), val)| println!("{i} {j}: {:.decimals$}", val));
    }

    /// Gets the size of the sparse matrix
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.ncols * self.nrows
    }

    /// Get's amount of 0s in the matrix
    #[inline(always)]
    pub fn get_zero_count(&self) -> usize {
        self.size() - self.data.len()
    }

    /// Calcualtes sparcity for the given matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(4);
    ///
    /// assert_eq!(sparse.sparcity(), 0.75);
    /// ```
    #[inline(always)]
    pub fn sparcity(&self) -> f64 {
        1.0 - self.data.par_iter().count() as f64 / self.size() as f64
    }

    /// Shape of the matrix outputted as a tuple
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.shape(), (3,3));
    /// ```
    pub fn shape(&self) -> Shape {
        (self.nrows, self.ncols)
    }
}

/// Operations on sparse matrices
impl<'a, T> SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Adds two sparse matrices together
    /// and return a new one
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// let res = sparse1.add(&sparse2).unwrap();
    ///
    /// assert_eq!(res.shape(), (3,3));
    /// assert_eq!(res.get(0,0).unwrap(), 2);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self, MatrixError> {
        Self::sparse_helper(&self, other, Operation::ADD)
    }

    /// Subtracts two sparse matrices
    /// and return a new one
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// let res = sparse1.sub(&sparse2).unwrap();
    ///
    /// assert_eq!(res.shape(), (3,3));
    /// assert_eq!(res.get(0,0).unwrap(), 2);
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self, MatrixError> {
        Self::sparse_helper(&self, other, Operation::SUB)
    }
    /// Multiplies two sparse matrices together
    /// and return a new one
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// let res = sparse1.mul(&sparse2).unwrap();
    ///
    /// assert_eq!(res.shape(), (3,3));
    /// assert_eq!(res.get(0,0).unwrap(), 2);
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self, MatrixError> {
        Self::sparse_helper(&self, other, Operation::MUL)
    }
    /// Divides two sparse matrices
    /// and return a new one
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// let res = sparse1.div(&sparse2).unwrap();
    ///
    /// assert_eq!(res.shape(), (3,3));
    /// assert_eq!(res.get(0,0).unwrap(), 2);
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self, MatrixError> {
        Self::sparse_helper(&self, other, Operation::DIV)
    }

    // =============================================================
    //
    //    Matrix operations modifying the lhs
    //
    // =============================================================

    /// Adds rhs matrix on to lhs matrix.
    /// All elements from rhs gets inserted into lhs
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse1 = SparseMatrix::<i32>::eye(3);
    /// let sparse2 = SparseMatrix::<i32>::eye(3);
    ///
    /// sparse1.add_self(&sparse2);
    ///
    /// assert_eq!(sparse1.shape(), (3,3));
    /// assert_eq!(sparse1.get(0,0).unwrap(), 2);
    /// ```
    pub fn add_self(&mut self, other: &Self) {
        Self::sparse_helper_self(self, other, Operation::ADD);
    }

    // =============================================================
    //
    //    Matrix operations  with a value
    //
    // =============================================================

    /// Adds value to all non zero values in the matrix
    /// and return a new matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<f32>::eye(3);
    /// let val: f32 = 4.5;
    ///
    /// let res = sparse.add_val(val);
    ///
    /// assert_eq!(res.get(0,0).unwrap(), 5.5);
    /// ```
    pub fn add_val(&self, value: T) -> Self {
        Self::sparse_helper_val(self, value, Operation::ADD)
    }

    // =============================================================
    //
    //    Matrix operations modyfing lhs  with a value
    //
    // =============================================================

    /// Adds value to all non zero elements in matrix
    ///
    /// Examples:
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let mut sparse = SparseMatrix::<f64>::eye(3);
    /// let val = 10.0;
    ///
    /// sparse.add_val_self(val);
    ///
    /// assert_eq!(sparse.get(0,0).unwrap(), 11.0);
    /// ```
    pub fn add_val_self(&mut self, value: T) {
        Self::sparse_helper_self_val(self, value, Operation::ADD)
    }

    /// Sparse matrix multiplication
    //sparse_/
    /// Coming soon
    fn matmul_sparse(&self, other: &Self) -> Self {
        unimplemented!()
    }
}

/// Predicates for sparse matrices
impl<'a, T> SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Returns whether or not predicate holds for all values
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.shape(), (3,3));
    /// assert_eq!(sparse.all(|(idx, val)| val >= 0), true);
    /// ```
    pub fn all<F>(&self, pred: F) -> bool
    where
        F: Fn((Shape, T)) -> bool + Sync + Send,
    {
        self.data.clone().into_par_iter().all(pred)
    }

    /// Returns whether or not predicate holds for any
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::SparseMatrix;
    ///
    /// let sparse = SparseMatrix::<i32>::eye(3);
    ///
    /// assert_eq!(sparse.shape(), (3,3));
    /// assert_eq!(sparse.any(|(_, val)| val == 1), true);
    /// ```
    pub fn any<F>(&self, pred: F) -> bool
    where
        F: Fn((Shape, T)) -> bool + Sync + Send,
    {
        self.data.clone().into_par_iter().any(pred)
    }
}
