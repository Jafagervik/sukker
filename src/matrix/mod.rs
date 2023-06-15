//! Making working with matrices in rust easier!
//!
//! For now, only basic operations are allowed, but more are to be added
//!
//! This file is sub 1500 lines and acts as the core file

mod error;

pub use error::*;

use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    fmt::{Debug, Display},
    fs,
    marker::PhantomData,
    ops::Div,
    str::FromStr,
};

use itertools::{iproduct, Itertools};
use num_traits::{
    pow,
    sign::{abs, Signed},
    Num, NumAssign, NumAssignOps, NumAssignRef, NumOps, One, Zero,
};
use rand::{distributions::uniform::SampleUniform, Rng};
use rayon::prelude::*;
use std::iter::{Product, Sum};

/// Shape represents the dimension size
/// of the matrix as a tuple of usize
pub type Shape = (usize, usize);

/// Helper method to swap to usizes
fn swap(lhs: &mut usize, rhs: &mut usize) {
    let temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}

/// Calculates 1D index from row and col
macro_rules! at {
    ($row:ident, $col:ident, $ncols:expr) => {
        $row * $ncols + $col
    };
}

#[derive(Clone, PartialEq, PartialOrd, Debug, Serialize, Deserialize)]
pub struct Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Vector containing all data
    data: Vec<T>,
    /// Shape of the matrix
    pub shape: Shape,
    /// Number of rows
    pub nrows: usize,
    /// Number of columns
    pub ncols: usize,
    _lifetime: PhantomData<&'a T>,
}

pub trait MatrixElement:
    Copy
    + Clone
    + PartialOrd
    + Signed
    + Sum
    + Product
    + Display
    + Debug
    + FromStr
    + Default
    + One
    + PartialEq
    + Zero
    + Send
    + Sync
    + Sized
    + Num
    + NumOps
    + NumAssignOps
    + NumAssignRef
    + NumAssign
    + SampleUniform
{
}

impl<'a, T> Error for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

unsafe impl<'a, T> Send for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

unsafe impl<'a, T> Sync for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
}

impl MatrixElement for i8 {}
impl MatrixElement for i16 {}
impl MatrixElement for i32 {}
impl MatrixElement for i64 {}
impl MatrixElement for i128 {}
impl MatrixElement for f32 {}
impl MatrixElement for f64 {}

impl<'a, T> FromStr for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Parse the input string and construct the matrix dynamically
        let v: Vec<T> = s
            .trim()
            .lines()
            .map(|l| {
                l.split_whitespace()
                    .map(|num| num.parse::<T>().unwrap())
                    .collect::<Vec<T>>()
            })
            .collect::<Vec<Vec<T>>>()
            .into_iter()
            .flatten()
            .collect();

        let rows = s.trim().lines().count();
        let cols = s.trim().lines().nth(0).unwrap().split_whitespace().count();

        Ok(Self::new(v, (rows, cols)))
    }
}

impl<'a, T> Display for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[");

        // Large matrices
        if self.nrows > 10 || self.ncols > 10 {
            write!(f, "...");
        }

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if i == 0 {
                    write!(f, "{:.4} ", self.get(i, j));
                } else {
                    write!(f, " {:.4}", self.get(i, j));
                }
            }
            // Print ] on same line if youre at the end
            if i == self.shape.0 - 1 {
                break;
            }
            write!(f, "\n");
        }
        writeln!(f, "], dtype={}", std::any::type_name::<T>())
    }
}

impl<'a, T> Default for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Represents a default identity matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f32> = Matrix::default();
    ///
    /// assert_eq!(matrix.size(), 9);
    /// assert_eq!(matrix.shape, (3,3));
    /// ```
    fn default() -> Self {
        Self {
            data: vec![T::one(); 9],
            shape: (3, 3),
            nrows: 3,
            ncols: 3,
            _lifetime: PhantomData::default(),
        }
    }
}

/// Printer functions for the matrix
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Prints out the matrix with however many decimals you want
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<i32> = Matrix::eye(2);
    /// matrix.print(4);
    ///
    /// ```
    pub fn print(&self, decimals: usize) {
        print!("[");

        // Large matrices
        if self.nrows > 10 || self.ncols > 10 {
            print!("...");
        }

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if i == 0 {
                    print!("{val:.dec$} ", dec = decimals, val = self.get(i, j));
                } else {
                    print!(" {val:.dec$}", dec = decimals, val = self.get(i, j));
                }
            }
            // Print ] on same line if youre at the end
            if i == self.shape.0 - 1 {
                break;
            }
            print!("\n");
        }
        println!("], dtype={}", std::any::type_name::<T>());
    }
}

/// Implementations of all creatins of matrices
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Creates a new matrix from a vector and the shape you want.
    /// Will default init if it does not work
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::new(vec![1.0,2.0,3.0,4.0], (2,2)).unwrap();
    ///
    /// assert_eq!(matrix.size(), 4);
    /// assert_eq!(matrix.shape, (2,2));
    /// ```
    pub fn new(data: Vec<T>, shape: Shape) -> Result<Self, MatrixError> {
        if shape.0 * shape.1 != data.len() {
            return Err(MatrixError::MatrixCreationError.into());
        }

        Ok(Self {
            data,
            shape,
            nrows: shape.0,
            ncols: shape.1,
            _lifetime: PhantomData::default(),
        })
    }

    /// Initializes a matrix with the same value
    /// given from parameter 'value'
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(4f32, (1,2));
    ///
    /// assert_eq!(matrix.data, vec![4f32,4f32]);
    /// assert_eq!(matrix.shape, (1,2));
    /// ```
    pub fn init(value: T, shape: Shape) -> Self {
        Self::from_shape(value, shape)
    }

    /// Returns an eye matrix which for now is the same as the
    /// identity matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f32> = Matrix::eye(2);
    ///
    /// assert_eq!(matrix.data, vec![1f32, 0f32, 0f32, 1f32]);
    /// assert_eq!(matrix.shape, (2,2));
    /// ```
    pub fn eye(size: usize) -> Self {
        let mut data: Vec<T> = vec![T::zero(); size * size];

        (0..size).for_each(|i| data[i * size + i] = T::one());

        // Safe to do since the library is setting the size
        Self::new(data, (size, size)).unwrap()
    }

    /// Identity is same as eye, just for nerds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<i64> = Matrix::identity(2);
    ///
    /// assert_eq!(matrix.data, vec![1i64, 0i64, 0i64, 1i64]);
    /// assert_eq!(matrix.shape, (2,2));
    /// ```
    pub fn identity(size: usize) -> Self {
        Self::eye(size)
    }

    /// Tries to create a matrix from a slize and shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let s = vec![1f32, 2f32, 3f32, 4f32];
    /// let matrix = Matrix::from_slice(&s, (4,1)).unwrap_or_else(|| Matrix::default());
    ///
    /// assert_eq!(matrix.unwrap().shape, (4,1));
    /// ```
    pub fn from_slice(arr: &[T], shape: Shape) -> Result<Self, MatrixError> {
        if shape.0 * shape.1 != arr.len() {
            return Err(MatrixError::MatrixCreationError.into());
        }

        Ok(Self::new(arr.to_owned(), shape).unwrap())
    }

    /// Creates a matrix where all values are 0.
    /// All sizes are based on a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<i32> = Matrix::zeros((4,1));
    ///
    /// assert_eq!(matrix.shape, (4,1));
    /// assert_eq!(matrix.data, vec![0i32; 4]);
    /// ```
    pub fn zeros(shape: Shape) -> Self {
        Self::from_shape(T::zero(), shape)
    }

    /// Creates a matrix where all values are 1.
    /// All sizes are based on a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f64> = Matrix::ones((4,1));
    ///
    /// assert_eq!(matrix.shape, (4,1));
    /// assert_eq!(matrix.data, vec![1f64; 4]);
    /// ```
    pub fn ones(shape: Shape) -> Self {
        Self::from_shape(T::one(), shape)
    }

    /// Creates a matrix where all values are 0.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1: Matrix<i8> = Matrix::default();
    /// let matrix2 = Matrix::zeros_like(&matrix1);
    ///
    /// assert_eq!(matrix2.shape, matrix1.shape);
    /// assert_eq!(matrix2.get(0,0), 0i8);
    /// ```
    pub fn zeros_like(other: &Self) -> Self {
        Self::from_shape(T::zero(), other.shape)
    }

    /// Creates a matrix where all values are 1.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1: Matrix<i64> = Matrix::default();
    /// let matrix2 = Matrix::ones_like(&matrix1);
    ///
    /// assert_eq!(matrix2.shape, matrix1.shape);
    /// assert_eq!(1i64, matrix2.get(0,0));
    /// ```
    pub fn ones_like(other: &Self) -> Self {
        Self::from_shape(T::one(), other.shape)
    }

    /// Creates a matrix where all values are random between 0 and 1.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1: Matrix<f32> = Matrix::default();
    /// let matrix2 = Matrix::random_like(&matrix1);
    ///
    /// assert_eq!(matrix1.shape, matrix2.shape);
    /// ```
    pub fn random_like(matrix: &Self) -> Self {
        Self::randomize_range(T::zero(), T::one(), matrix.shape)
    }

    /// Creates a matrix where all values are random between start..=end.
    /// Shape in new array is given through parameter 'shape'
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::randomize_range(1f32, 2f32, (2,3));
    /// let elem = matrix.get(1,1);
    ///
    /// assert_eq!(matrix.shape, (2,3));
    /// //assert!(elem >= 1f32 && 2f32 <= elem);
    /// ```
    pub fn randomize_range(start: T, end: T, shape: Shape) -> Self {
        let mut rng = rand::thread_rng();

        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data: Vec<T> = (0..len).map(|_| rng.gen_range(start..=end)).collect();

        // Safe because shape doesn't have to match data from a user
        Self::new(data, shape).unwrap()
    }

    /// Creates a matrix where all values are random between 0..=1.
    /// Shape in new array is given through parameter 'shape'
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix: Matrix<f64> = Matrix::randomize((2,3));
    ///
    /// assert_eq!(matrix.shape, (2,3));
    /// ```
    pub fn randomize(shape: Shape) -> Self {
        Self::randomize_range(T::zero(), T::one(), shape)
    }

    /// Parses from file, but will return a default matrix if nothing is given
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// // let m: Matrix<f32> = Matrix::from_file("../../test.txt").unwrap();
    ///
    /// // m.print(4);
    /// ```
    pub fn from_file(path: &'static str) -> Result<Self, MatrixError> {
        let data =
            fs::read_to_string(path).map_err(|_| MatrixError::MatrixFileReadError(path).into())?;

        data.parse::<Self>()
            .map_err(|_| MatrixError::MatrixParseError.into())
    }

    /// Helper function to create matrices
    fn from_shape(value: T, shape: Shape) -> Self {
        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data = vec![value; len];

        Self::new(data, shape).unwrap()
    }
}

/// Enum for specifying which dimension / axis to work with
pub enum Dimension {
    /// Row is defined as 0
    Row = 0,
    /// Col is defined as 1
    Col = 1,
}

/// Regular matrix methods that are not operating math on them
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement + Div<Output = T> + Sum<T>,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Reshapes a matrix if possible.
    /// If the shapes don't match up, the old shape will be retained
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(10.5, (2,3));
    /// matrix.reshape((3,2));
    ///
    /// assert_eq!(matrix.shape, (3,2));
    /// ```
    pub fn reshape(&mut self, new_shape: Shape) {
        if new_shape.0 * new_shape.1 != self.size() {
            println!("Can not reshape.. Keeping old dimensions for now");
            return;
        }

        self.shape = new_shape;
    }

    /// Get the total size of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,3));
    ///
    /// assert_eq!(matrix.size(), 6);
    /// ```
    pub fn size(&self) -> usize {
        self.nrows * self.ncols
    }

    ///  Gets element based on is and js
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,3));
    ///
    /// assert_eq!(matrix.get(1,2).unwrap(), 10.5);:
    /// ```
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        let idx = at!(i, j, self.ncols);

        if idx >= self.size() {
            return None;
        }

        Some(self.data[at!(i, j, self.ncols)])
    }

    ///  Gets a piece of the matrix out as a vector
    ///
    ///  If some indeces are out of bounds, the vec up until that point
    ///  will be returned
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (4,4));
    /// let slice = matrix.get_vec_slice((1,1), (2,2));
    ///
    /// assert_eq!(slice, vec![10.5,10.5,10.5,10.5]);
    /// ```
    pub fn get_vec_slice(&self, start_idx: Shape, size: Shape) -> Vec<T> {
        let (start_row, start_col) = start_idx;
        let (dx, dy) = size;

        iproduct!(start_row..start_row + dy, start_col..start_col + dx)
            .filter_map(|(i, j)| self.get(i, j))
            .collect()
    }

    ///  Gets a piece of the matrix out as a matrix
    ///
    ///  If some indeces are out of bounds, unlike `get_vec_slice`
    ///  this function will return an IndexOutOfBoundsError
    ///  and will not return data
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (4,4));
    /// let sub_matrix = matrix.get_sub_matrix((1,1), (2,2)).unwrap();
    ///
    /// assert_eq!(sub_matrix.data, vec![10.5,10.5,10.5,10.5]);
    /// ```
    pub fn get_sub_matrix(&self, start_idx: Shape, size: Shape) -> Result<Self, MatrixError> {
        let (start_row, start_col) = start_idx;
        let (dx, dy) = size;

        let data = iproduct!(start_row..start_row + dy, start_col..start_col + dx)
            .filter_map(|(i, j)| self.get(i, j))
            .collect();

        return match Self::new(data, size) {
            Ok(a) => Ok(a),
            Err(e) => Err(MatrixError::MatrixIndexOutOfBoundsError.into()),
        };
    }

    /// Concat two mtrices on a dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    /// use sukker::Dimension;
    ///
    /// let matrix = Matrix::init(10.5, (4,4));
    /// let matrix2 = Matrix::init(10.5, (1,4));
    ///
    /// let res = matrix.concat(&matrix2, Dimension::Row).unwrap();
    ///
    /// assert_eq!(res.shape, (5,4));
    /// ```
    pub fn concat(&self, other: &Self, dim: Dimension) -> Result<Self, MatrixError> {
        match dim {
            Dimension::Row => {
                if self.ncols != other.ncols {
                    return Err(MatrixError::MatrixConcatinationError.into());
                }

                let mut new_data = self.data;

                new_data.extend(other.data.iter());

                let nrows = self.nrows + other.nrows;
                let shape = (nrows, self.ncols);

                return Ok(Self::new(new_data, shape).unwrap());
            }

            Dimension::Col => {
                if self.nrows != other.nrows {
                    return Err(MatrixError::MatrixConcatinationError.into());
                }

                let mut new_data: Vec<T> = Vec::new();

                let take_self = self.ncols;
                let take_other = other.ncols;

                for (idx, _) in self.data.iter().step_by(take_self).enumerate() {
                    // Add from self, then other
                    let row = (idx / take_self) * take_self;
                    new_data.extend(self.data.iter().skip(row).take(take_self));
                    new_data.extend(other.data.iter().skip(row).take(take_other));
                }

                let ncols = self.ncols + other.ncols;
                let shape = (self.nrows, ncols);

                return Ok(Self::new(new_data, shape).unwrap());
            }
        };
    }

    // TODO: Add option to transpose to be able to extend
    // Doens't change anything if dimension mismatch

    /// Extend a matrix with another on a dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    /// use sukker::Dimension;
    ///
    /// let mut matrix = Matrix::init(10.5, (4,4));
    /// let matrix2 = Matrix::init(10.5, (4,1));
    ///
    /// matrix.extend(&matrix2, Dimension::Col)
    ///
    /// assert_eq!(matrix.shape, (4,5));
    /// ```
    pub fn extend(&mut self, other: &Self, dim: Dimension) {
        match dim {
            Dimension::Row => {
                if self.ncols != other.ncols {
                    eprintln!("Error: Dimension mismatch");
                    return;
                }

                self.data.extend(other.data.iter());

                self.nrows += other.nrows;
                self.shape = (self.nrows, self.ncols);
            }

            Dimension::Col => {
                if self.nrows != other.nrows {
                    eprintln!("Error: Dimension mismatch");
                    return;
                }

                let mut new_data: Vec<T> = Vec::new();

                let take_self = self.ncols;
                let take_other = other.ncols;

                for (idx, _) in self.data.iter().step_by(take_self).enumerate() {
                    // Add from self, then other
                    let row = (idx / take_self) * take_self;
                    new_data.extend(self.data.iter().skip(row).take(take_self));
                    new_data.extend(other.data.iter().skip(row).take(take_other));
                }

                self.ncols += other.ncols;
                self.shape = (self.nrows, self.ncols);
            }
        };
    }

    ///  Sets element based on is and js
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(10.5, (2,3));
    /// matrix.set(1,2, 11.5);
    ///
    /// assert_eq!(matrix.get(1,2), 11.5);
    /// ```
    pub fn set(&mut self, i: usize, j: usize, value: T) {
        self.data[at!(i, j, self.ncols)] = value;
    }

    /// Calculates the (row, col) for a matrix by a single index
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,2));
    /// let inv = matrix.inverse_at(1);
    ///
    /// assert_eq!(inv, (0,1));
    /// ```
    pub fn inverse_at(&self, idx: usize) -> Shape {
        let row = idx / self.ncols;
        let col = idx % self.ncols;

        (row, col)
    }

    /// Finds maximum element in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,3));
    ///
    /// assert_eq!(matrix.max(), 10.5);
    /// ```
    pub fn max(&self) -> T {
        // Matrix must have at least one element, thus we can unwrap
        *self
            .data
            .par_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds minimum element in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let mut matrix = Matrix::init(10.5, (2,3));
    /// matrix.data[2] = 1.0;
    ///
    /// assert_eq!(matrix.min(), 1.0);
    /// ```
    pub fn min(&self) -> T {
        // Matrix must have at least one element, thus we can unwrap
        *self
            .data
            .par_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds position in matrix where value is highest.
    /// Restricted to find this across a row or column
    /// in the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, Dimension};
    ///
    /// let mut matrix = Matrix::init(1.0, (3,3));
    /// matrix.data[2] = 15.0;
    ///
    /// ```
    pub fn argmax(&self, rowcol: usize, dimension: Dimension) -> Option<Shape> {
        match dimension {
            Dimension::Row => {
                if rowcol >= self.nrows - 1 {
                    return None;
                }

                let mut highest: T = T::one();
                let mut i = 0;

                for (idx, elem) in self
                    .data
                    .iter()
                    .enumerate()
                    .skip(rowcol * self.ncols)
                    .take(self.ncols)
                {
                    if *elem >= highest {
                        i = idx;
                    }
                }

                Some(self.inverse_at(i))
            }

            Dimension::Col => {
                if rowcol >= self.ncols - 1 {
                    return None;
                }

                let mut highest: T = T::one();

                let mut i = 0;

                for (idx, elem) in self
                    .data
                    .iter()
                    .enumerate()
                    .skip(rowcol)
                    .step_by(self.ncols)
                {
                    if *elem >= highest {
                        i = idx;
                    }
                }

                Some(self.inverse_at(i))
            }
        }
    }

    /// Finds position in matrix where value is lowest.
    /// Restricted to find this across a row or column
    /// in the matrix.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, Dimension};
    ///
    /// let mut matrix = Matrix::init(10.5, (3,3));
    /// matrix.data[1] = 1.0;
    ///
    /// assert_eq!(matrix.argmin(1, Dimension::Col), Some(1));
    /// ```
    pub fn argmin(&self, rowcol: usize, dimension: Dimension) -> Option<Shape> {
        match dimension {
            Dimension::Row => {
                if rowcol >= self.nrows - 1 {
                    return None;
                }

                let mut lowest: T = T::zero();

                let mut i = 0;

                for (idx, elem) in self
                    .data
                    .iter()
                    .enumerate()
                    .skip(rowcol * self.ncols)
                    .take(self.ncols)
                {
                    if *elem < lowest {
                        i = idx;
                    }
                }

                Some(self.inverse_at(i))
            }

            Dimension::Col => {
                if rowcol >= self.ncols - 1 {
                    return None;
                }

                let mut lowest: T = T::zero();

                let mut i = 0;

                for (idx, elem) in self
                    .data
                    .iter()
                    .enumerate()
                    .skip(rowcol)
                    .step_by(self.ncols)
                {
                    if *elem <= lowest {
                        i = idx;
                    }
                }

                Some(self.inverse_at(i))
            }
        }
    }

    /// Finds total sum of matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.cumsum(), 40.0);
    /// ```
    pub fn cumsum(&self) -> T {
        self.data.par_iter().copied().sum()
    }

    /// Multiplies  all elements in matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.cumprod(), 10000.0);
    /// ```
    pub fn cumprod(&self) -> T {
        self.data.par_iter().copied().product()
    }

    /// Gets the average of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.avg(), 10.0);
    /// ```
    pub fn avg(&self) -> T {
        let mut size: T = T::zero();

        self.data.iter().for_each(|_| size += T::one());

        let tot: T = self.data.par_iter().copied().sum::<T>();

        tot / size
    }

    /// Gets the mean of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.mean(), 10.0);
    /// ```
    pub fn mean(&self) -> T {
        self.avg()
    }

    /// Gets the median of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::new(vec![1.0, 4.0, 6.0, 5.0], (2,2));
    ///
    /// assert!(matrix.median() >= 4.45 && matrix.median() <= 4.55);
    /// ```
    pub fn median(&self) -> T {
        match self.data.len() % 2 {
            0 => {
                let half: usize = self.data.len() / 2;

                self.data
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .skip(half - 1)
                    .take(2)
                    .copied()
                    .sum::<T>()
                    / (T::one() + T::one())
            }
            1 => {
                let half: usize = self.data.len() / 2;

                self.data
                    .iter()
                    .sorted_by(|a, b| a.partial_cmp(&b).unwrap())
                    .nth(half)
                    .copied()
                    .unwrap()
            }
            _ => unreachable!(),
        }
    }

    /// Sums up elements over given dimension and axis
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    /// use sukker::Dimension;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.sum(0, Dimension::Row), 20.0);
    /// assert_eq!(matrix.sum(0, Dimension::Col), 20.0);
    /// ```
    pub fn sum(&self, rowcol: usize, dimension: Dimension) -> T {
        match dimension {
            Dimension::Row => self
                .data
                .par_iter()
                .skip(rowcol * self.ncols)
                .take(self.ncols)
                .copied()
                .sum(),
            Dimension::Col => self
                .data
                .par_iter()
                .skip(rowcol)
                .step_by(self.ncols)
                .copied()
                .sum(),
        }
    }

    /// Prods up elements over given rowcol and dimension
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    /// use sukker::Dimension;
    ///
    /// let matrix = Matrix::init(10f32, (2,2));
    ///
    /// assert_eq!(matrix.prod(0, Dimension::Row), 100.0);
    /// assert_eq!(matrix.prod(0, Dimension::Col), 100.0);
    /// ```
    pub fn prod(&self, rowcol: usize, dimension: Dimension) -> T {
        match dimension {
            Dimension::Row => self
                .data
                .par_iter()
                .skip(rowcol * self.ncols)
                .take(self.ncols)
                .copied()
                .product(),
            Dimension::Col => self
                .data
                .par_iter()
                .skip(rowcol)
                .step_by(self.ncols)
                .copied()
                .product(),
        }
    }
}

/// trait MatrixLinAlg contains all common Linear Algebra functions to be
/// performed on matrices
pub trait MatrixLinAlg<'a, T>
where
    T: MatrixElement,
{
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn sub_abs(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;
    fn add_val(&self, val: T) -> Self;
    fn sub_val(&self, val: T) -> Self;
    fn mul_val(&self, val: T) -> Self;
    fn div_val(&self, val: T) -> Self;
    fn add_self(&mut self, other: &Self);
    fn sub_self(&mut self, other: &Self);
    fn mul_self(&mut self, other: &Self);
    fn div_self(&mut self, other: &Self);
    fn abs_self(&mut self);
    fn add_val_self(&mut self, val: T);
    fn sub_val_self(&mut self, val: T);
    fn mul_val_self(&mut self, val: T);
    fn div_val_self(&mut self, val: T);
    fn matmul(&self, other: &Self) -> Self;
    fn transpose(&mut self);
    fn t(&mut self);
    fn transpose_copy(&self) -> Self;
    fn eigenvalue(&self) -> T;

    fn log(&self, base: T) -> Self;
    fn ln(&self) -> Self;
    fn tanh(&self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn pow(&self, val: usize) -> Self;
    fn abs(&self) -> Self;
}

impl<'a, T> MatrixLinAlg<'a, T> for Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Adds one matrix to another
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix1 = Matrix::init(10.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.add(&matrix2).data[0], 20.0);
    /// ```
    fn add(&self, other: &Self) -> Self {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            panic!("NOOO!");
        }

        let data = (0..self.nrows)
            .flat_map(|i| (0..self.ncols).map(move |j| self.get(i, j) + other.get(i, j)))
            .collect_vec();

        Self::new(data, self.shape)
    }

    /// Subtracts one array from another
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.sub(&matrix2).data[0], 10.0);
    /// ```
    fn sub(&self, other: &Self) -> Self {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            panic!("NOOO!");
        }

        let data = (0..self.nrows)
            .flat_map(|i| (0..self.ncols).map(move |j| self.get(i, j) - other.get(i, j)))
            .collect_vec();

        Self::new(data, self.shape)
    }

    /// Subtracts one array from another and returns the absolute value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix1 = Matrix::init(10.0f32, (2,2));
    /// let matrix2 = Matrix::init(15.0f32, (2,2));
    ///
    /// assert_eq!(matrix1.sub_abs(&matrix2).data[0], 5.0);
    /// ```
    fn sub_abs(&self, other: &Self) -> Self {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            panic!("NOOO!");
        }

        let data = (0..self.nrows)
            .flat_map(|i| {
                (0..self.ncols).map(move |j| {
                    let a = self.get(i, j);
                    let b = other.get(i, j);

                    if a > b {
                        a - b
                    } else {
                        b - a
                    }
                })
            })
            .collect_vec();

        Self::new(data, self.shape)
    }

    /// Dot product of two matrices
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.mul(&matrix2).data[0], 200.0);
    /// ```
    fn mul(&self, other: &Self) -> Self {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            panic!("NOOO!");
        }

        let data = (0..self.nrows)
            .flat_map(|i| (0..self.ncols).map(move |j| self.get(i, j) * other.get(i, j)))
            .collect_vec();

        Self::new(data, self.shape)
    }

    /// Bad handling of zero div
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(10.0, (2,2));
    ///
    /// assert_eq!(matrix1.div(&matrix2).data[0], 2.0);
    /// ```
    fn div(&self, other: &Self) -> Self {
        if self.nrows != other.nrows || self.ncols != other.ncols {
            panic!("NOOO!");
        }

        // TODO: Double check
        if other.any(|e| e == &T::zero()) {
            panic!("NOOOOO")
        }

        let data = (0..self.nrows)
            .flat_map(|i| (0..self.ncols).map(move |j| self.get(i, j) / other.get(i, j)))
            .collect_vec();

        Self::new(data, self.shape)
    }

    /// Adds a value to a matrix and returns a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    /// assert_eq!(matrix.add_val(value).data[0], 22.0);
    /// ```
    fn add_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e + val).collect();

        Self::new(data, self.shape)
    }

    /// Substracts a value to a matrix and returns a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    /// assert_eq!(matrix.sub_val(value).data[0], 18.0);
    /// ```
    fn sub_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e - val).collect();

        Self::new(data, self.shape)
    }

    /// Multiplies a value to a matrix and returns a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    /// assert_eq!(matrix.mul_val(value).data[0], 40.0);
    /// ```
    fn mul_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e * val).collect();

        Self::new(data, self.shape)
    }

    /// Divides a value to a matrix and returns a new matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// let result_mat = matrix.div_val(value);
    ///
    /// assert_eq!(result_mat.data[0], 10.0);
    /// ```
    fn div_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e / val).collect();

        Self::new(data, self.shape)
    }

    /// Takes the logarithm of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(2.0, (2,2));
    ///
    /// ```
    fn log(&self, base: T) -> Self {
        unimplemented!()
        // let data: Vec<T> = self.data.iter().map(|&e| e.log(base)).collect();
        //
        //  Self::new(data, self.shape)
    }

    /// Takes the natural logarithm of each element in a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    /// use sukker::constants::EF64;
    ///
    /// let matrix: Matrix<f64> = Matrix::init(EF64, (2,2));
    ///
    /// // TBI
    /// ```
    fn ln(&self) -> Self {
        unimplemented!()
        // let data: Vec<T> = self.data.iter().map(|&e| e.ln()).collect();
        //
        // Self::new(data, self.shape)
    }

    /// Gets tanh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    /// use sukker::constants::EF32;
    ///
    /// let matrix = Matrix::init(EF32, (2,2));
    ///
    /// ```
    fn tanh(&self) -> Self {
        unimplemented!()
        // let data: Vec<T> = self.data.iter().map(|&e| e.tanh()).collect();
        //
        // Self::new(data, self.shape)
    }

    /// Gets sinh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    /// use sukker::constants::EF32;
    ///
    /// let matrix = Matrix::init(EF32, (2,2));
    ///
    /// ```
    fn sinh(&self) -> Self {
        unimplemented!()
        // let data: Vec<T> = self.data.iter().map(|&e| e.tanh()).collect();
        //
        // Self::new(data, self.shape)
    }

    /// Gets cosh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    /// use sukker::constants::EF32;
    ///
    /// let matrix = Matrix::init(EF32, (2,2));
    ///
    /// ```
    fn cosh(&self) -> Self {
        unimplemented!()
        // let data: Vec<T> = self.data.iter().map(|&e| e.tanh()).collect();
        //
        // Self::new(data, self.shape)
    }

    /// Pows each value in a matrix by val times
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(2.0, (2,2));
    ///
    /// let result_mat = matrix.pow(2);
    ///
    /// assert_eq!(result_mat.data, vec![4.0, 4.0, 4.0, 4.0]);
    /// ```
    fn pow(&self, val: usize) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| pow(e, val)).collect();

        Self::new(data, self.shape)
    }

    /// Takes the absolute values of the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(20.0, (2,2));
    ///
    /// let res = matrix.abs();
    ///
    /// // assert_eq!(matrix1.get(0,0), 22.0);
    /// ```
    fn abs(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| abs(e)).collect();

        Self::new(data, self.shape)
    }

    /// Adds a matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(2.0, (2,2));
    ///
    /// matrix1.add_self(&matrix2);
    ///
    /// assert_eq!(matrix1.get(0,0), 22.0);
    /// ```
    fn add_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a += *b);
    }

    /// Subtracts a matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(2.0, (2,2));
    ///
    /// matrix1.sub_self(&matrix2);
    ///
    /// assert_eq!(matrix1.get(0,0), 18.0);
    /// ```
    fn sub_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a -= *b);
    }

    /// Multiplies a matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(2.0, (2,2));
    ///
    /// matrix1.mul_self(&matrix2);
    ///
    /// assert_eq!(matrix1.get(0,0), 40.0);
    /// ```
    fn mul_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a *= *b);
    }

    /// Divides a matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix1 = Matrix::init(20.0, (2,2));
    /// let matrix2 = Matrix::init(2.0, (2,2));
    ///
    /// matrix1.div_self(&matrix2);
    ///
    /// assert_eq!(matrix1.get(0,0), 10.0);
    /// ```
    fn div_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a /= *b);
    }

    /// Abs matrix in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    ///
    /// matrix.abs_self()
    ///
    /// // assert_eq!(matrix1.get(0,0), 22.0);
    /// ```
    fn abs_self(&mut self) {
        self.data.par_iter_mut().for_each(|e| *e = abs(*e))
    }

    /// Adds a value in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// matrix.add_val_self(value);
    ///
    /// assert_eq!(matrix.get(0,0), 22.0);
    /// ```
    fn add_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e += val);
    }

    /// Subtracts a value in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// matrix.sub_val_self(value);
    ///
    /// assert_eq!(matrix.get(0,0), 18.0);
    /// ```
    fn sub_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e -= val);
    }

    /// Mults a value in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// matrix.mul_val_self(value);
    ///
    /// assert_eq!(matrix.get(0,0), 40.0);
    /// ```
    fn mul_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e *= val);
    }

    /// Divs a value in-place to a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix = Matrix::init(20.0, (2,2));
    /// let value: f32 = 2.0;
    ///
    /// matrix.div_val_self(value);
    ///
    /// assert_eq!(matrix.get(0,0), 10.0);
    /// ```
    fn div_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e /= val);
    }

    /// Transposed matrix multiplications
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix1 = Matrix::init(2.0, (2,4));
    /// let matrix2 = Matrix::init(2.0, (4,2));
    ///
    /// let result = matrix1.matmul(&matrix2);
    ///
    /// assert_eq!(result.get(0,0), 16.0);
    /// assert_eq!(result.shape, (2,2));
    /// ```
    fn matmul(&self, other: &Self) -> Self {
        // assert M N x N P
        if self.ncols != other.nrows {
            panic!("Oops, dimensions do not match");
        }

        let r1 = self.nrows;
        let c1 = self.ncols;
        let c2 = other.ncols;

        let mut data = vec![T::zero(); c2 * r1];

        let t_other = other.transpose_copy();

        for i in 0..r1 {
            for j in 0..c2 {
                data[at!(i, j, c2)] = (0..c1)
                    .into_par_iter()
                    .map(|k| self.data[at!(i, k, c1)] * t_other.data[at!(j, k, t_other.ncols)])
                    .sum();
            }
        }

        Self::new(data, (c2, r1))
    }

    /// Transpose a matrix in-place
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix = Matrix::init(2.0, (2,100));
    /// matrix.transpose();
    ///
    /// assert_eq!(matrix.shape, (100,2));
    /// ```
    fn transpose(&mut self) {
        for i in 0..self.nrows {
            for j in (i + 1)..self.ncols {
                let lhs = at!(i, j, self.ncols);
                let rhs = at!(j, i, self.nrows);
                self.data.swap(lhs, rhs);
            }
        }

        swap(&mut self.shape.0, &mut self.shape.1);
        swap(&mut self.nrows, &mut self.ncols);
    }

    /// Shorthand call for transpose
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix = Matrix::init(2.0, (2,100));
    /// matrix.t();
    ///
    /// assert_eq!(matrix.shape, (100,2));
    /// ```
    fn t(&mut self) {
        self.transpose()
    }

    /// Transpose a matrix and return a copy
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(2.0, (2,100));
    /// let result = matrix.transpose_copy();
    ///
    /// assert_eq!(result.shape, (100,2));
    /// ```
    fn transpose_copy(&self) -> Self {
        let mut res = self.clone();
        res.transpose();
        res
    }

    /// Find the eigenvale of a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let mut matrix = Matrix::init(2.0, (2,100));
    ///
    /// assert_eq!(42f32, 42f32);
    /// ```
    fn eigenvalue(&self) -> T {
        todo!()
    }
}

/// Implementations for predicates
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    /// Counts all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0f32, (2,4));
    ///
    /// assert_eq!(matrix.count_where(|&e| e == 2.0), 8);
    /// ```
    pub fn count_where<F>(&'a self, pred: F) -> usize
    where
        F: Fn(&T) -> bool + Sync,
    {
        self.data.par_iter().filter(|&e| pred(e)).count()
    }

    /// Sums all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.sum_where(|&e| e == 2.0), 16.0);
    /// ```
    pub fn sum_where<F>(&self, pred: F) -> T
    where
        F: Fn(&T) -> bool + Sync,
    {
        self.data
            .par_iter()
            .filter(|&e| pred(e))
            .copied()
            .sum::<T>()
    }

    /// Return whether or not a predicate holds at least once
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.any(|&e| e == 2.0), true);
    /// ```
    pub fn any<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool + Sync + Send,
    {
        self.data.par_iter().any(pred)
    }

    /// Returns whether or not predicate holds for all values
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::randomize_range(1.0, 4.0, (2,4));
    ///
    /// assert_eq!(matrix.all(|&e| e >= 1.0), true);
    /// ```
    pub fn all<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool + Sync + Send,
    {
        self.data.par_iter().all(pred)
    }

    /// Finds first index where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2f32, (2,4));
    ///
    /// assert_eq!(matrix.find(|&e| e >= 1f32), Some((0,0)));
    /// ```
    pub fn find<F>(&self, pred: F) -> Option<Shape>
    where
        F: Fn(&T) -> bool + Sync,
    {
        if let Some((idx, _)) = self.data.iter().find_position(|&e| pred(e)) {
            return Some(self.inverse_at(idx));
        }

        None
    }

    /// Finds all indeces where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.find_all(|&e| e >= 3.0), None);
    /// ```
    pub fn find_all<F>(&self, pred: F) -> Option<Vec<Shape>>
    where
        F: Fn(&T) -> bool + Sync,
    {
        let data: Vec<Shape> = self
            .data
            .par_iter()
            .enumerate()
            .filter_map(|(idx, elem)| {
                if pred(elem) {
                    Some(self.inverse_at(idx))
                } else {
                    None
                }
            })
            .collect();

        if data.is_empty() {
            None
        } else {
            Some(data)
        }
    }
}
