//! Core module for defining everything matrices
#![warn(missing_docs)]

// mod determinant_helpers;

// use determinant_helpers::*;
use itertools::{iproduct, Itertools};
use rand::Rng;
use rayon::prelude::*;
use std::fmt::Display;
//use std::fmt::Display;
use std::fs;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::process::Output;
use std::{
    fmt::{Debug, Error},
    str::FromStr,
};

/// Type definitions
pub type Shape = (usize, usize);

/// calculates index for matrix
macro_rules! at {
    ($i:ident, $j:ident, $cols:expr) => {
        $i * $cols + $j
    };
}

/// Represents all data we want to get
#[derive(Clone, PartialEq, PartialOrd)]
pub struct Matrix<'a, T>
where
    T: MatrixElement,
{
    /// Data contained inside the matrix with f32 values
    pub data: Vec<T>,
    /// Shape of matrix, (rows, cols)
    pub shape: Shape,
    /// Number of rows
    pub nrows: usize,
    /// Number of cols
    pub ncols: usize,
    _lifetime: PhantomData<&'a T>,
}

/// These are the traits needed to be valid datatype in matrix
pub trait MatrixElement:
    Copy
    + Clone
    + PartialOrd
    + Add
    + Sub
    + Mul
    + Div
    + Display
    + Debug
    + FromStr
    + rand::distributions::uniform::SampleUniform
{
}

impl MatrixElement for f32 {}
impl MatrixElement for f64 {}
impl MatrixElement for i32 {}
impl MatrixElement for i64 {}

impl<'a, T> Debug for Matrix<'a, T>
where
    T: MatrixElement,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[");

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                if i == 0 {
                    write!(f, "{} ", self.data[at!(i, j, self.cols())]);
                } else {
                    write!(f, " {}", self.data[at!(i, j, self.cols())]);
                }
            }
            // Print ] on same line if youre at the end
            if i == self.shape.0 - 1 {
                break;
            }
            write!(f, "\n");
        }
        write!(f, "], dtype={}", std::any::type_name::<T>())
    }
}

impl<'a, T> FromStr for Matrix<'a, T>
where
    T: MatrixElement,
{
    type Err = Error;
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

/// Printer functions for the matrix
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
{
    /// Prints out the matrix based on shape. Also outputs datatype
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::eye(2);
    /// matrix.print();
    ///
    /// ```
    pub fn print(&self) {
        print!("[");

        // Large matrices
        if self.nrows > 10 || self.ncols > 10 {
            print!("...");
        }

        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if i == 0 {
                    print!("{:.2} ", self.data[at!(i, j, self.ncols)]);
                } else {
                    print!(" {:.2}", self.data[at!(i, j, self.ncols)]);
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
{
    /// Creates a new matrix from a vector and the shape you want.
    /// Will default init if it does not work
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::new(vec![1.0,2.0,3.0,4.0], (2,2));
    ///
    /// assert_eq!(matrix.size(), 4);
    /// assert_eq!(matrix.shape, (2,2));
    /// ```
    pub fn new(data: Vec<T>, shape: Shape) -> Self {
        if shape.0 * shape.1 != data.len() {
            return Self::default();
        }

        Self {
            data,
            shape,
            nrows: shape.0,
            ncols: shape.1,
            _lifetime: PhantomData::default(),
        }
    }

    /// Represents a default identity matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::default();
    ///
    /// assert_eq!(matrix.size(), 9);
    /// assert_eq!(matrix.shape, (3,3));
    /// assert_eq!(matrix.get(0,0), 0f32);
    /// ```
    pub fn default() -> Self {
        Self {
            data: vec![0; 9],
            shape: (3, 3),
            nrows: 3,
            ncols: 3,
            _lifetime: PhantomData::default(),
        }
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
    /// let matrix = Matrix::eye(2);
    ///
    /// assert_eq!(matrix.data, vec![1f32, 0f32, 0f32, 1f32]);
    /// assert_eq!(matrix.shape, (2,2));
    /// ```
    pub fn eye(size: usize) -> Self {
        let mut data: Vec<T> = vec![0; size * size];

        (0..size).for_each(|i| data[i * size + i] = 1 as T);

        Self::new(data, (size, size))
    }

    /// Identity is same as eye, just for nerds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::identity(2);
    ///
    /// assert_eq!(matrix.data, vec![1f32, 0f32, 0f32, 1f32]);
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
    /// let matrix = Matrix::from_slice(&s, (4,1));
    ///
    /// assert_eq!(matrix.unwrap().shape, (4,1));
    /// ```
    pub fn from_slice(arr: &[T], shape: Shape) -> Option<Self> {
        if shape.0 * shape.1 != arr.len() {
            return None;
        }

        Some(Self::new(arr.to_owned(), shape))
    }

    /// Creates a matrix where all values are 0.
    /// All sizes are based on a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::zeros((4,1));
    ///
    /// assert_eq!(matrix.shape, (4,1));
    /// assert_eq!(matrix.data, vec![0f32; 4]);
    /// ```
    pub fn zeros(shape: Shape) -> Self {
        Self::from_shape(0 as T, shape)
    }

    /// Creates a matrix where all values are 1.
    /// All sizes are based on a shape
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::ones((4,1));
    ///
    /// assert_eq!(matrix.shape, (4,1));
    /// assert_eq!(matrix.data, vec![1f32; 4]);
    /// ```
    pub fn ones(shape: Shape) -> Self {
        Self::from_shape(1 as T, shape)
    }

    /// Creates a matrix where all values are 0.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::default();
    /// let matrix2 = Matrix::zeros_like(&matrix1);
    ///
    /// assert_eq!(matrix2.shape, matrix1.shape);
    /// ```
    pub fn zeros_like(other: &Self) -> Self {
        Self::from_shape(0 as T, other.shape)
    }

    /// Creates a matrix where all values are 1.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::default();
    /// let matrix2 = Matrix::ones_like(&matrix1);
    ///
    /// assert_eq!(matrix2.shape, matrix1.shape);
    /// assert_eq!(1f32, matrix2.data[0]);
    /// ```
    pub fn ones_like(other: &Self) -> Self {
        Self::from_shape(1 as T, other.shape)
    }

    /// Creates a matrix where all values are random between 0 and 1.
    /// All sizes are based on an already existent matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix1 = Matrix::default();
    /// let matrix2 = Matrix::random_like(&matrix1);
    ///
    /// assert_eq!(matrix1.shape, matrix2.shape);
    /// ```
    pub fn random_like(matrix: &Self) -> Self {
        Self::randomize_range(0 as T, 1 as T, matrix.shape)
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
    ///
    /// assert_eq!(matrix.shape, (2,3));
    /// ```
    pub fn randomize_range(start: T, end: T, shape: Shape) -> Self {
        let mut rng = rand::thread_rng();

        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data: Vec<T> = (0..len).map(|_| rng.gen_range(start..=end)).collect();

        Self::new(data, shape)
    }

    /// Creates a matrix where all values are random between 0..=1.
    /// Shape in new array is given through parameter 'shape'
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::randomize((2,3));
    ///
    /// assert_eq!(matrix.shape, (2,3));
    /// ```
    pub fn randomize(shape: Shape) -> Self {
        Self::randomize_range(0 as T, 1 as T, shape)
    }

    /// Parses from file, but will return a default matrix if nothing is given
    ///
    /// # Examples
    pub fn from_file(path: &'static str) -> Self {
        let data =
            fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read file: {}", path));

        data.parse::<Self>().unwrap_or_else(|_| Self::default())
    }

    /// HELPER, name is too retarded for public usecases
    fn from_shape(value: T, shape: Shape) -> Self {
        let (rows, cols) = shape;

        let len: usize = rows * cols;

        let data = vec![value.clone(); len];

        Self::new(data, shape)
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
    T: MatrixElement + std::iter::Sum<&'a T>,
{
    /// Converts matrix to a tensor
    pub fn to_tensor(&self) {
        todo!()
    }

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
        }

        self.shape = new_shape;
    }

    /// Get number of columns in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,3));
    ///
    /// assert_eq!(matrix.cols(), 3);
    /// ```
    pub fn cols(&self) -> usize {
        self.shape.1
    }

    /// Get number of rows in the matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (2,3));
    ///
    /// assert_eq!(matrix.rows(), 2);
    /// ```
    pub fn rows(&self) -> usize {
        self.shape.0
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
        self.shape.0 * self.shape.1
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
    /// assert_eq!(matrix.get(1,2), 10.5);
    /// ```
    pub fn get(&self, i: usize, j: usize) -> T {
        self.data[at!(i, j, self.cols())]
    }

    ///  Gets a vec slice
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::Matrix;
    ///
    /// let matrix = Matrix::init(10.5, (4,4));
    /// let slice = matrix.get_vec_slice(1,1, 2,2);
    ///
    /// assert_eq!(slice, vec![10.5,10.5,10.5,10.5]);
    /// ```
    pub fn get_vec_slice(
        &self,
        start_row: usize,
        start_col: usize,
        dy: usize,
        dx: usize,
    ) -> Vec<T> {
        iproduct!(start_row..start_row + dy, start_col..start_col + dx)
            .map(|(i, j)| self.get(i, j))
            .collect()
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
        self.data[i * j + self.ncols] = value;
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
        let row = idx / self.cols();
        let col = idx % self.cols();

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
            .iter()
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
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }

    /// Finds argmax based on row or col
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, Dimension};
    ///
    /// let mut matrix = Matrix::init(1.0, (3,3));
    /// matrix.data[2] = 15.0;
    ///
    /// assert_eq!(matrix.argmax(0, Dimension::Row), Some(15.0));
    /// assert_eq!(matrix.argmax(1, Dimension::Row), Some(1.0));
    /// ```
    pub fn argmax(&self, rowcol: usize, dimension: Dimension) -> Option<T> {
        match dimension {
            Dimension::Row => {
                if rowcol >= self.rows() - 1 {
                    return None;
                }

                self.data
                    .par_iter()
                    .skip(rowcol * self.cols())
                    .take(self.cols())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            }

            Dimension::Col => {
                if rowcol >= self.cols() - 1 {
                    return None;
                }

                self.data
                    .par_iter()
                    .skip(rowcol)
                    .step_by(self.cols())
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            }
        }
    }

    /// Finds argmax based on row or col
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, Dimension};
    ///
    /// let mut matrix = Matrix::init(10.5, (3,3));
    /// matrix.data[1] = 1.0;
    ///
    /// assert_eq!(matrix.argmin(1, Dimension::Col), Some(1.0));
    /// assert_eq!(matrix.argmin(1, Dimension::Row), Some(10.5));
    /// ```
    pub fn argmin(&self, rowcol: usize, dimension: Dimension) -> Option<T> {
        match dimension {
            Dimension::Row => {
                if rowcol >= self.rows() - 1 {
                    return None;
                }

                self.data
                    .par_iter()
                    .skip(rowcol * self.cols())
                    .take(self.cols())
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
            }

            Dimension::Col => {
                if rowcol >= self.cols() - 1 {
                    return None;
                }

                self.data
                    .par_iter()
                    .skip(rowcol)
                    .step_by(self.cols())
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
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
        self.data.par_iter().sum()
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
        self.data.par_iter().product()
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
        self.data.par_iter().sum::<T>() / self.data.len() as T
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
                    .sum::<T>()
                    / 2 as T
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
                .skip(rowcol * self.cols())
                .take(self.cols())
                .sum(),
            Dimension::Col => self.data.par_iter().skip(rowcol).step_by(self.cols()).sum(),
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
                .skip(rowcol * self.cols())
                .take(self.cols())
                .product(),
            Dimension::Col => self
                .data
                .par_iter()
                .skip(rowcol)
                .step_by(self.cols())
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
    fn add(&self, other: &Self) -> Self;

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
    fn sub(&self, other: &Self) -> Self;

    /// Subtracts one array from another and returns absval
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix1 = Matrix::init(10.0, (2,2));
    /// let matrix2 = Matrix::init(15.0, (2,2));
    ///
    /// assert_eq!(matrix1.sub_abs(&matrix2).data[0], 5.0);
    /// ```
    fn sub_abs(&self, other: &Self) -> Self;

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
    fn mul(&self, other: &Self) -> Self;

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
    fn div(&self, other: &Self) -> Self;

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
    fn add_val(&self, val: T) -> Self;

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
    fn sub_val(&self, val: T) -> Self;

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
    fn mul_val(&self, val: T) -> Self;

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
    fn div_val(&self, val: T) -> Self;

    /// Takes the logarithm of each element
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(2.0, (2,2));
    ///
    /// let result_mat = matrix.log(2.0);
    ///
    /// assert_eq!(result_mat.data, vec![1.0, 1.0, 1.0, 1.0]);
    /// ```
    fn log(&self, base: T) -> Self;

    /// Takes the natural logarithm of each element in a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    /// use sukker::constants::E;
    ///
    /// let matrix = Matrix::init(E, (2,2));
    ///
    /// let result_mat = matrix.ln();
    ///
    /// assert_eq!(result_mat.shape, (2,2));
    /// ```
    fn ln(&self) -> Self;

    /// Gets tanh of every value
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    /// use sukker::constants::E;
    ///
    /// let matrix = Matrix::init(E, (2,2));
    ///
    /// let result_mat = matrix.tanh();
    ///
    /// assert_eq!(result_mat.shape, (2,2));
    /// ```
    fn tanh(&self) -> Self;

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
    fn pow(&self, val: i32) -> Self;

    /// Pows each value in a matrix by val times
    /// val can be a float
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(2.0, (2,2));
    ///
    /// let result_mat = matrix.powf(2.0);
    ///
    /// assert_eq!(result_mat.data, vec![4.0, 4.0, 4.0, 4.0]);
    /// ```
    fn powf(&self, val: f32) -> Self;

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
    fn abs(&self) -> Self;

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
    fn add_self(&mut self, other: &Self);

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
    fn sub_self(&mut self, other: &Self);

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
    fn mul_self(&mut self, other: &Self);

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
    fn div_self(&mut self, other: &Self);

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
    fn abs_self(&mut self);

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
    fn add_val_self(&mut self, val: T);

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
    fn sub_val_self(&mut self, val: T);

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
    fn mul_val_self(&mut self, val: T);

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
    fn div_val_self(&mut self, val: T);

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
    fn matmul(&self, other: &Self) -> Self;

    /// Find the determinant of a matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixLinAlg};
    ///
    /// let matrix = Matrix::init(2.0, (2,2));
    /// let det = matrix.determinant();
    ///
    /// assert_eq!(0.0, 0.0);
    /// ```
    fn determinant(&self) -> T;

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
    fn transpose(&mut self);

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
    fn t(&mut self);

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
    fn transpose_copy(&self) -> Self;

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
    fn eigenvalue(&self) -> T;
}

impl<'a, T> MatrixLinAlg<'a, T> for Matrix<'a, T>
where
    T: MatrixElement + std::ops::Div<Output = T> + std::ops::Mul<Output = T>,
    &'a T: PartialEq<T>,
{
    fn add(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = a + b;
            }
        }
        mat
    }

    fn sub(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = a - b;
            }
        }
        mat
    }

    fn sub_abs(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = (a - b).abs();
            }
        }
        mat
    }

    fn mul(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = a * b;
            }
        }
        mat
    }

    fn div(&self, other: &Self) -> Self {
        if self.rows() != other.rows() || self.cols() != other.cols() {
            panic!("NOOO!");
        }

        // TODO: Double check
        if other.any(|e| e == &0 as T) {
            panic!("NOOOOO")
        }

        let mut mat = Self::zeros_like(self);

        for i in 0..self.rows() {
            for j in 0..self.cols() {
                let a = self.get(i, j);
                let b = other.get(i, j);
                mat.data[at!(i, j, self.cols())] = a / b;
            }
        }
        mat
    }

    fn add_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e + val).collect();

        Self::new(data, self.shape)
    }

    fn sub_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e - val).collect();

        Self::new(data, self.shape)
    }

    fn mul_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e * val).collect();

        Self::new(data, self.shape)
    }

    fn div_val(&self, val: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e / val).collect();

        Self::new(data, self.shape)
    }

    fn log(&self, base: T) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.log(base)).collect();

        Self::new(data, self.shape)
    }

    fn ln(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.ln()).collect();

        Self::new(data, self.shape)
    }

    fn tanh(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.tanh()).collect();

        Self::new(data, self.shape)
    }

    fn pow(&self, val: i32) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.powi(val)).collect();

        Self::new(data, self.shape)
    }

    fn powf(&self, val: f32) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.powf(val)).collect();

        Self::new(data, self.shape)
    }

    fn abs(&self) -> Self {
        let data: Vec<T> = self.data.par_iter().map(|&e| e.abs()).collect();

        Self::new(data, self.shape)
    }

    fn add_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a += b);
    }

    fn sub_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a -= b);
    }

    fn mul_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a *= b);
    }

    fn div_self(&mut self, other: &Self) {
        self.data
            .par_iter_mut()
            .zip(&other.data)
            .for_each(|(a, b)| *a /= b);
    }

    fn abs_self(&mut self) {
        self.data.par_iter_mut().for_each(|e| *e = e.abs());
    }

    fn add_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e += val);
    }

    fn sub_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e -= val);
    }

    fn mul_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e *= val);
    }

    fn div_val_self(&mut self, val: T) {
        self.data.par_iter_mut().for_each(|e| *e /= val);
    }

    fn matmul(&self, other: &Self) -> Self {
        // assert M N x N P
        if self.cols() != other.rows() {
            panic!("Oops, dimensions do not match");
        }

        let r1 = self.rows();
        let c1 = self.cols();
        let c2 = other.cols();

        let mut data = vec![0 as T; c2 * r1];

        let t_other = other.transpose_copy();

        for i in 0..r1 {
            for j in 0..c2 {
                let dot_product = (0..c1)
                    .into_par_iter()
                    .map(|k| self.data[at!(i, k, c1)] * t_other.data[at!(j, k, t_other.cols())])
                    .sum();

                data[at!(i, j, c2)] = dot_product;
            }
        }

        Self::new(data, (c2, r1))
    }

    fn determinant(&self) -> T {
        match self.size() {
            1 => self.data[0],
            4 => todo!(),
            9 => todo!(),
            _ => {
                todo!()
                // let mut det = 0.0;
                // let mut sign = 1.0;
                //
                // for i in 0..3 {
                //     let minor = get_minor(self, 3, i);
                //     det += sign * self.data[i] * determinant(&minor);
                //     sign *= -1.0;
                // }
                //
                // det
            }
        }
    }

    fn transpose(&mut self) {
        for i in 0..self.rows() {
            for j in (i + 1)..self.cols() {
                let lhs = at!(i, j, self.cols());
                let rhs = at!(j, i, self.rows());
                self.data.swap(lhs, rhs);
            }
        }

        swap(&mut self.shape.0, &mut self.shape.1);
    }

    fn t(&mut self) {
        self.transpose()
    }

    fn transpose_copy(&self) -> Self {
        let mut res = self.clone();
        res.transpose();
        res
    }

    fn eigenvalue(&self) -> T {
        todo!()
    }
}

/// Matrix functions with predicates
pub trait MatrixPredicates<'a, T>
where
    T: MatrixElement,
{
    /// Counts all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixPredicates};
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.count_where(|&e| e == 2.0), 8);
    /// ```
    fn count_where<F>(&self, pred: F) -> usize
    where
        F: Fn(&T) -> bool;

    /// Sums all occurances where predicate holds
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixPredicates};
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.sum_where(|&e| e == 2.0), 16.0);
    /// ```
    fn sum_where<F>(&self, pred: F) -> T
    where
        F: Fn(&T) -> bool;

    /// Return whether or not a predicate holds at least once
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixPredicates};
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.any(|&e| e == 2.0), true);
    /// ```
    fn any<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool;

    /// Returns whether or not predicate holds for all values
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixPredicates};
    ///
    /// let matrix = Matrix::randomize_range(1.0, 4.0, (2,4));
    ///
    /// assert_eq!(matrix.all(|&e| e >= 1.0), true);
    /// ```
    fn all<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool;

    /// Finds first index where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixPredicates};
    ///
    /// let matrix = Matrix::init(2f32, (2,4));
    ///
    /// assert_eq!(matrix.find(|&e| e >= 1f32), Some((0,0)));
    /// ```
    fn find<F>(&self, pred: F) -> Option<Shape>
    where
        F: Fn(&T) -> bool;

    /// Finds all indeces where predicates holds if possible
    ///
    /// # Examples
    ///
    /// ```
    /// use sukker::{Matrix, MatrixPredicates};
    ///
    /// let matrix = Matrix::init(2.0, (2,4));
    ///
    /// assert_eq!(matrix.find_all(|&e| e >= 3.0), None);
    /// ```
    fn find_all<F>(&self, pred: F) -> Option<Vec<Shape>>
    where
        F: Fn(&T) -> bool;
}

impl<'a, T> MatrixPredicates<'a, T> for Matrix<'a, T>
where
    T: MatrixElement,
{
    fn count_where<F>(&self, pred: F) -> usize
    where
        F: Fn(&T) -> bool,
    {
        self.data.par_iter().filter(|&e| pred(e)).count()
    }

    fn sum_where<F>(&self, pred: F) -> T
    where
        F: Fn(&T) -> bool,
    {
        self.data.par_iter().filter(|&e| pred(e)).sum()
    }

    fn any<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        self.data.par_iter().any(pred)
    }

    fn all<F>(&self, pred: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        self.data.par_iter().all(pred)
    }

    fn find<F>(&self, pred: F) -> Option<Shape>
    where
        F: Fn(&T) -> bool,
    {
        if let Some((idx, _)) = self.data.iter().find_position(|&e| pred(e)) {
            return Some(self.inverse_at(idx));
        }

        None
    }

    fn find_all<F>(&self, pred: F) -> Option<Vec<Shape>>
    where
        F: Fn(&T) -> bool,
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

/// Helper method to swap to usizes
fn swap(lhs: &mut usize, rhs: &mut usize) {
    let temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}
