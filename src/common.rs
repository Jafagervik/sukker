//! All the common traits, types and information
//! for dense and sparse matrices are to be found here

use std::{
    error::Error,
    fmt::{Debug, Display},
    str::FromStr,
};

use num_traits::{
    sign::Signed, Float, Num, NumAssign, NumAssignOps, NumAssignRef, NumOps, One, Zero,
};
use rand::distributions::uniform::SampleUniform;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use std::iter::{Product, Sum};

/// Trait MatrixElement represent all traits
/// a datatype has to have to be used in a matrix
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

impl MatrixElement for i8 {}
impl MatrixElement for i16 {}
impl MatrixElement for i32 {}
impl MatrixElement for i64 {}
impl MatrixElement for i128 {}
impl MatrixElement for f32 {}
impl MatrixElement for f64 {}

// TODO: Add For ints
pub trait MatrixMathCommon {}

/// Some operations can only be done on floats,
/// and these can be implemented both for Matrix,
/// and sparse matrix
pub trait MatrixMathFloats<'a, T>
where
    T: MatrixElement + Float + 'a,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    fn log(&self, base: T) -> Self;
    fn ln(&self) -> Self;
    fn sin(&self) -> Self;
    fn tan(&self) -> Self;
    fn cos(&self) -> Self;
    fn sinh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn eigenvalue(&self) -> T;
}
