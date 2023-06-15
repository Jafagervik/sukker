//! All the common traits, types and information
//! for dense and sparse matrices are to be found here

use std::{
    fmt::{Debug, Display},
    str::FromStr,
};

use num_traits::{sign::Signed, Num, NumAssign, NumAssignOps, NumAssignRef, NumOps, One, Zero};
use rand::distributions::uniform::SampleUniform;
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
