use std::{error::Error, str::FromStr};

use std::arch::x86_64;

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{at, Matrix, MatrixElement};

pub fn swap(lhs: &mut usize, rhs: &mut usize) {
    let temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}

// simd
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    // General helper function calling out to other matmuls based on target architecture
    pub fn matmul_helper(&self, other: &Self) -> Self {
        if is_x86_feature_detected!("avx") {
            self.naive_matmul(other)
        } else if is_x86_feature_detected!("sse") {
            self.naive_matmul(other)
        } else {
            self.naive_matmul(other)
        }
    }

    /// Naive matmul if you don't have any SIMD intrinsincts
    pub fn naive_matmul(&self, other: &Self) -> Self {
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
        Self::new(data, (c2, r1)).unwrap()
    }
}
