use std::{error::Error, mem::size_of, str::FromStr};

use crate::matrix::optim;
use rayon::prelude::*;

use crate::{at, Matrix, MatrixElement};

pub fn swap(lhs: &mut usize, rhs: &mut usize) {
    let temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}

// simd
impl<'a, T> Matrix<'a, T>
where
    T: MatrixElement + 'a,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    pub fn determinant_helper(&self) -> T {
        match self.nrows {
            1 => self.at(0, 0),
            2 => Self::det_2x2(self),
            3 => Self::det_3x3(self),
            n => Self::det_nxn(self.data.clone(), n),
        }
    }

    // General helper function calling out to other matmuls based on target architecture
    pub fn matmul_helper(&self, other: &Self) -> Self {
        match (self.shape(), other.shape()) {
            ((1, 2), (2, 1)) => return self.onetwo_by_twoone(other),
            ((2, 2), (2, 1)) => return self.twotwo_by_twoone(other),
            ((1, 2), (2, 2)) => return self.onetwo_by_twotwo(other),
            ((2, 2), (2, 2)) => return self.twotwo_by_twotwo(other),
            _ => {}
        };

        // Target Detection

        // if let Some(result) = optim::get_optimized_matmul(self, other) {
        //     return result;
        // }

        if self.shape() == other.shape() {
            // Calculated from lowest possible size where
            // nrows & blck_size == 0.
            // Block size will never be more than 50
            let blck_size = (3..=50)
                .collect::<Vec<_>>()
                .into_par_iter()
                .find_last(|b| self.nrows % b == 0)
                .unwrap();

            return self.blocked_matmul(other, blck_size);
        }

        self.naive_matmul(other)
    }

    // ===================================================
    //           Determinant
    // ===================================================

    #[inline(always)]
    fn det_2x2(&self) -> T {
        self.at(0, 0) * self.at(1, 1) - self.at(0, 1) * self.at(1, 0)
    }

    #[inline(always)]
    fn det_3x3(&self) -> T {
        let a = self.at(0, 0);
        let b = self.at(0, 1);
        let c = self.at(0, 2);
        let d = self.at(1, 0);
        let e = self.at(1, 1);
        let f = self.at(1, 2);
        let g = self.at(2, 0);
        let h = self.at(2, 1);
        let i = self.at(2, 2);

        a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    }

    fn det_nxn(matrix: Vec<T>, n: usize) -> T {
        if n == 1 {
            return matrix[0];
        }

        let mut det = T::zero();
        let mut sign = T::one();
        // println!("{:?}", matrix);

        for col in 0..n {
            let sub_det = Self::det_nxn(Self::submatrix(matrix.clone(), n, 0, col), n - 1);

            det += sign * matrix[col] * sub_det;

            sign *= -T::one();
        }

        det
    }

    fn submatrix(matrix: Vec<T>, n: usize, row_to_remove: usize, col_to_remove: usize) -> Vec<T> {
        matrix
            .par_iter()
            .enumerate()
            .filter_map(|(i, &value)| {
                let row = i / n;
                let col = i % n;
                if row != row_to_remove && col != col_to_remove {
                    Some(value)
                } else {
                    None
                }
            })
            .collect()
    }

    // ===================================================
    //           Extremely specific optimizations
    // ===================================================

    // 1x2 @ 2x1 matrix
    #[inline(always)]
    fn onetwo_by_twoone(&self, other: &Self) -> Self {
        let a = self.at(0, 0) * other.at(0, 0) + self.at(0, 1) * other.at(1, 0);

        Self::new(vec![a], (1, 1)).unwrap()
    }

    // 2x2 @ 2x1 matrix
    #[inline(always)]
    fn twotwo_by_twoone(&self, other: &Self) -> Self {
        let a = self.at(0, 0) * other.at(0, 0) + self.at(0, 1) * other.at(1, 0);
        let b = self.at(1, 0) * other.at(0, 0) + self.at(1, 1) * other.at(1, 0);

        Self::new(vec![a, b], (2, 1)).unwrap()
    }

    // 1x2 @ 2x2 matrix
    #[inline(always)]
    fn onetwo_by_twotwo(&self, other: &Self) -> Self {
        let a = self.at(0, 0) * other.at(0, 0) + self.at(0, 1) * other.at(1, 0);
        let b = self.at(0, 0) * other.at(1, 0) + self.at(0, 1) * other.at(1, 1);

        Self::new(vec![a, b], (1, 2)).unwrap()
    }

    // 2x2 @ 2x2 matrix
    #[inline(always)]
    fn twotwo_by_twotwo(&self, other: &Self) -> Self {
        let a = self.at(0, 0) * other.at(0, 0) + self.at(1, 0) * other.at(1, 0);
        let b = self.at(0, 0) * other.at(0, 1) + self.at(0, 1) * other.at(1, 1);
        let c = self.at(1, 0) * other.at(0, 0) + self.at(1, 1) * other.at(1, 0);
        let d = self.at(1, 0) * other.at(1, 0) + self.at(1, 1) * other.at(1, 1);

        Self::new(vec![a, b, c, d], (2, 2)).unwrap()
    }

    // ========================================================================
    //
    //    General solutions for matrix multiplication
    //
    // ========================================================================

    /// Naive matmul if you don't have any SIMD intrinsincts
    fn naive_matmul(&self, other: &Self) -> Self {
        let r1 = self.nrows;
        let c1 = self.ncols;
        let c2 = other.ncols;

        let mut data = vec![T::zero(); c2 * r1];

        let t_other = other.transpose_copy();

        for i in 0..r1 {
            for j in 0..c2 {
                data[at!(i, j, c2)] = (0..c1)
                    .into_par_iter()
                    .map(|k| self.at(i, k) * t_other.at(j, k))
                    .sum();
            }
        }
        Self::new(data, (c2, r1)).unwrap()
    }

    // Blocked matmul if you don't have any SIMD intrinsincts
    // https://csapp.cs.cmu.edu/public/waside/waside-blocking.pdf
    //
    // Modification involves transposing the B matrix, at the cost
    // of increased space complexity, but better cache hit rate
    fn blocked_matmul(&self, other: &Self, block_size: usize) -> Self {
        let N = self.nrows;

        let en = block_size * (N / block_size);

        let mut data = vec![T::zero(); N * N];

        let t_other = other.transpose_copy();

        for kk in (0..N).step_by(en) {
            for jj in (0..N).step_by(en) {
                for i in 0..N {
                    for j in jj..jj + block_size {
                        data[at!(i, j, N)] = (kk..kk + block_size)
                            .into_par_iter()
                            .map(|k| self.at(i, k) * t_other.at(j, k))
                            .sum();

                        // println!("i {} j {} val {}", i, j, data[at!(i, j, N)]);
                    }
                }
            }
        }
        Self::new(data, (N, N)).unwrap()
    }
}
