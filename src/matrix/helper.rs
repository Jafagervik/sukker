use std::{error::Error, mem::size_of, str::FromStr};

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

        #[cfg(any(target_arch = "x86", target_arch = "x86-64"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.avx_matmul(other) }
            } else if is_x86_feature_detected!("sse") {
                self.blocked_matmul(other, 4)
            }
        }

        if self.shape() == other.shape() {
            // Calculated from lowest possible size where
            // nrows & blck_size == 0.
            // Block size will never be more than 50
            let blck_size = (3..=50)
                .collect::<Vec<_>>()
                .into_par_iter()
                .find_first(|b| self.nrows % b == 0)
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
        println!("12 21");
        let a = self.at(0, 0) * other.at(0, 0) + self.at(0, 1) * other.at(1, 0);

        Self::new(vec![a], (1, 1)).unwrap()
    }

    // 2x2 @ 2x1 matrix
    #[inline(always)]
    fn twotwo_by_twoone(&self, other: &Self) -> Self {
        println!("22 21");
        let a = self.at(0, 0) * other.at(0, 0) + self.at(0, 1) * other.at(1, 0);
        let b = self.at(1, 0) * other.at(0, 0) + self.at(1, 1) * other.at(1, 0);

        Self::new(vec![a, b], (2, 1)).unwrap()
    }

    // 1x2 @ 2x2 matrix
    #[inline(always)]
    fn onetwo_by_twotwo(&self, other: &Self) -> Self {
        println!("12 22");
        let a = self.at(0, 0) * other.at(0, 0) + self.at(0, 1) * other.at(1, 0);
        let b = self.at(0, 0) * other.at(1, 0) + self.at(0, 1) * other.at(1, 1);

        Self::new(vec![a, b], (1, 2)).unwrap()
    }

    // 2x2 @ 2x2 matrix
    #[inline(always)]
    fn twotwo_by_twotwo(&self, other: &Self) -> Self {
        println!("22 22");

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

    /// AVX matmul for the IEEE754 Double Precision Floating Point Datatype
    /// https://www.akkadia.org/drepper/cpumemory.pdf
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn avx_matmul(&self, other: &Self) -> Self {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::_mm256_add_epi64;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{
            __m128d, _mm256_add_epi64, _mm_add_pd, _mm_load_pd, _mm_mul_pd, _mm_prefetch,
            _mm_store_pd, _mm_unpacklo_pd, _MM_HINT_NTA,
        };

        // let N = self.nrows;
        //
        // let res = [0.0; 4];
        // let mul1 = [0.0; 4];
        // let mul2 = [0.0; 4];
        //
        // let CLS = 420;
        //
        // let SM = CLS / size_of::<f64>();
        //
        // let mut rres: *const f64;
        // let mut rmul1: *const f64;
        // let mut rmul2: *const f64;
        //
        // for i in (0..N).step_by(SM) {
        //     for j in (0..N).step_by(SM) {
        //         for k in (0..N).step_by(SM) {
        //             for i2 in 0..SM {
        //                 _mm_prefetch(&rmul1[8]);
        //
        //                 for k2 in 0..SM {
        //                     rmul2 = 1.0;
        //
        //                     // load m1d
        //                     let m1d: __m128d = _mm_load_pd(rmul1);
        //
        //                     for j2 in (0..SM).step_by(2) {
        //                         let m2: __m128d = _mm_load_pd(rmul2);
        //                         let r2: __m128d = _mm_load_pd(rres);
        //                         _mm_store_pd(rres, _mm_add_pd(_mm_mul_pd(m2, m1d), r2));
        //
        //                         // Inner most computations
        //                     }
        //
        //                     rmul2 += N as f64;
        //                 }
        //             }
        //         }
        //     }
        // }

        Self::default()
    }

    /// Blocked matmul if you don't have any SIMD intrinsincts
    fn blocked_matmul(&self, other: &Self, block_size: usize) -> Self {
        let N = self.nrows;

        let mut data = vec![T::zero(); N * N];

        let t_other = other.transpose_copy();

        for ii in 0..N {
            for jj in 0..N {
                for kk in 0..N {
                    for i in (ii - 1) * block_size..ii * block_size {
                        for j in (jj - 1) * block_size..jj * block_size {
                            data[at!(i, j, N)] = ((kk - 1) * block_size..kk * block_size)
                                .into_par_iter()
                                .map(|k| self.at(i, k) * t_other.at(j, k))
                                .sum();
                        }
                    }
                }
            }
        }
        Self::new(data, (N, N)).unwrap()
    }
}
