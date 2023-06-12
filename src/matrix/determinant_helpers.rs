// //! Internal helper methods for determinant calculation
// #![warn(missing_docs)]
// use std::ops::{Add, Mul, Sub};
//
// use crate::{matrix, Matrix};
//
// /// Calculates a 2x2 determinant
// ///
// /// # Examples
// pub fn determinant_2x2<'a, T>(matrix: &Matrix<T>) -> T
// where
//     T: matrix::MatrixElement + Mul<Output = T> + Sub<Output = T>,
// {
//     let a = matrix.data[0];
//     let b = matrix.data[1];
//     let c = matrix.data[2];
//     let d = matrix.data[3];
//
//     a * d - b * c
// }
//
// /// Calculates a 3x3 determinant
// pub fn determinant_3x3<'a, T>(matrix: &Matrix<T>) -> T
// where
//     T: matrix::MatrixElement + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
// {
//     let a = matrix.data[0];
//     let b = matrix.data[1];
//     let c = matrix.data[2];
//     let d = matrix.data[3];
//     let e = matrix.data[4];
//     let f = matrix.data[5];
//     let g = matrix.data[6];
//     let h = matrix.data[7];
//     let i = matrix.data[8];
//
//     a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
// }
//
// /// Gets the next inner layer of a determinant calculation
// pub fn get_minor<'a, T>(matrix: &Matrix<T>, size: usize, col: usize) -> Vec<T>
// where
//     T: matrix::MatrixElement,
// {
//     (1..size)
//         .flat_map(|i| (0..size).map(move |j| (i, j)))
//         .filter_map(|(i, j)| {
//             if j != col {
//                 Some(matrix.data[i * size + j])
//             } else {
//                 None
//             }
//         })
//         .collect()
// }
