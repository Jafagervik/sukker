//! Internal helpers

use std::{collections::HashMap, error::Error, str::FromStr};

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{MatrixElement, MatrixError, SparseMatrix, SparseMatrixData};

// Enum for operations
pub enum Operation {
    ADD,
    SUB,
    MUL,
    DIV,
}

pub fn swap(lhs: &mut usize, rhs: &mut usize) {
    let temp = *lhs;
    *lhs = *rhs;
    *rhs = temp;
}

impl<'a, T> SparseMatrix<'a, T>
where
    T: MatrixElement,
    <T as FromStr>::Err: Error + 'static,
    Vec<T>: IntoParallelIterator,
    Vec<&'a T>: IntoParallelRefIterator<'a>,
{
    // Helper for add, sub, mul and div on SparseMatrix - SparseMatrix operation
    #[doc(hidden)]
    pub fn sparse_helper(&self, other: &Self, op: Operation) -> Result<Self, MatrixError> {
        if self.shape() != other.shape() {
            return Err(MatrixError::MatrixDimensionMismatchError.into());
        }

        let mut result_mat = Self::new(self.nrows, self.ncols);

        for (&idx, &val) in self.data.iter() {
            result_mat.set(idx, val);
        }

        for (&idx, &val) in other.data.iter() {
            match result_mat.data.get_mut(&idx) {
                Some(value) => match op {
                    Operation::ADD => *value += val,
                    Operation::SUB => *value += val,
                    Operation::MUL => *value += val,
                    Operation::DIV => *value += val,
                },
                None => result_mat.set(idx, val),
            };
        }

        Ok(result_mat)
    }

    #[doc(hidden)]
    pub fn sparse_helper_self(&mut self, other: &Self, op: Operation) {
        // TODO: Mismatch in dimensions might not be an issue? Find out
        if self.shape() != other.shape() {
            eprintln!("Oops, mismatch in dims");
            return;
        }

        for (&idx, &val) in other.data.iter() {
            match self.data.get_mut(&idx) {
                Some(value) => match op {
                    Operation::ADD => *value += val,
                    Operation::SUB => *value -= val,
                    Operation::MUL => *value *= val,
                    Operation::DIV => *value /= val,
                },
                None => self.set(idx, val),
            };
        }
    }

    #[doc(hidden)]
    pub fn sparse_helper_val(&self, value: T, op: Operation) -> Self {
        let mut result_mat = Self::new(self.nrows, self.ncols);

        for (&idx, &old_value) in self.data.iter() {
            let new_value = match op {
                Operation::ADD => old_value + value,
                Operation::SUB => old_value - value,
                Operation::MUL => old_value * value,
                Operation::DIV => old_value / value,
            };

            result_mat.set(idx, new_value);
        }

        result_mat
    }

    #[doc(hidden)]
    pub fn sparse_helper_self_val(&mut self, val: T, op: Operation) {
        for (_, value) in self.data.iter_mut() {
            match op {
                Operation::ADD => *value += val,
                Operation::SUB => *value -= val,
                Operation::MUL => *value *= val,
                Operation::DIV => *value /= val,
            }
        }
    }

    // =============================================================
    //     Sparse Matrix Mulitplication helpers
    // =============================================================

    // For nn x nn
    #[doc(hidden)]
    pub fn matmul_sparse_nn(&self, other: &Self) -> Self {
        self.matmul_sparse_mnnp(other)
    }

    // mn x np
    #[doc(hidden)]
    pub fn matmul_sparse_mnnp(&self, other: &Self) -> Self {
        let x = self.nrows;
        let y = self.ncols;
        let z = other.ncols;

        let mut data: SparseMatrixData<T> = HashMap::new();

        for i in 0..x {
            for j in 0..y {
                // We notice that most times, we wont calculate the new sparse matrix
                // so therefore we do an early continue if this is the case
                if self.at(i, j) == T::zero() {
                    continue;
                }

                let result = (0..z)
                    .into_par_iter()
                    .map(|k| self.at(i, j) * other.at(j, k))
                    .sum();

                data.insert((i, j), result);
            }
        }

        Self::init(data, (x, z))
    }
}
