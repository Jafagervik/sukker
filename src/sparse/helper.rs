//! Internal helpers

use std::{error::Error, str::FromStr};

use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};

use crate::{MatrixElement, MatrixError, SparseMatrix};

// Enum for operations
pub enum Operation {
    ADD,
    SUB,
    MUL,
    DIV,
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
                    Operation::SUB => *value += val,
                    Operation::MUL => *value += val,
                    Operation::DIV => *value += val,
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
                Operation::SUB => *value += val,
                Operation::MUL => *value += val,
                Operation::DIV => *value += val,
            }
        }
    }
}
