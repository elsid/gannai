#[derive(Clone, Debug)]
pub struct Matrix<'r, T: 'r> {
    column_len: usize,
    values: &'r [T],
}

impl <'r, T: Clone> Matrix<'r, T> {
    pub fn new(column_len: usize, values: &'r [T]) -> Matrix<'r, T> {
        Matrix {
            column_len: column_len,
            values: values,
        }
    }

    pub fn column_len(&self) -> usize {
        self.column_len
    }

    pub fn values(&self) -> &[T] {
        self.values
    }

    pub fn row(&self, row: usize) -> &[T] {
        &self.values[row * self.column_len..(row + 1) * self.column_len]
    }
}

impl <'a, T: PartialEq> PartialEq for Matrix<'a, T> {
    fn eq(&self, other: &Matrix<'a, T>) -> bool {
        self.column_len == other.column_len && self.values == other.values
    }

    fn ne(&self, other: &Matrix<'a, T>) -> bool {
        self.column_len != other.column_len || self.values != other.values
    }
}

#[derive(Debug)]
pub struct MatrixMut<'r, T: 'r> {
    column_len: usize,
    values: &'r mut [T],
}

impl <'r, T: Clone> MatrixMut<'r, T> {
    pub fn new(column_len: usize, values: &'r mut [T]) -> MatrixMut<'r, T> {
        MatrixMut {
            column_len: column_len,
            values: values,
        }
    }

    pub fn column_len(&self) -> usize {
        self.column_len
    }

    pub fn values(&mut self) -> &mut [T] {
        self.values
    }

    pub fn set(&mut self, row: usize, column: usize, value: T) {
        let index = self.to_index(row, column);
        self.set_by_index(index, value)
    }

    pub fn set_by_index(&mut self, index: usize, value: T) {
        self.values[index] = value
    }

    fn to_index(&self, row: usize, column: usize) -> usize {
        row * self.column_len + column
    }
}

#[derive(Clone, Debug)]
pub struct MatrixBuf<T> {
    column_len: usize,
    values: Vec<T>,
}

impl <T: Clone> MatrixBuf<T> {
    pub fn new(column_len: usize, value: T) -> MatrixBuf<T> {
        use std::iter::repeat;
        MatrixBuf {
            column_len: column_len,
            values: repeat(value).take(column_len*column_len).collect::<Vec<_>>(),
        }
    }

    pub fn as_matrix<'r>(&'r self) -> Matrix<'r, T> {
        Matrix::new(self.column_len, &self.values[..])
    }

    pub fn as_matrix_mut<'r>(&'r mut self) -> MatrixMut<'r, T> {
        MatrixMut::new(self.column_len, &mut self.values[..])
    }
}

#[test]
fn test_new_should_succeed() {
    MatrixBuf::new(42, 4.2);
}

#[test]
fn test_column_len_should_succeed() {
    assert_eq!(MatrixBuf::new(42, 4.2).as_matrix().column_len(), 42);
}

#[test]
fn test_row_should_succeed() {
    let matrix = MatrixBuf::new(42, 4.2);
    assert_eq!(matrix.as_matrix().row(13)[7], 4.2);
}

#[test]
fn test_set_should_succeed() {
    let mut matrix = MatrixBuf::new(42, 4.2);
    matrix.as_matrix_mut().set(13, 7, 0.42);
    assert_eq!(matrix.as_matrix().row(13)[7], 0.42);
}

#[test]
fn test_set_by_index_should_succeed() {
    let mut matrix = MatrixBuf::new(42, 4.2);
    matrix.as_matrix_mut().set_by_index(42 * 13 + 7, 0.42);
    assert_eq!(matrix.as_matrix().row(13)[7], 0.42);
}

#[test]
fn test_partial_eq_should_succeed() {
    assert_eq!(MatrixBuf::new(42, 4.2).as_matrix(),
               MatrixBuf::new(42, 4.2).as_matrix());
}
