/// Multi-variable Quadratic Function: `F(x) = x^T * A * x + b^T * x + c`
///
/// Ported from Scan Tailor's `QuadraticFunction`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadraticFunction {
    /// N x N matrix stored in row-major order (size N * N)
    pub a: Vec<f64>,
    /// Vector of N linear coefficients
    pub b: Vec<f64>,
    /// Constant term
    pub c: f64,
    num_vars: usize,
}

impl QuadraticFunction {
    pub fn new(num_vars: usize) -> Self {
        Self {
            a: vec![0.0; num_vars * num_vars],
            b: vec![0.0; num_vars],
            c: 0.0,
            num_vars,
        }
    }

    pub fn reset(&mut self) {
        self.a.fill(0.0);
        self.b.fill(0.0);
        self.c = 0.0;
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(self.num_vars, x.len(), "Dimension mismatch");
        let n = self.num_vars;
        let mut sum = self.c;

        for i in 0..n {
            sum += self.b[i] * x[i];
            let mut ax_i = 0.0;
            for j in 0..n {
                ax_i += self.a[i * n + j] * x[j];
            }
            sum += x[i] * ax_i;
        }

        sum
    }

    /// Evaluates gradient `nabla F(x) = (A + A^T) * x + b` (for symmetric A: `2*A*x + b`)
    pub fn gradient_at(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(self.num_vars, x.len(), "Dimension mismatch");
        let n = self.num_vars;
        let mut grad = self.b.clone();

        for i in 0..n {
            for j in 0..n {
                grad[i] += (self.a[i * n + j] + self.a[j * n + i]) * x[j];
            }
        }

        grad
    }

    pub fn add_assign(&mut self, other: &Self) {
        assert_eq!(self.num_vars, other.num_vars, "Dimension mismatch");
        for i in 0..self.a.len() {
            self.a[i] += other.a[i];
        }
        for i in 0..self.b.len() {
            self.b[i] += other.b[i];
        }
        self.c += other.c;
    }

    pub fn scale_assign(&mut self, scalar: f64) {
        for v in &mut self.a {
            *v *= scalar;
        }
        for v in &mut self.b {
            *v *= scalar;
        }
        self.c *= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_function() {
        let mut q = QuadraticFunction::new(2);
        // F(x, y) = x^2 + 2y^2 + 3x + 4y + 5
        // A = [[1, 0], [0, 2]]
        q.a[0] = 1.0;
        q.a[3] = 2.0;
        q.b[0] = 3.0;
        q.b[1] = 4.0;
        q.c = 5.0;

        let x = [2.0, 3.0];
        // 2^2 + 2*(3^2) + 3*2 + 4*3 + 5 = 4 + 18 + 6 + 12 + 5 = 45
        assert_eq!(q.evaluate(&x), 45.0);

        let grad = q.gradient_at(&x);
        // dF/dx = 2x + 3 = 2(2) + 3 = 7
        // dF/dy = 4y + 4 = 4(3) + 4 = 16
        assert_eq!(grad, vec![7.0, 16.0]);
    }
}
