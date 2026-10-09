/// Multi-variable Linear Function: `F(x) = a^T * x + b`
#[derive(Debug, Clone, PartialEq)]
pub struct LinearFunction {
    pub a: Vec<f64>,
    pub b: f64,
}

impl LinearFunction {
    pub fn new(num_vars: usize) -> Self {
        Self {
            a: vec![0.0; num_vars],
            b: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.a.fill(0.0);
        self.b = 0.0;
    }

    pub fn num_vars(&self) -> usize {
        self.a.len()
    }

    pub fn evaluate(&self, x: &[f64]) -> f64 {
        assert_eq!(self.a.len(), x.len(), "Dimension mismatch in evaluate");
        let mut sum = self.b;
        for (&ai, &xi) in self.a.iter().zip(x.iter()) {
            sum += ai * xi;
        }
        sum
    }

    pub fn add_assign(&mut self, other: &Self) {
        assert_eq!(
            self.a.len(),
            other.a.len(),
            "Dimension mismatch in addition"
        );
        for (ai, &o_ai) in self.a.iter_mut().zip(other.a.iter()) {
            *ai += o_ai;
        }
        self.b += other.b;
    }

    pub fn scale_assign(&mut self, scalar: f64) {
        for val in &mut self.a {
            *val *= scalar;
        }
        self.b *= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_function() {
        let mut f = LinearFunction::new(3);
        f.a = vec![2.0, -1.0, 3.0];
        f.b = 5.0;

        let x = vec![1.0, 2.0, 3.0];
        // 2*1 + (-1)*2 + 3*3 + 5 = 2 - 2 + 9 + 5 = 14
        assert_eq!(f.evaluate(&x), 14.0);
    }
}
