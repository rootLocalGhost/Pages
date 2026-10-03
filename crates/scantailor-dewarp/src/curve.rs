use scantailor_math::{CubicSpline2D, Point2d};

/// Represents a curved line (such as the top or bottom edge of a curled book page).
#[derive(Debug, Clone, PartialEq)]
pub struct DewarpCurve {
    spline: CubicSpline2D,
}

impl DewarpCurve {
    pub fn new(control_points: Vec<Point2d>) -> Self {
        Self {
            spline: CubicSpline2D::new(control_points),
        }
    }

    /// Evaluates the curve point at parameter `t` in `[0.0, 1.0]`.
    pub fn evaluate_normalized(&self, t: f64) -> Point2d {
        let n = self.spline.points().len();
        if n < 2 {
            return Point2d::default();
        }
        let max_t = (n - 1) as f64;
        self.spline.evaluate(t * max_t)
    }

    /// Samples the curve into `num_points` evenly spaced points.
    pub fn sample_points(&self, num_points: usize) -> Vec<Point2d> {
        self.spline.sample_polyline(num_points)
    }

    /// Calculates total approximate arc length along the sampled curve.
    pub fn arc_length(&self, num_samples: usize) -> f64 {
        let pts = self.sample_points(num_samples);
        let mut length = 0.0;
        for i in 1..pts.len() {
            length += pts[i].distance_to(pts[i - 1]);
        }
        length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dewarp_curve_arc_length() {
        let pts = vec![
            Point2d::new(0.0, 0.0),
            Point2d::new(50.0, 10.0),
            Point2d::new(100.0, 0.0),
        ];
        let curve = DewarpCurve::new(pts);
        let len = curve.arc_length(50);
        assert!(len > 100.0 && len < 110.0);
    }
}
