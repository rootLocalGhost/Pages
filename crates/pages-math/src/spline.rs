use crate::point::Point2d;

/// 2D Catmull-Rom and Cubic Spline interpolation.
///
/// Used for smooth curve fitting of book page edges and dewarping meshes.
#[derive(Debug, Clone, PartialEq)]
pub struct CubicSpline2D {
    control_points: Vec<Point2d>,
}

impl CubicSpline2D {
    pub fn new(control_points: Vec<Point2d>) -> Self {
        Self { control_points }
    }

    pub fn points(&self) -> &[Point2d] {
        &self.control_points
    }

    /// Evaluates Catmull-Rom spline at normalized parameter `t` across all segments.
    /// `t` is in `[0.0, (n - 1) as f64]`.
    pub fn evaluate(&self, t: f64) -> Point2d {
        let n = self.control_points.len();
        if n == 0 {
            return Point2d::default();
        }
        if n == 1 {
            return self.control_points[0];
        }

        let num_segments = (n - 1) as f64;
        let clamped_t = t.clamp(0.0, num_segments);
        let segment = (clamped_t.floor() as usize).min(n - 2);
        let local_t = clamped_t - segment as f64;

        let p0 = if segment == 0 {
            self.control_points[0]
        } else {
            self.control_points[segment - 1]
        };
        let p1 = self.control_points[segment];
        let p2 = self.control_points[segment + 1];
        let p3 = if segment + 2 < n {
            self.control_points[segment + 2]
        } else {
            p2
        };

        Self::catmull_rom_segment(p0, p1, p2, p3, local_t)
    }

    /// Computes point on a single Catmull-Rom segment with tension = 0.5.
    fn catmull_rom_segment(p0: Point2d, p1: Point2d, p2: Point2d, p3: Point2d, t: f64) -> Point2d {
        let t2 = t * t;
        let t3 = t2 * t;

        let c0 = -0.5 * t3 + t2 - 0.5 * t;
        let c1 = 1.5 * t3 - 2.5 * t2 + 1.0;
        let c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
        let c3 = 0.5 * t3 - 0.5 * t2;

        p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3
    }

    /// Sample the spline into `samples` equidistant evaluation points.
    pub fn sample_polyline(&self, num_samples: usize) -> Vec<Point2d> {
        let n = self.control_points.len();
        if n < 2 || num_samples < 2 {
            return self.control_points.clone();
        }

        let max_t = (n - 1) as f64;
        let step = max_t / (num_samples - 1) as f64;
        (0..num_samples)
            .map(|i| self.evaluate(i as f64 * step))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cubic_spline() {
        let pts = vec![
            Point2d::new(0.0, 0.0),
            Point2d::new(10.0, 10.0),
            Point2d::new(20.0, 0.0),
            Point2d::new(30.0, 10.0),
        ];
        let spline = CubicSpline2D::new(pts);

        // Start and end should closely match the inner control points
        let p_start = spline.evaluate(1.0);
        assert!((p_start.x - 10.0).abs() < 1e-6);
        assert!((p_start.y - 10.0).abs() < 1e-6);

        let p_mid = spline.evaluate(2.0);
        assert!((p_mid.x - 20.0).abs() < 1e-6);
        assert!((p_mid.y - 0.0).abs() < 1e-6);

        let poly = spline.sample_polyline(31);
        assert_eq!(poly.len(), 31);
    }
}
