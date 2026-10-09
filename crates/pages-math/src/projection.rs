use crate::line::Line2d;
use crate::point::Point2d;

/// Projects points onto an infinite 2D line.
#[derive(Debug, Clone, Copy)]
pub struct ToLineProjector {
    origin: Point2d,
    vec: Point2d,
    mat: Point2d,
}

impl ToLineProjector {
    pub fn new(line: Line2d) -> Self {
        let origin = line.p1;
        let vec = line.vector();
        let mut mat = vec;

        let at_a = mat.dot(mat);
        if at_a.abs() > f64::EPSILON {
            mat /= at_a;
        } else {
            mat = Point2d::new(0.0, 0.0);
        }

        Self { origin, vec, mat }
    }

    /// Solves `line.p1 + s * (line.p2 - line.p1) = projection_point`
    #[inline]
    pub fn projection_scalar(&self, pt: Point2d) -> f64 {
        let b = pt - self.origin;
        self.mat.dot(b)
    }

    /// Finds the closest point on the line to `pt`
    #[inline]
    pub fn projection_point(&self, pt: Point2d) -> Point2d {
        self.origin + self.vec * self.projection_scalar(pt)
    }

    /// Equivalent to `projection_point(pt) - pt`
    #[inline]
    pub fn projection_vector(&self, pt: Point2d) -> Point2d {
        self.projection_point(pt) - pt
    }

    /// Distance from `pt` to the infinite line
    #[inline]
    pub fn distance_to(&self, pt: Point2d) -> f64 {
        self.projection_vector(pt).length()
    }
}

/// Checks whether points `p1` and `p2` lie on the same side of `line`.
///
/// Returns:
/// - `> 0` if both points are strictly on the same side
/// - `< 0` if they are on opposite sides
/// - `0` if at least one point lies on the line
pub fn sides_of_line(line: Line2d, p1: Point2d, p2: Point2d) -> f64 {
    let normal = match line.normal_vector() {
        Some(n) => n,
        None => return 0.0,
    };

    let v1 = p1 - line.p1;
    let v2 = p2 - line.p1;

    let dot1 = normal.dot(v1);
    let dot2 = normal.dot(v2);

    dot1 * dot2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_line_projector() {
        let line = Line2d::new(Point2d::new(0.0, 0.0), Point2d::new(10.0, 0.0));
        let proj = ToLineProjector::new(line);

        let pt = Point2d::new(5.0, 3.0);
        assert_eq!(proj.projection_scalar(pt), 0.5);
        assert_eq!(proj.projection_point(pt), Point2d::new(5.0, 0.0));
        assert_eq!(proj.distance_to(pt), 3.0);
    }

    #[test]
    fn test_sides_of_line() {
        let line = Line2d::new(Point2d::new(0.0, 0.0), Point2d::new(10.0, 0.0));
        let p_above1 = Point2d::new(2.0, 5.0);
        let p_above2 = Point2d::new(8.0, 3.0);
        let p_below = Point2d::new(4.0, -2.0);

        assert!(sides_of_line(line, p_above1, p_above2) > 0.0);
        assert!(sides_of_line(line, p_above1, p_below) < 0.0);
    }
}
