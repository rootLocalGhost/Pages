use crate::point::{Point2D, Point2d};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Line2D<T = f64> {
    pub p1: Point2D<T>,
    pub p2: Point2D<T>,
}

pub type Line2d = Line2D<f64>;

impl<T> Line2D<T> {
    #[inline]
    pub const fn new(p1: Point2D<T>, p2: Point2D<T>) -> Self {
        Self { p1, p2 }
    }
}

impl Line2d {
    #[inline]
    pub fn vector(self) -> Point2d {
        self.p2 - self.p1
    }

    #[inline]
    pub fn dx(self) -> f64 {
        self.p2.x - self.p1.x
    }

    #[inline]
    pub fn dy(self) -> f64 {
        self.p2.y - self.p1.y
    }

    #[inline]
    pub fn length_squared(self) -> f64 {
        self.vector().length_squared()
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.vector().length()
    }

    #[inline]
    pub fn point_at(self, t: f64) -> Point2d {
        self.p1 + self.vector() * t
    }

    /// Returns a normal unit vector perpendicular to the line (rotated +90 degrees)
    #[inline]
    pub fn normal_vector(self) -> Option<Point2d> {
        let v = self.vector();
        Point2d::new(-v.y, v.x).normalized()
    }

    /// Angle in radians
    #[inline]
    pub fn angle_rad(self) -> f64 {
        self.dy().atan2(self.dx())
    }

    /// Angle in degrees in range [0, 360)
    #[inline]
    pub fn angle_deg(self) -> f64 {
        let mut deg = self.angle_rad().to_degrees();
        if deg < 0.0 {
            deg += 360.0;
        }
        deg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_methods() {
        let line = Line2d::new(Point2d::new(0.0, 0.0), Point2d::new(4.0, 3.0));
        assert_eq!(line.length(), 5.0);
        assert_eq!(line.point_at(0.5), Point2d::new(2.0, 1.5));
        let norm = line.normal_vector().unwrap();
        assert!((norm.dot(line.vector())).abs() < 1e-9);
    }
}
