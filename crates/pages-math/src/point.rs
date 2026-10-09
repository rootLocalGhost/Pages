use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Point2D<T = f64> {
    pub x: T,
    pub y: T,
}

pub type Point2d = Point2D<f64>;
pub type Point2i = Point2D<i32>;

impl<T> Point2D<T> {
    #[inline]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

impl<T: Copy> Point2D<T> {
    #[inline]
    pub fn to_array(self) -> [T; 2] {
        [self.x, self.y]
    }
}

impl Point2d {
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn distance_to(self, other: Self) -> f64 {
        (self - other).length()
    }

    #[inline]
    pub fn normalized(self) -> Option<Self> {
        let len = self.length();
        if len > f64::EPSILON {
            Some(self / len)
        } else {
            None
        }
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        self + (other - self) * t
    }

    #[inline]
    pub fn rotate(self, angle_rad: f64) -> Self {
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        Self::new(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )
    }
}

impl<T: Add<Output = T>> Add for Point2D<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl<T: AddAssign> AddAssign for Point2D<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: Sub<Output = T>> Sub for Point2D<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl<T: SubAssign> SubAssign for Point2D<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Point2D<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl<T: MulAssign + Copy> MulAssign<T> for Point2D<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl<T: Div<Output = T> + Copy> Div<T> for Point2D<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl<T: DivAssign + Copy> DivAssign<T> for Point2D<T> {
    #[inline]
    fn div_assign(&mut self, rhs: T) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

impl<T: Neg<Output = T>> Neg for Point2D<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_arithmetic() {
        let p1 = Point2d::new(1.0, 2.0);
        let p2 = Point2d::new(3.0, 4.0);
        assert_eq!(p1 + p2, Point2d::new(4.0, 6.0));
        assert_eq!(p2 - p1, Point2d::new(2.0, 2.0));
        assert_eq!(p1 * 2.0, Point2d::new(2.0, 4.0));
        assert_eq!(p2 / 2.0, Point2d::new(1.5, 2.0));
    }

    #[test]
    fn test_point_geometry() {
        let p1 = Point2d::new(3.0, 4.0);
        assert_eq!(p1.length(), 5.0);
        let norm = p1.normalized().unwrap();
        assert!((norm.length() - 1.0).abs() < 1e-9);
        assert_eq!(p1.dot(Point2d::new(4.0, -3.0)), 0.0);
    }
}
