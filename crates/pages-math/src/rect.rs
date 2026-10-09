use crate::point::Point2d;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Rect2D<T = f64> {
    pub x: T,
    pub y: T,
    pub width: T,
    pub height: T,
}

pub type Rect2d = Rect2D<f64>;
pub type Rect2i = Rect2D<i32>;

impl<T> Rect2D<T> {
    #[inline]
    pub const fn new(x: T, y: T, width: T, height: T) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

impl Rect2d {
    #[inline]
    pub fn from_points(p1: Point2d, p2: Point2d) -> Self {
        let min_x = p1.x.min(p2.x);
        let min_y = p1.y.min(p2.y);
        let max_x = p1.x.max(p2.x);
        let max_y = p1.y.max(p2.y);
        Self::new(min_x, min_y, max_x - min_x, max_y - min_y)
    }

    #[inline]
    pub fn left(self) -> f64 {
        self.x
    }

    #[inline]
    pub fn right(self) -> f64 {
        self.x + self.width
    }

    #[inline]
    pub fn top(self) -> f64 {
        self.y
    }

    #[inline]
    pub fn bottom(self) -> f64 {
        self.y + self.height
    }

    #[inline]
    pub fn top_left(self) -> Point2d {
        Point2d::new(self.left(), self.top())
    }

    #[inline]
    pub fn top_right(self) -> Point2d {
        Point2d::new(self.right(), self.top())
    }

    #[inline]
    pub fn bottom_left(self) -> Point2d {
        Point2d::new(self.left(), self.bottom())
    }

    #[inline]
    pub fn bottom_right(self) -> Point2d {
        Point2d::new(self.right(), self.bottom())
    }

    #[inline]
    pub fn center(self) -> Point2d {
        Point2d::new(self.x + self.width * 0.5, self.y + self.height * 0.5)
    }

    #[inline]
    pub fn contains_point(self, pt: Point2d) -> bool {
        pt.x >= self.left() && pt.x <= self.right() && pt.y >= self.top() && pt.y <= self.bottom()
    }

    #[inline]
    pub fn intersects(self, other: Self) -> bool {
        self.left() <= other.right()
            && self.right() >= other.left()
            && self.top() <= other.bottom()
            && self.bottom() >= other.top()
    }

    #[inline]
    pub fn intersection(self, other: Self) -> Option<Self> {
        let left = self.left().max(other.left());
        let right = self.right().min(other.right());
        let top = self.top().max(other.top());
        let bottom = self.bottom().min(other.bottom());

        if left <= right && top <= bottom {
            Some(Self::new(left, top, right - left, bottom - top))
        } else {
            None
        }
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        let left = self.left().min(other.left());
        let right = self.right().max(other.right());
        let top = self.top().min(other.top());
        let bottom = self.bottom().max(other.bottom());
        Self::new(left, top, right - left, bottom - top)
    }

    #[inline]
    pub fn adjusted(self, dl: f64, dt: f64, dr: f64, db: f64) -> Self {
        Self::new(
            self.x + dl,
            self.y + dt,
            self.width - dl + dr,
            self.height - dt + db,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rect_bounds() {
        let r = Rect2d::new(10.0, 20.0, 30.0, 40.0);
        assert_eq!(r.left(), 10.0);
        assert_eq!(r.right(), 40.0);
        assert_eq!(r.top(), 20.0);
        assert_eq!(r.bottom(), 60.0);
        assert_eq!(r.center(), Point2d::new(25.0, 40.0));
        assert!(r.contains_point(Point2d::new(20.0, 30.0)));
        assert!(!r.contains_point(Point2d::new(5.0, 30.0)));
    }

    #[test]
    fn test_rect_intersection() {
        let r1 = Rect2d::new(0.0, 0.0, 20.0, 20.0);
        let r2 = Rect2d::new(10.0, 10.0, 20.0, 20.0);
        let inter = r1.intersection(r2).unwrap();
        assert_eq!(inter, Rect2d::new(10.0, 10.0, 10.0, 10.0));
    }
}
