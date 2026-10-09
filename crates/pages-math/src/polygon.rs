use crate::point::Point2d;
use crate::rect::Rect2d;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Polygon2D {
    pub points: Vec<Point2d>,
}

impl Polygon2D {
    pub fn new(points: Vec<Point2d>) -> Self {
        Self { points }
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn bounding_rect(&self) -> Option<Rect2d> {
        if self.points.is_empty() {
            return None;
        }

        let mut min_x = self.points[0].x;
        let mut max_x = self.points[0].x;
        let mut min_y = self.points[0].y;
        let mut max_y = self.points[0].y;

        for p in &self.points[1..] {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }

        Some(Rect2d::new(min_x, min_y, max_x - min_x, max_y - min_y))
    }

    /// Computes signed area using shoelace formula (positive for counter-clockwise)
    pub fn signed_area(&self) -> f64 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.points[i].x * self.points[j].y;
            area -= self.points[j].x * self.points[i].y;
        }
        area * 0.5
    }

    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    pub fn centroid(&self) -> Option<Point2d> {
        let n = self.points.len();
        if n < 3 {
            return None;
        }

        let a = self.signed_area();
        if a.abs() < f64::EPSILON {
            return None;
        }

        let mut cx = 0.0;
        let mut cy = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            let factor = self.points[i].x * self.points[j].y - self.points[j].x * self.points[i].y;
            cx += (self.points[i].x + self.points[j].x) * factor;
            cy += (self.points[i].y + self.points[j].y) * factor;
        }

        let div = 6.0 * a;
        Some(Point2d::new(cx / div, cy / div))
    }

    /// Point-in-polygon test via ray-casting (Jordan curve theorem)
    pub fn contains_point(&self, pt: Point2d) -> bool {
        let n = self.points.len();
        if n < 3 {
            return false;
        }

        let mut inside = false;
        let mut j = n - 1;

        for i in 0..n {
            let pi = self.points[i];
            let pj = self.points[j];

            if ((pi.y > pt.y) != (pj.y > pt.y))
                && (pt.x < (pj.x - pi.x) * (pt.y - pi.y) / (pj.y - pi.y) + pi.x)
            {
                inside = !inside;
            }
            j = i;
        }

        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_area_centroid() {
        let poly = Polygon2D::new(vec![
            Point2d::new(0.0, 0.0),
            Point2d::new(4.0, 0.0),
            Point2d::new(4.0, 4.0),
            Point2d::new(0.0, 4.0),
        ]);

        assert_eq!(poly.area(), 16.0);
        assert_eq!(poly.centroid(), Some(Point2d::new(2.0, 2.0)));
        assert!(poly.contains_point(Point2d::new(2.0, 2.0)));
        assert!(!poly.contains_point(Point2d::new(5.0, 2.0)));
    }
}
