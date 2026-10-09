use crate::line::Line2d;
use crate::point::Point2d;
use crate::rect::Rect2d;

/// Solves for scalars `s1` and `s2` such that:
/// `line1.p1 + s1 * line1.vector() == line2.p1 + s2 * line2.vector()`
///
/// Returns `Some((s1, s2))` if the lines intersect, or `None` if they are parallel.
pub fn line_intersection_scalar(line1: Line2d, line2: Line2d) -> Option<(f64, f64)> {
    let p1 = line1.p1;
    let p2 = line2.p1;
    let v1 = line1.vector();
    let v2 = line2.vector();

    // System of equations:
    // s1 * v1.x - s2 * v2.x = p2.x - p1.x
    // s1 * v1.y - s2 * v2.y = p2.y - p1.y
    // In matrix form: [v1 -v2] * [s1; s2] = p2 - p1
    let det_a = v2.x * v1.y - v1.x * v2.y;
    if det_a.abs() < f64::EPSILON {
        return None;
    }

    let r_det_a = 1.0 / det_a;
    let b = p2 - p1;
    let s1 = (v2.x * b.y - b.x * v2.y) * r_det_a;
    let s2 = (v1.x * b.y - b.x * v1.y) * r_det_a;

    Some((s1, s2))
}

/// Computes the intersection point of two lines, if any.
pub fn line_intersection_point(line1: Line2d, line2: Line2d) -> Option<Point2d> {
    line_intersection_scalar(line1, line2).map(|(s1, _)| line1.point_at(s1))
}

/// Clips an infinite line by a rectangle, returning the segment inside the rectangle.
pub fn line_bounded_by_rect(line: Line2d, rect: Rect2d) -> Option<Line2d> {
    let rect_lines = [
        Line2d::new(rect.top_left(), rect.top_right()),
        Line2d::new(rect.bottom_left(), rect.bottom_right()),
        Line2d::new(rect.top_left(), rect.bottom_left()),
        Line2d::new(rect.top_right(), rect.bottom_right()),
    ];

    let mut max_s = f64::NEG_INFINITY;
    let mut min_s = f64::INFINITY;

    for &rect_line in &rect_lines {
        if let Some((s1, s2)) = line_intersection_scalar(rect_line, line) {
            // Check if intersection is on the rect's edge segment
            if (0.0..=1.0).contains(&s1) {
                if s2 > max_s {
                    max_s = s2;
                }
                if s2 < min_s {
                    min_s = s2;
                }
            }
        }
    }

    if max_s > min_s {
        Some(Line2d::new(line.point_at(min_s), line.point_at(max_s)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_intersection() {
        let l1 = Line2d::new(Point2d::new(0.0, 5.0), Point2d::new(10.0, 5.0));
        let l2 = Line2d::new(Point2d::new(5.0, 0.0), Point2d::new(5.0, 10.0));

        let (s1, s2) = line_intersection_scalar(l1, l2).unwrap();
        assert_eq!(s1, 0.5);
        assert_eq!(s2, 0.5);

        let pt = line_intersection_point(l1, l2).unwrap();
        assert_eq!(pt, Point2d::new(5.0, 5.0));
    }

    #[test]
    fn test_line_bounded_by_rect() {
        let l = Line2d::new(Point2d::new(-10.0, 5.0), Point2d::new(20.0, 5.0));
        let rect = Rect2d::new(0.0, 0.0, 10.0, 10.0);

        let clipped = line_bounded_by_rect(l, rect).unwrap();
        assert!((clipped.p1.x - 0.0).abs() < 1e-9);
        assert!((clipped.p1.y - 5.0).abs() < 1e-9);
        assert!((clipped.p2.x - 10.0).abs() < 1e-9);
        assert!((clipped.p2.y - 5.0).abs() < 1e-9);
    }
}
