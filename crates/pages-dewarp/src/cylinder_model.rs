use crate::curve::DewarpCurve;
use pages_math::Point2d;

/// 2.5D Cylindrical surface distortion model.
///
/// Models a page curled into the book spine, defined by a top boundary curve
/// and a bottom boundary curve using cylindrical parametric transformation.
#[derive(Debug, Clone, PartialEq)]
pub struct CylindricalSurfaceModel {
    pub top_curve: DewarpCurve,
    pub bottom_curve: DewarpCurve,
    pub target_width: u32,
    pub target_height: u32,
}

impl CylindricalSurfaceModel {
    pub fn new(
        top_curve: DewarpCurve,
        bottom_curve: DewarpCurve,
        target_width: u32,
        target_height: u32,
    ) -> Self {
        Self {
            top_curve,
            bottom_curve,
            target_width,
            target_height,
        }
    }

    /// Maps normalized unrolled coordinates `(u, v)` in `[0.0, 1.0] x [0.0, 1.0]`
    /// to warped source image space `Point2d(x, y)`.
    pub fn map_unrolled_to_source(&self, u: f64, v: f64) -> Point2d {
        let p_top = self.top_curve.evaluate_normalized(u);
        let p_bottom = self.bottom_curve.evaluate_normalized(u);

        // Linear interpolation along rulings between top and bottom curve
        p_top.lerp(p_bottom, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cylinder_mapping() {
        let top = DewarpCurve::new(vec![
            Point2d::new(0.0, 10.0),
            Point2d::new(50.0, 0.0),
            Point2d::new(100.0, 10.0),
        ]);
        let bottom = DewarpCurve::new(vec![
            Point2d::new(0.0, 110.0),
            Point2d::new(50.0, 100.0),
            Point2d::new(100.0, 110.0),
        ]);

        let model = CylindricalSurfaceModel::new(top, bottom, 100, 100);

        let p_mid = model.map_unrolled_to_source(0.5, 0.5);
        assert!((p_mid.x - 50.0).abs() < 1e-4);
        assert!((p_mid.y - 50.0).abs() < 1e-4);
    }
}
