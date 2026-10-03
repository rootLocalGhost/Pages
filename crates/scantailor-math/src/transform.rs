use crate::point::Point2d;

/// 3x3 Projective Homography Transform.
///
/// Ported from Scan Tailor's `HomographicTransform<2, double>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography2D {
    // Row-major 3x3 matrix:
    // [ m00, m01, m02 ]
    // [ m10, m11, m12 ]
    // [ m20, m21, m22 ]
    pub m: [f64; 9],
}

impl Default for Homography2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl Homography2D {
    pub const fn identity() -> Self {
        Self {
            m: [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ],
        }
    }

    pub const fn from_matrix(m: [f64; 9]) -> Self {
        Self { m }
    }

    /// Transforms a 2D point using projective homogenous coordinates.
    pub fn transform_point(&self, pt: Point2d) -> Point2d {
        let w = self.m[6] * pt.x + self.m[7] * pt.y + self.m[8];
        let inv_w = if w.abs() > f64::EPSILON {
            1.0 / w
        } else {
            1.0
        };

        let x = (self.m[0] * pt.x + self.m[1] * pt.y + self.m[2]) * inv_w;
        let y = (self.m[3] * pt.x + self.m[4] * pt.y + self.m[5]) * inv_w;
        Point2d::new(x, y)
    }

    /// Computes the 3x3 matrix determinant.
    pub fn determinant(&self) -> f64 {
        let m = &self.m;
        m[0] * (m[4] * m[8] - m[5] * m[7])
            - m[1] * (m[3] * m[8] - m[5] * m[6])
            + m[2] * (m[3] * m[7] - m[4] * m[6])
    }

    /// Inverts the homography transform.
    pub fn invert(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f64::EPSILON {
            return None;
        }

        let inv_det = 1.0 / det;
        let m = &self.m;

        let inv = [
            (m[4] * m[8] - m[5] * m[7]) * inv_det,
            (m[2] * m[7] - m[1] * m[8]) * inv_det,
            (m[1] * m[5] - m[2] * m[4]) * inv_det,
            (m[5] * m[6] - m[3] * m[8]) * inv_det,
            (m[0] * m[8] - m[2] * m[6]) * inv_det,
            (m[2] * m[3] - m[0] * m[5]) * inv_det,
            (m[3] * m[7] - m[4] * m[6]) * inv_det,
            (m[1] * m[6] - m[0] * m[7]) * inv_det,
            (m[0] * m[4] - m[1] * m[3]) * inv_det,
        ];

        Some(Self { m: inv })
    }

    /// Multiplies this matrix with another: `self * other`
    pub fn multiply(&self, rhs: &Self) -> Self {
        let mut res = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                res[row * 3 + col] = self.m[row * 3] * rhs.m[col]
                    + self.m[row * 3 + 1] * rhs.m[3 + col]
                    + self.m[row * 3 + 2] * rhs.m[6 + col];
            }
        }
        Self { m: res }
    }

    /// Computes the 3x3 perspective homography mapping 4 source points to 4 destination points.
    pub fn from_four_point_correspondences(
        src: [Point2d; 4],
        dst: [Point2d; 4],
    ) -> Option<Self> {
        // Direct Linear Transformation (DLT) with 8 equations for 8 parameters (h22 = 1)
        let mut a = [[0.0; 8]; 8];
        let mut b = [0.0; 8];

        for i in 0..4 {
            let sx = src[i].x;
            let sy = src[i].y;
            let dx = dst[i].x;
            let dy = dst[i].y;

            let row1 = i * 2;
            let row2 = row1 + 1;

            a[row1][0] = sx;
            a[row1][1] = sy;
            a[row1][2] = 1.0;
            a[row1][6] = -dx * sx;
            a[row1][7] = -dx * sy;
            b[row1] = dx;

            a[row2][3] = sx;
            a[row2][4] = sy;
            a[row2][5] = 1.0;
            a[row2][6] = -dy * sx;
            a[row2][7] = -dy * sy;
            b[row2] = dy;
        }

        // Gaussian elimination with partial pivoting for 8x8 system
        let mut h = [0.0; 8];
        if !solve_linear_system_8x8(&mut a, &mut b, &mut h) {
            return None;
        }

        Some(Self {
            m: [
                h[0], h[1], h[2],
                h[3], h[4], h[5],
                h[6], h[7], 1.0,
            ],
        })
    }
}

/// 2D Affine Transformation (rotation, scale, shear, translation)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineTransform2D {
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    pub tx: f64,
    pub ty: f64,
}

impl Default for AffineTransform2D {
    fn default() -> Self {
        Self::identity()
    }
}

impl AffineTransform2D {
    pub const fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx,
            ty,
        }
    }

    pub fn rotation_rad(rad: f64) -> Self {
        let cos_a = rad.cos();
        let sin_a = rad.sin();
        Self {
            a: cos_a,
            b: -sin_a,
            c: sin_a,
            d: cos_a,
            tx: 0.0,
            ty: 0.0,
        }
    }

    pub fn rotation_around(center: Point2d, rad: f64) -> Self {
        let t1 = Self::translation(-center.x, -center.y);
        let rot = Self::rotation_rad(rad);
        let t2 = Self::translation(center.x, center.y);
        t2.multiply(&rot.multiply(&t1))
    }

    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            tx: 0.0,
            ty: 0.0,
        }
    }

    pub fn transform_point(&self, pt: Point2d) -> Point2d {
        Point2d::new(
            self.a * pt.x + self.b * pt.y + self.tx,
            self.c * pt.x + self.d * pt.y + self.ty,
        )
    }

    pub fn multiply(&self, rhs: &Self) -> Self {
        Self {
            a: self.a * rhs.a + self.b * rhs.c,
            b: self.a * rhs.b + self.b * rhs.d,
            c: self.c * rhs.a + self.d * rhs.c,
            d: self.c * rhs.b + self.d * rhs.d,
            tx: self.a * rhs.tx + self.b * rhs.ty + self.tx,
            ty: self.c * rhs.tx + self.d * rhs.ty + self.ty,
        }
    }

    pub fn to_homography(&self) -> Homography2D {
        Homography2D {
            m: [
                self.a, self.b, self.tx,
                self.c, self.d, self.ty,
                0.0,    0.0,    1.0,
            ],
        }
    }
}

fn solve_linear_system_8x8(
    a: &mut [[f64; 8]; 8],
    b: &mut [f64; 8],
    x: &mut [f64; 8],
) -> bool {
    const N: usize = 8;
    for i in 0..N {
        // Pivot selection
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for k in (i + 1)..N {
            let val = a[k][i].abs();
            if val > max_val {
                max_val = val;
                max_row = k;
            }
        }

        if max_val < 1e-12 {
            return false;
        }

        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }

        // Eliminate
        let pivot = a[i][i];
        for k in (i + 1)..N {
            let factor = a[k][i] / pivot;
            a[k][i] = 0.0;
            for j in (i + 1)..N {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Back substitution
    for i in (0..N).rev() {
        let mut sum = b[i];
        for j in (i + 1)..N {
            sum -= a[i][j] * x[j];
        }
        x[i] = sum / a[i][i];
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homography_identity() {
        let h = Homography2D::identity();
        let pt = Point2d::new(12.0, 34.0);
        assert_eq!(h.transform_point(pt), pt);
    }

    #[test]
    fn test_homography_inversion() {
        let aff = AffineTransform2D::rotation_around(Point2d::new(10.0, 10.0), 0.5);
        let h = aff.to_homography();
        let h_inv = h.invert().unwrap();

        let pt = Point2d::new(25.0, 35.0);
        let transformed = h.transform_point(pt);
        let restored = h_inv.transform_point(transformed);

        assert!((restored.x - pt.x).abs() < 1e-6);
        assert!((restored.y - pt.y).abs() < 1e-6);
    }

    #[test]
    fn test_four_point_correspondence() {
        let src = [
            Point2d::new(0.0, 0.0),
            Point2d::new(100.0, 0.0),
            Point2d::new(100.0, 100.0),
            Point2d::new(0.0, 100.0),
        ];
        let dst = [
            Point2d::new(10.0, 20.0),
            Point2d::new(120.0, 15.0),
            Point2d::new(115.0, 125.0),
            Point2d::new(5.0, 110.0),
        ];

        let h = Homography2D::from_four_point_correspondences(src, dst).unwrap();
        for i in 0..4 {
            let mapped = h.transform_point(src[i]);
            assert!((mapped.x - dst[i].x).abs() < 1e-4);
            assert!((mapped.y - dst[i].y).abs() < 1e-4);
        }
    }
}
