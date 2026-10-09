use crate::binary_image::BinaryImage;
use pages_math::{Point2d, Point2i, Rect2i};

#[derive(Debug, Clone, PartialEq)]
pub struct ConnectedComponent {
    pub id: u32,
    pub bounding_box: Rect2i,
    pub pixel_count: usize,
    pub centroid: Point2d,
    pub seed: Point2i,
}

/// Connected Components Labeling (CCL) using Two-Pass Disjoint-Set (Union-Find) algorithm.
pub struct ConnectedComponentLabeler {
    pub width: u32,
    pub height: u32,
    pub labels: Vec<u32>,
    pub components: Vec<ConnectedComponent>,
}

impl ConnectedComponentLabeler {
    /// Runs 8-connected labeling on a `BinaryImage` (where `true` = black / foreground).
    pub fn label_8_connected(img: &BinaryImage) -> Self {
        let w = img.width as usize;
        let h = img.height as usize;
        let mut labels = vec![0u32; w * h];
        let mut parent: Vec<u32> = vec![0]; // label 0 is background

        let mut next_label = 1u32;

        fn find(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                parent[i as usize] = parent[parent[i as usize] as usize];
                i = parent[i as usize];
            }
            i
        }

        fn union(parent: &mut [u32], i: u32, j: u32) {
            let root_i = find(parent, i);
            let root_j = find(parent, j);
            if root_i != root_j {
                parent[root_j as usize] = root_i;
            }
        }

        // First pass
        for y in 0..h {
            for x in 0..w {
                if !img.get_pixel(x as u32, y as u32) {
                    continue;
                }

                // Check 4 already-visited neighbors for 8-connectivity:
                // (x-1, y-1), (x, y-1), (x+1, y-1), (x-1, y)
                let mut neighbor_labels = [0u32; 4];
                let mut num_neighbors = 0;

                if x > 0 && y > 0 {
                    let l = labels[(y - 1) * w + (x - 1)];
                    if l > 0 {
                        neighbor_labels[num_neighbors] = l;
                        num_neighbors += 1;
                    }
                }
                if y > 0 {
                    let l = labels[(y - 1) * w + x];
                    if l > 0 {
                        neighbor_labels[num_neighbors] = l;
                        num_neighbors += 1;
                    }
                }
                if x + 1 < w && y > 0 {
                    let l = labels[(y - 1) * w + (x + 1)];
                    if l > 0 {
                        neighbor_labels[num_neighbors] = l;
                        num_neighbors += 1;
                    }
                }
                if x > 0 {
                    let l = labels[y * w + (x - 1)];
                    if l > 0 {
                        neighbor_labels[num_neighbors] = l;
                        num_neighbors += 1;
                    }
                }

                if num_neighbors == 0 {
                    let l = next_label;
                    parent.push(l);
                    labels[y * w + x] = l;
                    next_label += 1;
                } else {
                    let min_l = neighbor_labels[0..num_neighbors]
                        .iter()
                        .copied()
                        .min()
                        .unwrap();
                    labels[y * w + x] = min_l;

                    for &n_label in &neighbor_labels[0..num_neighbors] {
                        union(&mut parent, min_l, n_label);
                    }
                }
            }
        }

        // Second pass: flatten roots and collect stats
        struct CompStats {
            min_x: i32,
            max_x: i32,
            min_y: i32,
            max_y: i32,
            pixel_count: usize,
            sum_x: f64,
            sum_y: f64,
            seed: Point2i,
        }

        let mut stats_map = std::collections::HashMap::<u32, CompStats>::new();

        for y in 0..h {
            for x in 0..w {
                let l = labels[y * w + x];
                if l == 0 {
                    continue;
                }
                let root = find(&mut parent, l);
                labels[y * w + x] = root;

                let xi = x as i32;
                let yi = y as i32;

                stats_map
                    .entry(root)
                    .and_modify(|s| {
                        s.min_x = s.min_x.min(xi);
                        s.max_x = s.max_x.max(xi);
                        s.min_y = s.min_y.min(yi);
                        s.max_y = s.max_y.max(yi);
                        s.pixel_count += 1;
                        s.sum_x += xi as f64;
                        s.sum_y += yi as f64;
                    })
                    .or_insert(CompStats {
                        min_x: xi,
                        max_x: xi,
                        min_y: yi,
                        max_y: yi,
                        pixel_count: 1,
                        sum_x: xi as f64,
                        sum_y: yi as f64,
                        seed: Point2i::new(xi, yi),
                    });
            }
        }

        let mut components = Vec::with_capacity(stats_map.len());
        for (id, s) in stats_map {
            let count_f = s.pixel_count as f64;
            components.push(ConnectedComponent {
                id,
                bounding_box: Rect2i::new(
                    s.min_x,
                    s.min_y,
                    s.max_x - s.min_x + 1,
                    s.max_y - s.min_y + 1,
                ),
                pixel_count: s.pixel_count,
                centroid: Point2d::new(s.sum_x / count_f, s.sum_y / count_f),
                seed: s.seed,
            });
        }

        Self {
            width: img.width,
            height: img.height,
            labels,
            components,
        }
    }

    /// Despeckle: produces a new binary image with components having fewer than `min_pixels` removed.
    pub fn despeckle(&self, min_pixels: usize) -> BinaryImage {
        let valid_labels: std::collections::HashSet<u32> = self
            .components
            .iter()
            .filter(|c| c.pixel_count >= min_pixels)
            .map(|c| c.id)
            .collect();

        let mut out = BinaryImage::new(self.width, self.height);
        let w = self.width as usize;
        let h = self.height as usize;

        for y in 0..h {
            for x in 0..w {
                let l = self.labels[y * w + x];
                if l > 0 && valid_labels.contains(&l) {
                    out.set_pixel(x as u32, y as u32, true);
                }
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccl_and_despeckle() {
        let mut img = BinaryImage::new(50, 50);

        // Component 1: 3x3 block (9 pixels)
        for y in 10..13 {
            for x in 10..13 {
                img.set_pixel(x, y, true);
            }
        }

        // Component 2: single isolated noise pixel (speckle)
        img.set_pixel(40, 40, true);

        let ccl = ConnectedComponentLabeler::label_8_connected(&img);
        assert_eq!(ccl.components.len(), 2);

        let clean = ccl.despeckle(2);
        assert_eq!(clean.count_black_pixels(), 9);
        assert!(!clean.get_pixel(40, 40));
        assert!(clean.get_pixel(10, 10));
    }
}
