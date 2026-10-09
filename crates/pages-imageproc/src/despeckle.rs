use crate::binarize::otsu_threshold;
use crate::enhance::to_grayscale;
use image::{Rgb, RgbImage};

/// Purge isolated toner dust and scanner glass specks (< min_size pixels)
pub fn despeckle(img: &RgbImage, min_size: usize) -> RgbImage {
    let (w, h) = img.dimensions();
    let width = w as usize;
    let height = h as usize;

    let gray = to_grayscale(img);
    let thresh = otsu_threshold(&gray);

    let mut is_dark = vec![false; width * height];
    for y in 0..height {
        let row_start = y * width;
        for x in 0..width {
            let luma = gray.get_pixel(x as u32, y as u32)[0];
            is_dark[row_start + x] = luma <= thresh;
        }
    }

    let mut visited = vec![false; width * height];
    let mut out = img.clone();

    // 8-connected neighborhood offsets
    let neighbors: [(isize, isize); 8] = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];

    let mut queue = Vec::new();
    let mut component = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if !is_dark[idx] || visited[idx] {
                continue;
            }

            // Start BFS to collect this connected component
            visited[idx] = true;
            queue.clear();
            component.clear();

            queue.push((x as isize, y as isize));
            component.push((x as u32, y as u32));

            let mut head = 0;
            while head < queue.len() {
                let (cx, cy) = queue[head];
                head += 1;

                // Stop expanding if component is clearly larger than a speck
                if component.len() > min_size * 2 {
                    break;
                }

                for &(dx, dy) in &neighbors {
                    let nx = cx + dx;
                    let ny = cy + dy;

                    if nx >= 0 && nx < w as isize && ny >= 0 && ny < h as isize {
                        let n_idx = (ny as usize) * width + (nx as usize);
                        if is_dark[n_idx] && !visited[n_idx] {
                            visited[n_idx] = true;
                            queue.push((nx, ny));
                            component.push((nx as u32, ny as u32));
                        }
                    }
                }
            }

            // If the component is a tiny noise speck, wipe it to pure white
            if component.len() <= min_size {
                for &(px, py) in &component {
                    out.put_pixel(px, py, Rgb([255, 255, 255]));
                }
            }
        }
    }

    out
}
