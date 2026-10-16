#[cfg(feature = "ai")]
#[test]
fn test_pages_ai_engine_load_and_infer() {
    use image::{Rgb, RgbImage};
    use pages_core::PagesAiEngine;
    use std::path::Path;

    // Path to model in standalone Pages-AI project
    let model_path = Path::new("c:/Users/LEADER/Desktop/Pages-AI/onnx_models/pages_clean_v1.onnx");
    if !model_path.exists() {
        eprintln!("Model not found at {:?}, skipping test", model_path);
        return;
    }

    let engine = PagesAiEngine::load_from_file(model_path).expect("Failed to load ONNX model");
    assert!(engine.is_ready());

    // Create a 64x64 synthetic page tile
    let mut img = RgbImage::new(64, 64);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        if (x + y) % 2 == 0 {
            *pixel = Rgb([220, 220, 220]);
        } else {
            *pixel = Rgb([40, 40, 40]);
        }
    }

    let cleaned = engine.clean_page(&img).expect("Inference failed");
    assert_eq!(cleaned.width(), 64);
    assert_eq!(cleaned.height(), 64);
    println!("PagesAiEngine test completed successfully!");
}
