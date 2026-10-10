"""
export_onnx.py: Export trained Pages AI models to ONNX format.
Supports dynamic height and width for sliding-window tiled inference.
"""

import os
import sys
import torch
import onnx
import onnxruntime as ort
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.clean_net import PagesCleanNet
from models.geom_net import PagesGeomNet


def export_clean_net(checkpoint_path: str, output_path: str, opset_version: int = 18):
    print(f"\n--- Exporting PagesCleanNet ---")
    print(f"Loading checkpoint: {checkpoint_path}")
    
    device = torch.device("cpu")
    model = PagesCleanNet().to(device)
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Exporting initial weights.")
        
    model.eval()

    # Dummy input: (batch, channels, height, width)
    # Using 512x512 as nominal tile size
    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    # Dynamic axes allow processing any image dimensions or tile sizes in Rust
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"}
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )

    # Verify and bundle into a single self-contained file (no .data sidecars)
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    print("ONNX model validated and saved as self-contained file!")

    # Verify with ONNX Runtime
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": np.random.randn(1, 3, 512, 512).astype(np.float32)}
    ort_outputs = session.run(None, ort_inputs)
    print(f"ONNX Runtime inference verification successful! Output shape: {ort_outputs[0].shape}")
    print(f"Output range: min={ort_outputs[0].min():.4f}, max={ort_outputs[0].max():.4f}")


def export_geom_net(output_path: str, opset_version: int = 18):
    print(f"\n--- Exporting PagesGeomNet ---")
    device = torch.device("cpu")
    model = PagesGeomNet().to(device)
    model.eval()

    # Geometry model runs on fixed canonical size: 512x512 thumbnail
    dummy_input = torch.randn(1, 3, 512, 512, device=device)

    dynamic_axes = {
        "input": {0: "batch_size"},
        "corner_heatmaps": {0: "batch_size"},
        "boundary_mask": {0: "batch_size"}
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["corner_heatmaps", "boundary_mask"],
        dynamic_axes=dynamic_axes
    )

    # Verify and bundle into a single self-contained file (no .data sidecars)
    onnx_model = onnx.load(output_path, load_external_data=True)
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, output_path, save_as_external_data=False)
    data_file = output_path + ".data"
    if os.path.exists(data_file):
        os.remove(data_file)
    print("ONNX GeomNet validated and saved as self-contained file!")

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": np.random.randn(1, 3, 512, 512).astype(np.float32)}
    corners, boundary = session.run(None, ort_inputs)
    print(f"ONNX Runtime GeomNet inference successful! Corners shape: {corners.shape}, Boundary shape: {boundary.shape}")


if __name__ == "__main__":
    checkpoints_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    models_out_dir = os.path.join(os.path.dirname(__file__), "onnx_models")

    clean_ckpt = os.path.join(checkpoints_dir, "pages_clean_best.pt")
    clean_onnx = os.path.join(models_out_dir, "pages_clean_v1.onnx")
    geom_onnx = os.path.join(models_out_dir, "pages_geom_v1.onnx")

    export_clean_net(clean_ckpt, clean_onnx)
    export_geom_net(geom_onnx)
    print("\nAll Pages AI models exported to ONNX successfully!")
