"""
download_datasets.py: Tooling to download, extract, and index academic document datasets:
1. DIBCO (Document Image Binarization Contest - degraded historical documents, bleed-through, stains)
2. SmartDoc (Smartphone Document capture dataset - perspective distortion, boundary ground truth)
3. Sample public test pages for local validation
"""

import os
import sys
import urllib.request
import zipfile
import tarfile

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "raw")

# Public curated document benchmarks and samples
DATASET_URLS = {
    "sample_docs": "https://raw.githubusercontent.com/tesseract-ocr/test/master/testing/",
}

def setup_directories():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "clean"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "degraded"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "geometric"), exist_ok=True)
    print(f"Dataset root initialized at: {DATASET_ROOT}")

def download_file(url: str, dest_path: str):
    print(f"Downloading {url} -> {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def print_manual_dataset_instructions():
    print("""
========================================================================
 Academic Document Datasets for Extended Training:
========================================================================
To train on large-scale benchmarks:

1. DIBCO / H-DIBCO (Bleed-through, ink stains, degraded scans):
   - Web: https://tc11.cvc.uab.es/datasets/DIBCO-2019_1
   - Place image pairs (ground-truth clean & degraded scan) into:
     ai-training/dataset/raw/degraded/

2. SmartDoc Challenge (Corner detection & boundary rectification):
   - Web: http://smartdoc.univ-lr.fr/
   - Place distorted document photos & coordinate XMLs into:
     ai-training/dataset/raw/geometric/

3. Doc3D (3D Warped document meshes & illumination shadows):
   - Web: https://github.com/cvlab-stonybrook/doc3d
   - Use for training 3D unwarping and shadow compensation.

Note: In the absence of external downloads, the training pipeline
automatically generates synthetic 2D Gaussian illumination shadows,
toner speckles, and ink bleed-through on-the-fly via augmentations.py.
========================================================================
""")

if __name__ == "__main__":
    setup_directories()
    print_manual_dataset_instructions()
