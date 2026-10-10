import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
import glob
from PIL import Image
import torchvision.transforms as T

from models.clean_net import PagesCleanNet
from dataset.augmentations import DocumentDegradationPipeline

class SyntheticDocumentDataset(Dataset):
    """
    Loads clean scan images and generates dynamic real-time degraded pairs
    (shadows, toner noise, bleeds) for supervised training.
    """
    def __init__(self, image_paths, patch_size=512):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.degradation = DocumentDegradationPipeline()
        self.to_tensor = T.Compose([
            T.Resize((patch_size, patch_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return max(len(self.image_paths), 100) # Ensure minimum epoch length

    def __getitem__(self, idx):
        if len(self.image_paths) > 0:
            path = self.image_paths[idx % len(self.image_paths)]
            try:
                img = Image.open(path).convert('RGB')
                clean = self.to_tensor(img)
            except Exception:
                clean = torch.ones(3, self.patch_size, self.patch_size) * 0.95
        else:
            clean = torch.ones(3, self.patch_size, self.patch_size) * 0.95

        # Generate on-the-fly degraded paired input
        degraded = self.degradation.apply_pipeline(clean.unsqueeze(0)).squeeze(0)
        return degraded, clean

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

def train(epochs=10, batch_size=4, lr=2e-4):
    device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')
    print(f"================================================================")
    print(f" Starting Pages AI Restoration Training on: {torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'CPU'}")
    print(f" Target Hardware: Intel Arc A770 (16 GB VRAM) via PyTorch XPU")
    print(f" Precision: bfloat16 Mixed Precision via Intel XMX Engines")
    print(f"================================================================")

    # Collect training seeds from workspace scans/ directory
    workspace_scans = glob.glob('/mnt/c/Users/LEADER/Desktop/Scantailor-Next/scans/*.*')
    dataset = SyntheticDocumentDataset(workspace_scans, patch_size=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = PagesCleanNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = CharbonnierLoss()

    best_loss = float('inf')
    os.makedirs('/mnt/c/Users/LEADER/Desktop/Scantailor-Next/ai-training/checkpoints', exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.perf_counter()

        for batch_idx, (dirty, clean) in enumerate(loader):
            dirty = dirty.to(device)
            clean = clean.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type='xpu' if torch.xpu.is_available() else 'cpu', dtype=torch.bfloat16):
                restored = model(dirty)
                loss = criterion(restored, clean)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        epoch_time = time.perf_counter() - start_time
        avg_loss = total_loss / len(loader)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.5f} - LR: {scheduler.get_last_lr()[0]:.2e} - Time: {epoch_time:.2f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = '/mnt/c/Users/LEADER/Desktop/Scantailor-Next/ai-training/checkpoints/pages_clean_best.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Best model checkpoint saved to: {ckpt_path}")

    print("\nTraining completed successfully! Ready for ONNX export.")

if __name__ == '__main__':
    train(epochs=5, batch_size=4, lr=2e-4)
