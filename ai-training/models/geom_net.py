import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, s, 1, groups=in_c, bias=False)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class PagesGeomNet(nn.Module):
    """
    PagesGeomNet: Ultra-fast dual-head document geometry network.
    Head A: Predicts 4-corner Gaussian heatmaps for sub-pixel boundary quad localization.
    Head B: Predicts binary document boundary segmentation mask + central book spine crease.
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.stem = ConvBNAct(in_channels, base_channels, k=3, s=2, p=1) # 512 -> 256

        self.enc1 = DepthwiseSeparableConv(base_channels, base_channels * 2, s=2) # 256 -> 128
        self.enc2 = DepthwiseSeparableConv(base_channels * 2, base_channels * 4, s=2) # 128 -> 64
        self.enc3 = DepthwiseSeparableConv(base_channels * 4, base_channels * 8, s=2) # 64 -> 32
        self.bridge = DepthwiseSeparableConv(base_channels * 8, base_channels * 8, s=1)

        # Head A: 4-Corner Heatmaps (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
        self.up_corner1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2) # 32 -> 64
        self.up_corner2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2) # 64 -> 128
        self.corner_head = nn.Sequential(
            ConvBNAct(base_channels * 2, base_channels, 3, 1, 1),
            nn.Conv2d(base_channels, 4, 1),
            nn.Sigmoid()
        )

        # Head B: Full-Resolution Boundary & Spine Mask (512x512)
        self.up_mask1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.up_mask2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.up_mask3 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.up_mask4 = nn.ConvTranspose2d(base_channels, base_channels // 2, 2, 2)
        self.mask_head = nn.Sequential(
            nn.Conv2d(base_channels // 2, 2, 1), # Ch0: Document Polygon Mask, Ch1: Book Spine Seam
            nn.Sigmoid()
        )

    def forward(self, x):
        s0 = self.stem(x)       # [B, 32, 256, 256]
        s1 = self.enc1(s0)      # [B, 64, 128, 128]
        s2 = self.enc2(s1)      # [B, 128, 64, 64]
        s3 = self.enc3(s2)      # [B, 256, 32, 32]
        b = self.bridge(s3)

        # Corner Branch (Output resolution 128x128 for precise sub-pixel peak extraction)
        c = self.up_corner1(b) + s2
        c = self.up_corner2(c) + s1
        corners = self.corner_head(c)

        # Mask Branch (Output resolution 512x512)
        m = self.up_mask1(b) + s2
        m = self.up_mask2(m) + s1
        m = self.up_mask3(m) + s0
        m = self.up_mask4(m)
        masks = self.mask_head(m)

        return corners, masks

if __name__ == '__main__':
    model = PagesGeomNet()
    params = sum(p.numel() for p in model.parameters())
    print(f"PagesGeomNet Total Parameters: {params / 1e6:.2f}M")
    dummy = torch.randn(1, 3, 512, 512)
    corners, masks = model(dummy)
    print(f"Corners Heatmap Shape: {corners.shape} (4 points)")
    print(f"Masks Shape: {masks.shape} (Page Mask + Spine Seam)")
