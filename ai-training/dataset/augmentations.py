import torch
import torch.nn.functional as F
import numpy as np

class DocumentDegradationPipeline:
    """
    On-The-Fly Document Degradation Engine.
    Synthesizes real-world scanning defects (2D illumination shadows, toner speckles, ink bleed)
    onto clean ground-truth book/document pages during training.
    """
    def __init__(self, device='cpu'):
        self.device = device

    def generate_gaussian_shadow(self, h, w, min_dark=0.25, max_dark=0.75):
        """Generates realistic 2D illumination falloff / hand & phone camera shadow."""
        center_x = np.random.uniform(0.1, 0.9) * w
        center_y = np.random.uniform(0.1, 0.9) * h
        sigma_x = np.random.uniform(0.3, 1.2) * w
        sigma_y = np.random.uniform(0.3, 1.2) * h
        angle = np.random.uniform(0, np.pi)

        y_coords = np.arange(h).reshape(-1, 1)
        x_coords = np.arange(w).reshape(1, -1)

        # Rotate coordinates
        x_rot = (x_coords - center_x) * np.cos(angle) - (y_coords - center_y) * np.sin(angle)
        y_rot = (x_coords - center_x) * np.sin(angle) + (y_coords - center_y) * np.cos(angle)

        shadow = np.exp(-0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2))
        darkness = np.random.uniform(min_dark, max_dark)
        shadow_map = 1.0 - shadow * (1.0 - darkness)
        
        # Add linear gradient for ambient falloff
        grad = np.linspace(np.random.uniform(0.7, 1.0), np.random.uniform(0.7, 1.0), w)
        shadow_map = shadow_map * grad.reshape(1, -1)
        return torch.from_numpy(shadow_map.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    def add_toner_speckles(self, img_tensor, density=0.0008, max_size=3):
        """Adds isolated black dust and toner speckles commonly found on glass scanners."""
        b, c, h, w = img_tensor.shape
        speckle_mask = torch.rand(b, 1, h, w, device=img_tensor.device) < density
        if max_size > 1:
            # Dilate randomly to create 2-3px speckles
            speckle_mask = F.max_pool2d(speckle_mask.float(), kernel_size=max_size, stride=1, padding=max_size // 2).bool()
        noisy = img_tensor.clone()
        noisy[speckle_mask.expand_as(noisy)] = torch.rand_like(noisy[speckle_mask.expand_as(noisy)]) * 0.15
        return noisy

    def add_ink_bleed(self, clean_img, bleed_source, alpha_range=(0.04, 0.18)):
        """Blends faint reversed text from the other side of a book page."""
        alpha = np.random.uniform(*alpha_range)
        flipped_bleed = torch.flip(bleed_source, dims=[-1])
        # Smooth the bleed-through
        kernel_size = 5
        bleed_blur = F.avg_pool2d(flipped_bleed, kernel_size, stride=1, padding=kernel_size // 2)
        degraded = clean_img * (1.0 - alpha * (1.0 - bleed_blur))
        return degraded

    def apply_pipeline(self, clean_tensor, bleed_source=None):
        """Full pipeline: Clean Ground Truth -> Degraded Training Input"""
        b, c, h, w = clean_tensor.shape
        degraded = clean_tensor.clone()

        # 1. Apply Illumination Shadow
        shadow = self.generate_gaussian_shadow(h, w).to(clean_tensor.device)
        degraded = degraded * shadow.expand_as(degraded)

        # 2. Add Bleed-Through if source available
        if bleed_source is not None:
            degraded = self.add_ink_bleed(degraded, bleed_source)

        # 3. Add Toner Speckles
        degraded = self.add_toner_speckles(degraded)

        return torch.clamp(degraded, 0.0, 1.0)

if __name__ == '__main__':
    pipeline = DocumentDegradationPipeline()
    clean = torch.ones(1, 3, 512, 512)
    degraded = pipeline.apply_pipeline(clean)
    print(f"Degradation Pipeline Tested Successfully! Min pixel: {degraded.min():.2f}, Max: {degraded.max():.2f}")
