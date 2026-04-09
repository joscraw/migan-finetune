"""
Custom loss functions for MI-GAN fine-tuning.

Adds identity loss and boundary loss to fix the visible rectangular
boundary artifact where generated pixels meet reconstructed pixels.

MI-GAN mask convention: 1 = known pixel, 0 = hole (to be filled).
Image values: [-1, 1] range, sRGB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def reconstruction_loss(output, target, mask):
    """
    L1 loss on hole pixels only — forces the model to generate
    accurate content inside the masked region.

    Args:
        output: model output [B, 3, H, W] in [-1, 1]
        target: ground truth [B, 3, H, W] in [-1, 1]
        mask: binary mask [B, 1, H, W], 1=known, 0=hole
    """
    hole = 1.0 - mask
    num_hole = hole.sum().clamp(min=1.0)
    return (torch.abs(output - target) * hole).sum() / (num_hole * output.shape[1])


def identity_loss(output, target, mask):
    """
    L1 loss on known pixels — forces the model to perfectly
    reconstruct pixels it shouldn't have touched.

    This directly addresses the reconstruction drift that causes
    the brightness gap at the mask boundary.

    Args:
        output: model output [B, 3, H, W]
        target: ground truth [B, 3, H, W]
        mask: binary mask [B, 1, H, W], 1=known, 0=hole
    """
    num_known = mask.sum().clamp(min=1.0)
    return (torch.abs(output - target) * mask).sum() / (num_known * output.shape[1])


def boundary_loss(output, target, mask, width=5):
    """
    Extra L1 penalty on pixels near the mask boundary.

    Identifies a thin strip of pixels on both sides of the mask edge
    and applies extra reconstruction penalty there. This forces the
    model to be especially accurate at the seam.

    Args:
        output: model output [B, 3, H, W]
        target: ground truth [B, 3, H, W]
        mask: binary mask [B, 1, H, W], 1=known, 0=hole
        width: how many pixels on each side of the boundary to penalize
    """
    # Dilate: expand known region (1s) outward into hole region
    dilated = F.max_pool2d(mask, kernel_size=2 * width + 1, stride=1, padding=width)

    # Erode: shrink known region (1s) inward
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=2 * width + 1, stride=1, padding=width)

    # Boundary strip = the ring of pixels within `width` of the mask edge
    boundary = (dilated - eroded).clamp(0, 1)

    num_boundary = boundary.sum().clamp(min=1.0)
    if num_boundary < 1.0:
        return torch.tensor(0.0, device=output.device)

    return (torch.abs(output - target) * boundary).sum() / (num_boundary * output.shape[1])


class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss — compares high-level features rather
    than raw pixels. Preserves structural quality and prevents
    the output from becoming blurry during fine-tuning.

    Uses VGG16 features at multiple scales.
    """

    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()

        # Extract features at ReLU layers after conv blocks 1-4
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23] # relu4_3
        ])

        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization (VGG expects this)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """Convert from [-1,1] (MI-GAN range) to ImageNet-normalized."""
        x = (x + 1) / 2  # [-1,1] -> [0,1]
        return (x - self.mean) / self.std

    def forward(self, output, target):
        output = self.normalize(output)
        target = self.normalize(target)

        loss = 0.0
        x, y = output, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)

        return loss


class FinetuneLoss(nn.Module):
    """
    Combined loss for MI-GAN fine-tuning.

    Combines:
    - Reconstruction L1 (hole pixels): standard inpainting quality
    - Identity L1 (known pixels): prevents reconstruction drift
    - Boundary L1 (edge pixels): eliminates the visible seam
    - Perceptual (VGG features): preserves structural quality

    Default weights are a starting point — Cell 5 in the notebook
    tests with a quick 1K-step run so you can tune before committing.
    """

    def __init__(
        self,
        lambda_reconstruction=1.0,
        lambda_identity=10.0,
        lambda_boundary=5.0,
        lambda_perceptual=0.1,
        boundary_width=5,
        use_perceptual=True,
        device='cuda'
    ):
        super().__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_identity = lambda_identity
        self.lambda_boundary = lambda_boundary
        self.lambda_perceptual = lambda_perceptual
        self.boundary_width = boundary_width

        self.perceptual = None
        if use_perceptual:
            self.perceptual = PerceptualLoss(device=device)

    def forward(self, output, target, mask):
        """
        Args:
            output: model output [B, 3, H, W] in [-1, 1]
            target: ground truth image [B, 3, H, W] in [-1, 1]
            mask: binary mask [B, 1, H, W], 1=known, 0=hole

        Returns:
            total_loss, dict of individual loss values for logging
        """
        l_recon = reconstruction_loss(output, target, mask)
        l_ident = identity_loss(output, target, mask)
        l_bound = boundary_loss(output, target, mask, width=self.boundary_width)

        total = (
            self.lambda_reconstruction * l_recon +
            self.lambda_identity * l_ident +
            self.lambda_boundary * l_bound
        )

        losses = {
            'reconstruction': l_recon.item(),
            'identity': l_ident.item(),
            'boundary': l_bound.item(),
        }

        if self.perceptual is not None:
            l_percep = self.perceptual(output, target)
            total = total + self.lambda_perceptual * l_percep
            losses['perceptual'] = l_percep.item()

        losses['total'] = total.item()
        return total, losses
