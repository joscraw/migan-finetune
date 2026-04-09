"""
Custom loss functions for MI-GAN fine-tuning.

Uses MI-GAN's ORIGINAL trained discriminator (loaded from their .pkl checkpoint)
for adversarial sharpness. Adds identity loss and boundary loss on top to fix
the box artifact. Their stuff untouched, our additions layered on top.

MI-GAN mask convention: 1 = known pixel, 0 = hole (to be filled).
Image values: [-1, 1] range, sRGB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Our additions (fix the boundary)
# ============================================================================

def reconstruction_loss(output, target, mask):
    """L1 loss on hole pixels only."""
    hole = 1.0 - mask
    num_hole = hole.sum().clamp(min=1.0)
    return (torch.abs(output - target) * hole).sum() / (num_hole * output.shape[1])


def identity_loss(output, target, mask):
    """L1 loss on known pixels — prevents reconstruction drift."""
    num_known = mask.sum().clamp(min=1.0)
    return (torch.abs(output - target) * mask).sum() / (num_known * output.shape[1])


def boundary_loss(output, target, mask, width=5):
    """Extra L1 penalty on pixels near the mask boundary."""
    dilated = F.max_pool2d(mask, kernel_size=2 * width + 1, stride=1, padding=width)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=2 * width + 1, stride=1, padding=width)
    boundary = (dilated - eroded).clamp(0, 1)
    num_boundary = boundary.sum().clamp(min=1.0)
    if num_boundary < 1.0:
        return torch.tensor(0.0, device=output.device)
    return (torch.abs(output - target) * boundary).sum() / (num_boundary * output.shape[1])


# ============================================================================
# MI-GAN's original adversarial losses (same math they use)
# ============================================================================

def discriminator_step(disc, real_img, fake_img, mask):
    """
    MI-GAN's discriminator loss: non-saturating logistic + R1 optional.
    The discriminator sees composited images (real outside mask + generated inside).
    Input format: [mask - 0.5, image] — 4 channels, same as their training.
    """
    hole = 1.0 - mask
    fake_composite = fake_img.detach() * hole + real_img * mask

    # Discriminator inputs
    real_input = torch.cat([mask - 0.5, real_img], dim=1)
    fake_input = torch.cat([mask - 0.5, fake_composite], dim=1)

    real_logits = disc(real_input)
    fake_logits = disc(fake_input)

    # Non-saturating logistic loss (same as MI-GAN/StyleGAN2)
    d_loss = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()

    return d_loss


def r1_penalty(disc, real_img, mask):
    """R1 gradient penalty — same as MI-GAN's Dreg."""
    real_img = real_img.detach().requires_grad_(True)
    real_input = torch.cat([mask - 0.5, real_img], dim=1)
    real_logits = disc(real_input)
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_img,
        create_graph=True
    )[0]
    return grad.square().sum(dim=[1, 2, 3]).mean()


def generator_adversarial_loss(disc, fake_img, real_img, mask):
    """
    MI-GAN's generator adversarial loss: non-saturating logistic.
    This is what keeps the output SHARP.
    """
    hole = 1.0 - mask
    fake_composite = fake_img * hole + real_img * mask
    fake_input = torch.cat([mask - 0.5, fake_composite], dim=1)
    fake_logits = disc(fake_input)
    return F.softplus(-fake_logits).mean()


# ============================================================================
# Perceptual loss (VGG)
# ============================================================================

class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        self.blocks = nn.ModuleList([
            vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]
        ])
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        x = (x + 1) / 2
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

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
