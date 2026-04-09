"""
Custom loss functions for MI-GAN fine-tuning.

Adds identity loss and boundary loss ON TOP of MI-GAN's original adversarial
training. The adversarial loss (discriminator) keeps output sharp. The new
losses fix the boundary artifact.

MI-GAN mask convention: 1 = known pixel, 0 = hole (to be filled).
Image values: [-1, 1] range, sRGB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# New losses (our additions)
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
# Discriminator — keeps output sharp (PatchGAN with spectral normalization)
# ============================================================================

class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator. Outputs a grid of real/fake scores.
    Spectral normalization for stable GAN training.

    Input: 4 channels [mask - 0.5, composited_image]
    Output: [B, 1, H/16, W/16] grid of logits
    """

    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()

        def disc_block(in_ch, out_ch, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )

        ch = base_channels
        self.model = nn.Sequential(
            # No spectral norm on first layer (standard practice)
            nn.Conv2d(in_channels, ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            disc_block(ch, ch * 2),       # 64 -> 128
            disc_block(ch * 2, ch * 4),   # 128 -> 256
            disc_block(ch * 4, ch * 8, stride=1),  # 256 -> 512, no downsample
            nn.utils.spectral_norm(
                nn.Conv2d(ch * 8, 1, kernel_size=4, stride=1, padding=1, bias=False)
            ),
        )

    def forward(self, x):
        return self.model(x)


def discriminator_loss(discriminator, real_img, fake_img, mask):
    """
    Hinge loss for the discriminator.
    Sees composited images: real pixels outside mask + generated inside.
    """
    # Composite: real pixels in known region, generated in hole
    hole = 1.0 - mask
    fake_composite = fake_img * hole + real_img * mask

    # Discriminator input: [mask - 0.5, image]
    real_input = torch.cat([mask - 0.5, real_img], dim=1)
    fake_input = torch.cat([mask - 0.5, fake_composite.detach()], dim=1)

    real_logits = discriminator(real_input)
    fake_logits = discriminator(fake_input)

    # Hinge loss
    d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()

    return d_loss


def generator_adversarial_loss(discriminator, fake_img, real_img, mask):
    """
    Non-saturating adversarial loss for the generator.
    Pushes the generator to fool the discriminator → sharp output.
    """
    hole = 1.0 - mask
    fake_composite = fake_img * hole + real_img * mask
    fake_input = torch.cat([mask - 0.5, fake_composite], dim=1)
    fake_logits = discriminator(fake_input)

    # Non-saturating: maximize fake logits
    g_loss = F.softplus(-fake_logits).mean()

    return g_loss


def r1_penalty(discriminator, real_img, mask):
    """
    R1 gradient penalty — regularizes the discriminator.
    Standard in StyleGAN/MI-GAN training.
    """
    real_img = real_img.detach().requires_grad_(True)
    real_input = torch.cat([mask - 0.5, real_img], dim=1)
    real_logits = discriminator(real_input)

    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real_img,
        create_graph=True
    )[0]

    penalty = grad.square().sum(dim=[1, 2, 3]).mean()
    return penalty


# ============================================================================
# Perceptual loss (VGG)
# ============================================================================

class PerceptualLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()

        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23] # relu4_3
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
