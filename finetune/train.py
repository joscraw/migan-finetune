"""
MI-GAN fine-tuning script.

Uses MI-GAN's ORIGINAL trained discriminator (from their .pkl checkpoint)
for adversarial sharpness. Adds identity + boundary losses on top.
Their training setup untouched, our fixes layered on.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.model_zoo.migan_inference import Generator
from finetune.losses import (
    reconstruction_loss, identity_loss, boundary_loss,
    discriminator_step, generator_adversarial_loss,
    r1_penalty, PerceptualLoss,
)
from finetune.dataset import InpaintingDataset


def measure_boundary_gap(model, images, masks, device):
    """Quantify the brightness gap at the mask boundary."""
    model.eval()
    gaps = []
    with torch.no_grad():
        for img, mask in zip(images, masks):
            img = img.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            x = torch.cat([mask - 0.5, img * mask], dim=1)
            output = model(x)
            dilated = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
            eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=7, stride=1, padding=3)
            known_strip = eroded.logical_not() & mask.bool()
            hole_strip = dilated.bool() & mask.logical_not()
            if known_strip.sum() > 0 and hole_strip.sum() > 0:
                known_brightness = output[:, :, known_strip[0, 0]].mean().item()
                hole_brightness = output[:, :, hole_strip[0, 0]].mean().item()
                gaps.append(abs(known_brightness - hole_brightness))
    model.train()
    return np.mean(gaps) if gaps else 0.0


def save_comparison(model, img, mask, save_path, device):
    """Save side-by-side: masked input | full output | composited."""
    model.eval()
    with torch.no_grad():
        img_t = img.unsqueeze(0).to(device)
        mask_t = mask.unsqueeze(0).to(device)
        x = torch.cat([mask_t - 0.5, img_t * mask_t], dim=1)
        output = model(x)
        full_out = output[0]
        composited = img_t * mask_t + output * (1 - mask_t)
        comp_out = composited[0]
        masked_in = (img_t * mask_t)[0]

    def to_pil(t):
        t = ((t + 1) / 2).clamp(0, 1)
        t = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(t)

    images_list = [to_pil(masked_in), to_pil(full_out), to_pil(comp_out)]
    total_w = sum(im.width for im in images_list) + 20
    h = images_list[0].height
    canvas = Image.new('RGB', (total_w, h), (40, 40, 40))
    x_offset = 0
    for im in images_list:
        canvas.paste(im, (x_offset, 0))
        x_offset += im.width + 10
    canvas.save(save_path)
    model.train()


def train(
    model,
    disc,
    dataset,
    output_dir,
    num_steps=1000,
    batch_size=4,
    g_lr=2e-4,
    d_lr=2e-4,
    lambda_reconstruction=1.0,
    lambda_identity=10.0,
    lambda_boundary=5.0,
    lambda_adversarial=1.0,
    lambda_perceptual=0.1,
    r1_gamma=10.0,
    d_reg_every=16,
    use_perceptual=True,
    save_every=500,
    log_every=50,
    device='cuda',
):
    """
    Fine-tune MI-GAN with their trained discriminator + our boundary fixes.

    Args:
        model: MI-GAN inference Generator (pretrained)
        disc: MI-GAN's trained Discriminator (from .pkl checkpoint)
        dataset: InpaintingDataset
        ...
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    model = model.to(device).train()
    disc = disc.to(device).train()

    g_param_count = sum(p.numel() for p in model.parameters())
    d_param_count = sum(p.numel() for p in disc.parameters())
    print(f"[Generator] {g_param_count:,} parameters")
    print(f"[Discriminator] MI-GAN's trained D — {d_param_count:,} parameters")

    # Perceptual loss
    percep_loss = PerceptualLoss(device=device) if use_perceptual else None

    # Optimizers — same Adam config as MI-GAN original: betas=(0, 0.99)
    g_optimizer = torch.optim.Adam(model.parameters(), lr=g_lr, betas=(0.0, 0.99))
    d_optimizer = torch.optim.Adam(disc.parameters(), lr=d_lr, betas=(0.0, 0.99))

    g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        g_optimizer, T_max=num_steps, eta_min=g_lr * 0.01
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    data_iter = iter(dataloader)
    test_img, test_mask = dataset[0]

    print(f"\n{'='*60}")
    print(f"MI-GAN Fine-Tuning (with original trained discriminator)")
    print(f"{'='*60}")
    print(f"Steps: {num_steps} | Batch: {batch_size}")
    print(f"G lr: {g_lr} | D lr: {d_lr} | Adam betas: (0, 0.99)")
    print(f"Losses: recon={lambda_reconstruction}, identity={lambda_identity}, "
          f"boundary={lambda_boundary}, adversarial={lambda_adversarial}, "
          f"perceptual={lambda_perceptual}")
    print(f"R1: gamma={r1_gamma} every {d_reg_every} steps")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for step in range(1, num_steps + 1):
        try:
            images, masks = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, masks = next(data_iter)

        images = images.to(device)
        masks = masks.to(device)
        model_input = torch.cat([masks - 0.5, images * masks], dim=1)

        # ============================================================
        # DISCRIMINATOR STEP (their loss — keeps it calibrated)
        # ============================================================
        with torch.no_grad():
            fake_output = model(model_input)

        d_loss = discriminator_step(disc, images, fake_output, masks)

        # R1 regularization every N steps (same as their training)
        d_reg_val = 0.0
        if step % d_reg_every == 0:
            d_reg = r1_penalty(disc, images, masks)
            d_loss = d_loss + d_reg * (r1_gamma * 0.5) * d_reg_every
            d_reg_val = d_reg.item()

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ============================================================
        # GENERATOR STEP (their adversarial + our boundary fixes)
        # ============================================================
        output = model(model_input)

        # Their loss: adversarial (keeps output sharp)
        l_adv = generator_adversarial_loss(disc, output, images, masks)

        # Our losses: fix the boundary
        l_recon = reconstruction_loss(output, images, masks)
        l_ident = identity_loss(output, images, masks)
        l_bound = boundary_loss(output, images, masks, width=5)

        g_loss = (
            lambda_adversarial * l_adv +
            lambda_reconstruction * l_recon +
            lambda_identity * l_ident +
            lambda_boundary * l_bound
        )

        l_percep_val = 0.0
        if percep_loss is not None:
            l_percep = percep_loss(output, images)
            g_loss = g_loss + lambda_perceptual * l_percep
            l_percep_val = l_percep.item()

        g_optimizer.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        g_optimizer.step()
        g_scheduler.step()

        # ============================================================
        # LOGGING
        # ============================================================
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            eta = (num_steps - step) / (step / elapsed) if step > 0 else 0
            print(f"[{step:>6d}/{num_steps}] "
                  f"adv:{l_adv.item():.3f} "
                  f"recon:{l_recon.item():.3f} "
                  f"ident:{l_ident.item():.4f} "
                  f"bound:{l_bound.item():.4f} "
                  f"percep:{l_percep_val:.3f} "
                  f"d:{d_loss.item():.3f} "
                  f"r1:{d_reg_val:.3f} "
                  f"lr:{g_scheduler.get_last_lr()[0]:.1e} "
                  f"ETA:{eta/60:.1f}m")

        if step % save_every == 0 or step == 1 or step == num_steps:
            save_comparison(
                model, test_img, test_mask,
                os.path.join(output_dir, 'samples', f'step_{step:06d}.png'),
                device
            )
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'disc_state_dict': disc.state_dict(),
            }, os.path.join(output_dir, 'checkpoints', f'step_{step:06d}.pt'))

    final_path = os.path.join(output_dir, 'migan_finetuned.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")
    return model


def load_pretrained(checkpoint_path, resolution=512, device='cuda'):
    """Load the pretrained MI-GAN inference model."""
    model = Generator(resolution=resolution)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"[Model] Loaded pretrained MI-GAN ({resolution}x{resolution}) — "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")
    return model
