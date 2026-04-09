"""
MI-GAN fine-tuning script.

Fine-tunes the pretrained MI-GAN inference model with identity loss
and boundary loss to eliminate the visible rectangular boundary
artifact at the mask edge.

No adversarial training — keeps things simple and stable. The model
already produces sharp output from its original GAN training. We're
just teaching it: "don't create a brightness gap at the mask boundary."
"""

import os
import sys
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.model_zoo.migan_inference import Generator
from finetune.losses import FinetuneLoss
from finetune.dataset import InpaintingDataset


def measure_boundary_gap(model, images, masks, device):
    """
    Quantify the brightness gap at the mask boundary.

    Returns the average absolute brightness difference between
    pixels just inside vs just outside the mask edge in the model's output.
    This is the NUMBER we're trying to drive to zero.
    """
    model.eval()
    gaps = []

    with torch.no_grad():
        for img, mask in zip(images, masks):
            img = img.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)

            # Model input: [mask-0.5, img*mask]
            x = torch.cat([mask - 0.5, img * mask], dim=1)
            output = model(x)

            # Get boundary strips (3 pixels each side)
            dilated = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
            eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=7, stride=1, padding=3)

            # Known-side strip: pixels that are known AND near the boundary
            known_strip = eroded.logical_not() & mask.bool()
            # Hole-side strip: pixels that are hole AND near the boundary
            hole_strip = dilated.bool() & mask.logical_not()

            if known_strip.sum() > 0 and hole_strip.sum() > 0:
                # Average brightness on each side
                known_brightness = output[:, :, known_strip[0, 0]].mean().item()
                hole_brightness = output[:, :, hole_strip[0, 0]].mean().item()
                gap = abs(known_brightness - hole_brightness)
                gaps.append(gap)

    model.train()
    return np.mean(gaps) if gaps else 0.0


def save_comparison(model, img, mask, save_path, device):
    """Save a side-by-side comparison image: input | full output | composited."""
    model.eval()
    with torch.no_grad():
        img_t = img.unsqueeze(0).to(device)
        mask_t = mask.unsqueeze(0).to(device)
        x = torch.cat([mask_t - 0.5, img_t * mask_t], dim=1)
        output = model(x)

        # Full model output (what our AR pipeline uses)
        full_out = output[0]

        # Composited (standard inpainting evaluation)
        composited = img_t * mask_t + output * (1 - mask_t)
        comp_out = composited[0]

        # Masked input
        masked_in = (img_t * mask_t)[0]

    def to_pil(t):
        t = ((t + 1) / 2).clamp(0, 1)  # [-1,1] -> [0,1]
        t = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(t)

    # Side by side: masked input | full output | composited
    images = [to_pil(masked_in), to_pil(full_out), to_pil(comp_out)]
    widths = [im.width for im in images]
    total_w = sum(widths) + 20  # 10px gap between each
    h = images[0].height
    canvas = Image.new('RGB', (total_w, h), (40, 40, 40))
    x_offset = 0
    for im in images:
        canvas.paste(im, (x_offset, 0))
        x_offset += im.width + 10

    canvas.save(save_path)
    model.train()


def train(
    model,
    dataset,
    output_dir,
    num_steps=1000,
    batch_size=4,
    lr=1e-4,
    lambda_reconstruction=1.0,
    lambda_identity=10.0,
    lambda_boundary=5.0,
    lambda_perceptual=0.1,
    use_perceptual=True,
    save_every=500,
    log_every=50,
    device='cuda',
):
    """
    Fine-tune MI-GAN with identity + boundary losses.

    Args:
        model: MI-GAN inference Generator (pretrained weights loaded)
        dataset: InpaintingDataset
        output_dir: where to save checkpoints and samples
        num_steps: total fine-tuning steps
        batch_size: images per step
        lr: learning rate (lower than original since we're fine-tuning)
        lambda_*: loss weights
        save_every: save checkpoint every N steps
        log_every: print losses every N steps
        device: 'cuda' or 'cpu'
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    model = model.to(device).train()

    # Set up loss and optimizer
    criterion = FinetuneLoss(
        lambda_reconstruction=lambda_reconstruction,
        lambda_identity=lambda_identity,
        lambda_boundary=lambda_boundary,
        lambda_perceptual=lambda_perceptual,
        use_perceptual=use_perceptual,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.01)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    data_iter = iter(dataloader)

    # Keep a fixed test sample for consistent visualization
    test_img, test_mask = dataset[0]

    print(f"\n{'='*60}")
    print(f"MI-GAN Fine-Tuning")
    print(f"{'='*60}")
    print(f"Steps: {num_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Loss weights: recon={lambda_reconstruction}, identity={lambda_identity}, boundary={lambda_boundary}, perceptual={lambda_perceptual}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for step in range(1, num_steps + 1):
        # Get batch (restart iterator if exhausted)
        try:
            images, masks = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, masks = next(data_iter)

        images = images.to(device)
        masks = masks.to(device)

        # Model input: [mask-0.5, image*mask] — 4 channels
        model_input = torch.cat([masks - 0.5, images * masks], dim=1)

        # Forward pass
        output = model(model_input)

        # Compute losses
        total_loss, loss_dict = criterion(output, images, masks)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Logging
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta = (num_steps - step) / steps_per_sec if steps_per_sec > 0 else 0

            loss_str = ' | '.join(f'{k}: {v:.4f}' for k, v in loss_dict.items())
            print(f"[Step {step:>6d}/{num_steps}] {loss_str} | lr: {scheduler.get_last_lr()[0]:.2e} | ETA: {eta/60:.1f}min")

        # Save samples and checkpoints
        if step % save_every == 0 or step == 1 or step == num_steps:
            save_comparison(
                model, test_img, test_mask,
                os.path.join(output_dir, 'samples', f'step_{step:06d}.png'),
                device
            )

            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_dict': loss_dict,
            }, os.path.join(output_dir, 'checkpoints', f'step_{step:06d}.pt'))

    # Save final model
    final_path = os.path.join(output_dir, 'migan_finetuned.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")

    return model


def load_pretrained(checkpoint_path, resolution=512, device='cuda'):
    """
    Load the pretrained MI-GAN inference model.

    Args:
        checkpoint_path: path to the exported inference model state_dict (.pt)
        resolution: model resolution (512 for Places2)
        device: target device

    Returns:
        Generator model with pretrained weights
    """
    model = Generator(resolution=resolution)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"[Model] Loaded pretrained MI-GAN ({resolution}x{resolution}) — {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune MI-GAN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained inference model .pt')
    parser.add_argument('--images', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--output', type=str, default='./finetune_output', help='Output directory')
    parser.add_argument('--steps', type=int, default=50000, help='Number of fine-tuning steps')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda-identity', type=float, default=10.0, help='Identity loss weight')
    parser.add_argument('--lambda-boundary', type=float, default=5.0, help='Boundary loss weight')
    parser.add_argument('--no-perceptual', action='store_true', help='Disable perceptual loss')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_pretrained(args.checkpoint, device=device)
    dataset = InpaintingDataset(args.images)

    train(
        model=model,
        dataset=dataset,
        output_dir=args.output,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_identity=args.lambda_identity,
        lambda_boundary=args.lambda_boundary,
        use_perceptual=not args.no_perceptual,
        device=device,
    )
