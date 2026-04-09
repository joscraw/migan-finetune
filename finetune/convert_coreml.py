"""
Convert fine-tuned MI-GAN to Core ML format for iOS deployment.

Same conversion process as the original MI-GAN — produces a .mlmodelc
that's a drop-in replacement in the ExhibitXR Unity project.

Input:  [1, 4, 512, 512] float32 — "input_image"
Output: [1, 3, 512, 512] float32 — "output_image"
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.model_zoo.migan_inference import Generator


def convert_to_coreml(checkpoint_path, output_path, resolution=512):
    """
    Convert a MI-GAN state_dict to Core ML .mlpackage format.

    Args:
        checkpoint_path: path to the fine-tuned model state_dict (.pt)
        output_path: where to save the .mlpackage (e.g., 'migan_finetuned.mlpackage')
        resolution: model resolution
    """
    try:
        import coremltools as ct
    except ImportError:
        print("ERROR: coremltools not installed. Run: pip install coremltools")
        return

    print(f"Loading model from {checkpoint_path}...")
    model = Generator(resolution=resolution)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    print("Tracing model with TorchScript...")
    dummy_input = torch.ones((1, 4, resolution, resolution), dtype=torch.float32)
    traced = torch.jit.trace(model, dummy_input)

    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_image",
                shape=(1, 4, resolution, resolution),
                dtype=float,
            )
        ],
        outputs=[
            ct.TensorType(name="output_image")
        ],
        compute_precision=ct.precision.MIXED,
        minimum_deployment_target=ct.target.iOS15,
    )

    # Save as .mlpackage
    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)
    print(f"Done! Core ML model saved to {output_path}")
    print(f"\nTo compile for iOS, open in Xcode or use:")
    print(f"  xcrun coremlcompiler compile {output_path} .")
    print(f"\nThen copy the resulting .mlmodelc directory to:")
    print(f"  ExhibitXR/Assets/StreamingAssets/migan_512.mlmodelc")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MI-GAN to Core ML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to fine-tuned model .pt')
    parser.add_argument('--output', type=str, default='migan_finetuned.mlpackage', help='Output .mlpackage path')
    parser.add_argument('--resolution', type=int, default=512, help='Model resolution')
    args = parser.parse_args()

    convert_to_coreml(args.checkpoint, args.output, args.resolution)
