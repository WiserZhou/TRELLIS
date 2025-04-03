# Voxel-to-Latent Encoder for TRELLIS (Single File Version)
"""
This script encodes a single 3D voxelized shape into latent space representation.
It processes a single 3D shape file, encodes its voxel data into a latent vector,
and stores the representation for further use.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import torch
import numpy as np
import utils3d
from easydict import EasyDict as edict
import trellis.models as models

torch.set_grad_enabled(False)  # Disable gradient computation

def get_voxels(input_path, resolution):
    """
    Load voxel data from a file and convert to sparse tensor format.
    
    Args:
        input_path: Path to the input PLY file
        resolution: Voxel resolution
    
    Returns:
        Sparse tensor representation of voxel data
    """
    # Load point cloud representation from PLY file
    position = utils3d.io.read_ply(input_path)[0]
    # Convert positions to integer coordinates in the voxel grid
    coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
    # Create empty sparse tensor with specified resolution
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    # Set occupied voxels to 1 (creating a binary voxel grid)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="/mnt/pfs/users/yangyunhan/yufan/data/voxel/ab9804f981184f8db6f1f814c2b8c169.ply",
                        help='Path to input PLY voxel file')
    parser.add_argument('--output_dir', type=str, default="/mnt/pfs/users/yangyunhan/yufan/data/ss_latents",
                        help='Path to output latent file (NPZ format)')
    parser.add_argument('--enc_pretrained', type=str, default='JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution')
    opt = parser.parse_args()
    opt = edict(vars(opt))  # Convert to EasyDict

    # Load the appropriate encoder model
    if opt.enc_model is None:
        # Use pretrained model if no specific model is specified
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
        print(f'Loaded pretrained model from {opt.enc_pretrained}')
    else:
        # Load custom model from config and checkpoint
        # Load model configuration from JSON file
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        # Initialize encoder model with configuration parameters
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        # Load model weights from checkpoint
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()  # Set model to evaluation mode
        print(f'Loaded model from {ckpt_path}')
    
    # Create output directory if needed
    os.makedirs(opt.output_dir, exist_ok=True)

    # Process the single file
    try:
        print(f"Processing file: {opt.input_file}")
        # Load and prepare voxel data
        ss = get_voxels(opt.input_file, opt.resolution)[None].float().cuda()
        
        # Encode to latent representation
        latent = encoder(ss, sample_posterior=False)
        assert torch.isfinite(latent).all(), "Non-finite latent"
        
        # Save the latent representation
        result = {'mean': latent[0].cpu().numpy()}
        np.savez_compressed(opt.output_dir+"/"+os.path.splitext(os.path.basename(opt.input_file))[0], **result)
        print(f"Latent representation saved to {opt.output_dir}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
