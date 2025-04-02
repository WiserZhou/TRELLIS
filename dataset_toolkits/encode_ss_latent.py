# Voxel-to-Latent Encoder for TRELLIS
"""
This script encodes 3D voxelized shape data into latent space representations using a neural network encoder.
It processes a batch of 3D shapes (identified by SHA256 hashes), encodes their voxel data into latent vectors,
and stores these representations for further use in generative modeling, shape analysis, or other 3D deep learning tasks.

The script supports distributed processing, aesthetic score filtering, and can work with both pretrained
and custom encoder models. It handles processing in parallel to maximize throughput and tracks
the processing status in CSV format.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory to path for imports
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis.models as models


torch.set_grad_enabled(False)  # Disable gradient computation since we're only doing inference


def get_voxels(instance):
    """
    Load voxel data for a given instance and convert to sparse tensor format.
    
    Args:
        instance: Instance identifier (sha256 hash)
    
    Returns:
        Sparse tensor representation of voxel data with shape [1, resolution, resolution, resolution]
    """
    # Load point cloud representation from PLY file
    position = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{instance}.ply'))[0]
    # Convert positions to integer coordinates in the voxel grid
    coords = ((torch.tensor(position) + 0.5) * opt.resolution).int().contiguous()
    # Create empty sparse tensor with specified resolution
    ss = torch.zeros(1, opt.resolution, opt.resolution, opt.resolution, dtype=torch.long)
    # Set occupied voxels to 1 (creating a binary voxel grid)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
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
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0,
                        help='Current process rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of processes for distributed processing')
    opt = parser.parse_args()
    opt = edict(vars(opt))  # Convert to EasyDict for attribute-style access

    # Load the appropriate encoder model
    if opt.enc_model is None:
        # Use pretrained model if no specific model is specified
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()  # Load pretrained encoder and move to GPU
    else:
        # Load custom model from config and checkpoint
        latent_name = f'{opt.enc_model}_{opt.ckpt}'
        # Load model configuration from JSON file
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        # Initialize encoder model with configuration parameters
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        # Load model weights from checkpoint
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()  # Set model to evaluation mode
        print(f'Loaded model from {ckpt_path}')
    
    # Create output directory for latent representations
    os.makedirs(os.path.join(opt.output_dir, 'ss_latents', latent_name), exist_ok=True)

    # Load and filter metadata
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
        
    # Filter metadata based on command line arguments
    if opt.instances is not None:
        # If specific instances are provided, use only those
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        # Otherwise, filter based on aesthetic score and voxel availability
        if opt.filter_low_aesthetic_score is not None:
            # Apply aesthetic quality threshold filtering
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        # Only process objects that have been successfully voxelized
        metadata = metadata[metadata['voxelized'] == True]
        # Skip already processed objects
        if f'ss_latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'ss_latent_{latent_name}'] == False]

    # Divide work for distributed processing
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]  # Select subset of data for this process to handle
    records = []  # List to store processing results
    
    # Filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        # Check if latent representation already exists
        if os.path.exists(os.path.join(opt.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
            sha256s.remove(sha256)  # Remove from processing list

    # Encode latent representations using parallel processing
    load_queue = Queue(maxsize=4)  # Queue for communication between loaders and encoders
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
            
            def loader(sha256):
                """Worker function to load voxel data and put in queue"""
                try:
                    ss = get_voxels(sha256)[None].float()  # Load voxels and add batch dimension
                    load_queue.put((sha256, ss))
                except Exception as e:
                    print(f"Error loading features for {sha256}: {e}")
                    
            # Start loading objects in parallel
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                """Worker function to save encoded latent vectors"""
                save_path = os.path.join(opt.output_dir, 'ss_latents', latent_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)  # Save compressed numpy array
                records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
            
            # Process each object from the queue
            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, ss = load_queue.get()  # Get voxel data from queue
                ss = ss.cuda().float()  # Move to GPU and ensure correct data type
                latent = encoder(ss, sample_posterior=False)  # Encode to latent representation
                assert torch.isfinite(latent).all(), "Non-finite latent"  # Check for numerical instability
                pack = {
                    'mean': latent[0].cpu().numpy(),  # Save mean latent vector, moving back to CPU
                }
                # Schedule saving in parallel
                saver_executor.submit(saver, sha256, pack)
                
            # Ensure all saving tasks complete
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    # Save processing records to CSV
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'ss_latent_{latent_name}_{opt.rank}.csv'), index=False)
