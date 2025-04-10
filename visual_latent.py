import os
import sys
import json
import argparse
from easydict import EasyDict as edict

import numpy as np

from trellis.datasets.sparse_structure_latent import ImageConditionedSparseStructureLatent

def visual_ss_latent(cfg):
    """
    Visualize sparse structure latent samples and save images.
    
    Args:
        cfg (edict): Configuration containing dataset parameters and paths
    """
    try:
        # Check if latent path exists
        if not os.path.exists(cfg.latent_path):
            raise FileNotFoundError(f"Latent file not found: {cfg.latent_path}")
            
        print(f"Loading latent from: {cfg.latent_path}")
        with np.load(cfg.latent_path) as data:
            print("Available keys in the NPZ file:", data.files)
            ss_latent = data

        # 在调用visualize_sample之前，创建一个包含正确键名的新字典
        latent_dict = {}
        if 'x_0' not in ss_latent.files and 'mean' in ss_latent.files:
            # 如果没有x_0但有mean，把mean的数据赋值给x_0
            latent_dict['x_0'] = ss_latent['mean']
        else:
            # 保持原有键名
            for key in ss_latent.files:
                latent_dict[key] = ss_latent[key]

        # Setup output directories for visualization
        os.makedirs(cfg.visualize_sample_dir, exist_ok=True)

        # Initialize dataset for visualization
        ss_dataset = ImageConditionedSparseStructureLatent(cfg.data_dir, **cfg.dataset.args)
        stack_images = ss_dataset.visualize_sample(latent_dict)

        # Save the visualized images
        for i, img in enumerate(stack_images):
            img_path = os.path.join(cfg.visualize_sample_dir, f'sample_{i}.png')
            print(f"Saving image to: {img_path}")
            img.save(img_path)
        
        print(f"Successfully saved {len(stack_images)} visualization images")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize sparse structure latent samples')
    # Config file
    parser.add_argument('--config', type=str, required=True, help='Experiment config file')
    # Input/output and checkpoint options
    parser.add_argument('--output_dir', type=str, default="./visual", help='Output directory')
    parser.add_argument('--load_dir', type=str, default='', help='Load directory, default to output_dir')
    parser.add_argument('--ckpt', type=str, default='latest', help='Checkpoint step to resume training, default to latest')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Data directory')
    parser.add_argument('--latent_path', type=str, 
        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/ss_latents/ss_enc_conv3d_16l8_fp16/a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee.npz", 
        help='Path to the latent data file')
    parser.add_argument('--visualize_sample_dir', type=str, help='Directory to save visualization samples')
    
    # Process arguments
    opt = parser.parse_args()
    opt.load_dir = opt.load_dir if opt.load_dir != '' else opt.output_dir

    try:
        # Load configuration from JSON file
        with open(opt.config, 'r') as f:
            config = json.load(f)
        
        # Combine command-line arguments and JSON config
        cfg = edict()
        cfg.update(config)  # First add config file settings
        cfg.update(opt.__dict__)  # Then override with command-line args
        
        # Set default visualization directory if not specified
        if not hasattr(cfg, 'visualize_sample_dir') or not cfg.visualize_sample_dir:
            cfg.visualize_sample_dir = os.path.join(opt.output_dir, 'visualizations')
        
        print('\n\nConfiguration:')
        print('=' * 80)
        print(json.dumps(cfg.__dict__, indent=4))

        visual_ss_latent(cfg)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)