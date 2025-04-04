"""
Dataset Download Script
=======================
This script is designed to download dataset objects based on metadata information.
It supports:
- Filtering objects by aesthetic score or specific instances
- Parallel processing across multiple workers
- Custom dataset-specific download logic through importable modules
"""

import os
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict

if __name__ == '__main__':
    # Import the dataset-specific utility module based on the first command line argument
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    # Set up command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    # Add dataset-specific arguments through the utility module
    dataset_utils.add_args(parser)
    # Arguments for distributed processing
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    # Parse arguments and convert to easy dictionary access
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Ensure the output directory exists
    os.makedirs(opt.output_dir, exist_ok=True)

    # Load the metadata file containing information about objects to download
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    # Filter the metadata based on provided options
    if opt.instances is None:
        # Filter by aesthetic score if specified
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        # Skip already downloaded objects
        if 'local_path' in metadata.columns:
            metadata = metadata[metadata['local_path'].isna()]
    else:
        # Filter by specific instance IDs
        if os.path.exists(opt.instances):
            # Read instance IDs from a file
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            # Parse instance IDs from command line argument
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # Handle distributed processing by dividing the work
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
                
    print(f'Processing {len(metadata)} objects...')

    # Execute the dataset-specific download function and save the results
    downloaded = dataset_utils.download(metadata, **opt)
    downloaded.to_csv(os.path.join(opt.output_dir, f'downloaded_{opt.rank}.csv'), index=False)
