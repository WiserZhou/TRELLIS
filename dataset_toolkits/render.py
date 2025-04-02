"""
render.py: A utility script for rendering 3D models using Blender.

This script automates the process of rendering multiple views of 3D models. It:
1. Installs Blender if not already present
2. Takes 3D model files as input
3. Renders them from various camera angles using Hammersley sequence for uniform view distribution
4. Saves the rendered images and camera parameters

The script is designed to work with a dataset of 3D models and supports parallel processing.
"""

import os
import json
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from utils import sphere_hammersley_sequence


# Constants for Blender installation
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    """
    Check if Blender is installed. If not, download and install it.
    Installs necessary dependencies for Blender to run headless.
    """
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, sha256, output_dir, num_views):
    """
    Render a 3D model from multiple viewpoints.
    
    Args:
        file_path: Path to the 3D model file
        sha256: SHA256 hash of the model file, used as identifier
        output_dir: Directory to save rendered images
        num_views: Number of views to render
        
    Returns:
        Dictionary with rendering results
    """
    output_folder = os.path.join(output_dir, 'renders', sha256)
    
    # Generate camera positions using Hammersley sequence for uniform distribution on sphere
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views  # Distance from camera to object
    fov = [40 / 180 * np.pi] * num_views  # Field of view in radians
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    # Prepare arguments for Blender command
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',  # Use Cycles rendering engine for better quality
        '--save_mesh',
    ]
    # If input is a Blend file, add it as the first file parameter
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    # Execute Blender in headless mode with the rendering script
    call(args, stdout=DEVNULL, stderr=DEVNULL)
    
    # Check if rendering was successful
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}


if __name__ == '__main__':
    # Import dataset-specific utilities module based on the first argument
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    dataset_utils.add_args(parser)  # Add dataset-specific arguments
    parser.add_argument('--rank', type=int, default=0,
                        help='Process rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of processes for distributed processing')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Maximum number of parallel workers')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Create output directory for rendered images
    os.makedirs(os.path.join(opt.output_dir, 'renders'), exist_ok=True)
    
    # Install Blender if needed
    print('Checking blender...', flush=True)
    _install_blender()

    # Load and filter metadata
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    if opt.instances is None:
        # Filter metadata based on criteria
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'rendered' in metadata.columns:
            metadata = metadata[metadata['rendered'] == False]
    else:
        # Filter metadata based on specified instances
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # Distribute work for parallel processing
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # Skip objects that have already been rendered
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # Set up parallel rendering function with fixed parameters
    func = partial(_render, output_dir=opt.output_dir, num_views=opt.num_views)
    
    # Process objects in parallel using the dataset utility function
    # This distributes rendering tasks across multiple workers for better performance
    # Each object in metadata will be processed by the _render function
    rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, 
                                             max_workers=opt.max_workers, 
                                             desc='Rendering objects')
    
    # Combine newly rendered objects with previously processed records
    # This ensures we keep track of all objects, including those already rendered
    rendered = pd.concat([rendered, pd.DataFrame.from_records(records)])
    
    # Save the rendering results to a CSV file with rank identifier
    # This allows for tracking which objects were processed by this worker
    # in a distributed rendering environment
    rendered.to_csv(os.path.join(opt.output_dir, f'rendered_{opt.rank}.csv'), index=False)
