"""
voxelize.py - A script for converting 3D triangle meshes to voxel representations
This script converts triangle meshes into voxel grids within a normalized coordinate space.
The voxelized models are saved as PLY files containing point coordinates.
"""

import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d


def _voxelize(file, sha256, output_dir):
    """
    Voxelize a single 3D mesh into a grid representation.
    
    Args:
        file: The source file path (not used in current implementation)
        sha256: Hash identifier for the 3D mesh
        output_dir: Directory to save the resulting voxel file
    
    Returns:
        dict: Record containing metadata about the voxelization process
    """
    # Load the mesh from the specified directory
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', sha256, 'mesh.ply'))
    
    # Clamp vertices to the normalized range [-0.5, 0.5]
    # A small epsilon is added to ensure vertices are strictly within bounds
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Create a voxel grid from the mesh with 64³ resolution (voxel size 1/64)
    # Bounds are set to the normalized space of [-0.5, 0.5]³
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    
    # Extract voxel coordinates from the grid
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    
    # Ensure all voxels are within the expected 64³ grid
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    
    # Convert grid indices back to normalized coordinate space
    vertices = (vertices + 0.5) / 64 - 0.5
    
    # Save the voxel point cloud to a PLY file
    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{sha256}.ply'), vertices)
    
    # Return metadata about the voxelization process
    return {'sha256': sha256, 'voxelized': True, 'num_voxels': len(vertices)}


if __name__ == '__main__':
    # Import dataset-specific utility module based on the first command-line argument
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    # Set up command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--num_views', type=int, default=150,
                        help='Number of views to render')
    
    # Add dataset-specific arguments
    dataset_utils.add_args(parser)
    
    # Arguments for distributed processing
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=None)
    
    # Parse command-line arguments and convert to an edict for easier access
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Create output directory for voxelized models
    os.makedirs(os.path.join(opt.output_dir, 'voxels'), exist_ok=True)

    # Load metadata from CSV file
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    # Filter the metadata based on command-line arguments
    if opt.instances is None:
        # Filter by aesthetic score if specified
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        
        # Check if rendering has been completed
        if 'rendered' not in metadata.columns:
            raise ValueError('metadata.csv does not have "rendered" column, please run "build_metadata.py" first')
        metadata = metadata[metadata['rendered'] == True]
        
        # Skip already voxelized objects
        if 'voxelized' in metadata.columns:
            metadata = metadata[metadata['voxelized'] == False]
    else:
        # Process only specified instances from a file or comma-separated list
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # Divide work for distributed processing based on rank and world_size
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # Skip objects that are already processed
    # This prevents re-processing if the script was interrupted and restarted
    for sha256 in copy.copy(metadata['sha256'].values):
        voxel_path = os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')
        if os.path.exists(voxel_path):
            # Load the existing voxel file to get the count
            pts = utils3d.io.read_ply(voxel_path)[0]
            records.append({'sha256': sha256, 'voxelized': True, 'num_voxels': len(pts)})
            # Remove from the processing list
            metadata = metadata[metadata['sha256'] != sha256]
    
    print(f'Processing {len(metadata)} objects...')

    # Process the remaining objects
    # Create a partial function with the output_dir parameter fixed
    func = partial(_voxelize, output_dir=opt.output_dir)
    
    # Process instances in parallel using dataset-specific utilities
    voxelized = dataset_utils.foreach_instance(
        metadata, opt.output_dir, func, 
        max_workers=opt.max_workers, desc='Voxelizing'
    )
    
    # Combine newly processed and previously processed records
    voxelized = pd.concat([voxelized, pd.DataFrame.from_records(records)])
    
    # Save the results in a rank-specific CSV file
    voxelized.to_csv(os.path.join(opt.output_dir, f'voxelized_{opt.rank}.csv'), index=False)
