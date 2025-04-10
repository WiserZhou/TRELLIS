"""
convert_to_ply.py: A utility script for converting 3D model files to PLY format.

This script automates the process of converting GLB files to PLY format:
1. Takes 3D model files as input
2. Converts them to PLY format
3. Saves the PLY files in the specified output directory
"""

import os
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import copy
import numpy as np

def rotate_mesh(mesh):
    # 正向旋转矩阵：绕X轴旋转90度
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    
    # 应用旋转矩阵到顶点
    mesh.vertices = np.dot(mesh.vertices, rotation_matrix)
    return mesh

def _convert_to_ply(file_path, sha256, output_dir):
    """
    Convert a GLB file to PLY format with normalization and centering
    
    Args:
        file_path: Path to the 3D model file
        sha256: SHA256 hash of the model file, used as identifier
        output_dir: Directory to save converted PLY files
        
    Returns:
        Dictionary with conversion results
    """
    import trimesh

    # Create output folder for the specific model
    output_folder = os.path.join(output_dir, 'renders', sha256)
    os.makedirs(output_folder, exist_ok=True)
    
    ply_path = os.path.join(output_folder, 'mesh.ply')
    
    # # Skip if already converted
    # if os.path.exists(ply_path):
    #     return {'sha256': sha256, 'converted': True}
    
    try:
        # Load the model
        mesh_or_scene = trimesh.load(os.path.expanduser(file_path))
        
        # Convert scene to a single mesh if necessary
        if isinstance(mesh_or_scene, trimesh.Scene):
            # Extract all meshes and combine them
            if len(mesh_or_scene.geometry) > 0:
                meshes = list(mesh_or_scene.geometry.values())
                mesh = trimesh.util.concatenate(meshes)
            else:
                raise ValueError(f"No geometries found in {file_path}")
        else:
            mesh = mesh_or_scene
        
        mesh = rotate_mesh(mesh)

        # Center the model - move centroid to origin
        mesh.vertices -= mesh.centroid

        # Normalize to fit in a unit cube
        scale = 1.0 / max(mesh.bounding_box.extents)
        mesh.vertices *= scale

        # Calculate the current bounding box after scaling
        scaled_bbox_min = mesh.vertices.min(axis=0)
        scaled_bbox_max = mesh.vertices.max(axis=0)

        # Calculate the margin to shrink the mesh slightly inside the unit cube
        margin = 0.05  # You can adjust this value to control how much smaller the mesh should be
        shift = (scaled_bbox_max - scaled_bbox_min) * margin

        # Apply the shift to move the mesh inward
        mesh.vertices -= (scaled_bbox_min + scaled_bbox_max) / 2  # Move the mesh to center

        # Ensure that the shift is applied uniformly on all sides, not just on the x, y, z direction.
        mesh.vertices += shift  # Apply the margin to shrink it slightly inside

        # Recalculate bounding box after shifting
        scaled_bbox_min = mesh.vertices.min(axis=0)
        scaled_bbox_max = mesh.vertices.max(axis=0)

        # Now ensure there's no contact with the top boundary by adjusting again
        center_shift = (scaled_bbox_max - scaled_bbox_min) * margin
        mesh.vertices -= center_shift  # Adjust the mesh even further to ensure it doesn't touch the top or bottom

        # Export as PLY, force overwrite if file exists
        mesh.export(ply_path, file_type='ply')
        
        return {'sha256': sha256, 'converted': True}
    
    except Exception as e:

        print(f"Error converting {file_path}: {e}")
        return {'sha256': sha256, 'converted': False}

def _convert_to_ply_blender(file_path, sha256, output_dir):
    
    import bpy

    # 清空当前场景的所有物体
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 导入 GLB 文件
    bpy.ops.import_scene.gltf(filepath=os.path.expanduser(file_path))
    # Create output folder for the specific model
    output_folder = os.path.join(output_dir, 'renders', sha256)
    os.makedirs(output_folder, exist_ok=True)
    
    ply_path = os.path.join(output_folder, 'mesh.ply')
    # 导出为 PLY 格式
    bpy.ops.export_scene.ply(filepath=ply_path)

    return {'sha256': sha256, 'converted': True}

if __name__ == '__main__':
    # Import dataset-specific utilities module based on the first argument
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the converted files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)  # Add dataset-specific arguments
    parser.add_argument('--rank', type=int, default=0,
                        help='Process rank for distributed processing')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of processes for distributed processing')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Maximum number of parallel workers')
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)

    # Load and filter metadata
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    if opt.instances is None:
        # Filter metadata based on criteria
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'converted' in metadata.columns:
            metadata = metadata[metadata['converted'] == False]
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

    # # Skip objects that have already been converted
    # for sha256 in copy.copy(metadata['sha256'].values):
    #     if os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'mesh.ply')):
    #         records.append({'sha256': sha256, 'converted': True})
    #         metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # Set up parallel conversion function with fixed parameters
    func = partial(_convert_to_ply_blender, output_dir=opt.output_dir)
    
    # Process objects in parallel
    converted = dataset_utils.foreach_instance(metadata, opt.output_dir, func, 
                                             max_workers=opt.max_workers, 
                                             desc='Converting objects to PLY')
    
    # Combine new and previously processed records
    converted = pd.concat([converted, pd.DataFrame.from_records(records)])
    
    # Save results
    converted.to_csv(os.path.join(opt.output_dir, f'converted_{opt.rank}.csv'), index=False)
