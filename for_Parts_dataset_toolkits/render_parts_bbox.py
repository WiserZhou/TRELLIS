"""
File: render_parts_cond.py
Purpose: This script renders individual parts of 3D models for conditioning in machine learning models.
         It processes 3D objects, separates them into parts, and renders each part independently from
         multiple camera viewpoints. The renders are saved as image files with corresponding camera
         transformation data in a JSON file, which can be used for training or conditioning generative models.

The script utilizes the vrenderer library to handle the rendering process and works with Blender's
Python API for 3D object manipulation. It can process multiple models in parallel and supports
filtering by aesthetic scores.

The main workflow:
1. Load 3D models from specified paths
2. Initialize rendering settings with normalized scaling
3. Configure camera positions at different viewpoints around the model
4. Sort object parts by their vertical position
5. Render each part separately
6. Save rendering results with camera transformation data
"""

import os
import json
import copy
import sys
sys.path.append(os.path.abspath("."))
from argparse import ArgumentParser

import importlib
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np

def _render_cond(file_path, sha256, output_dir):
    """
    Extract and save bounding box information for a 3D model and its parts.
    
    Args:
        file_path: Path to the 3D model file
        sha256: SHA256 hash of the model file, used as identifier
        output_dir: Directory to save bounding box information
        num_views: Not used in this version as we're only saving bounding boxes

    Returns:
        Dictionary with processing results containing status information
    """

    from vrenderer.render import initialize
    from vrenderer.spec import InitializationSettings
    import bpy
    from mathutils import Vector

    # Create output directory using the model's hash as identifier
    output_folder = os.path.join(output_dir, 'renders_cond', sha256)
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the full path to the model file
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(parent_dir, file_path)

    print(file_path)
    
    # Initialize model settings with normalized scale
    initialization_settings = InitializationSettings(
        file_path=file_path,
        merge_vertices=True, 
        normalizing_scale=0.5
    )
    initialization_output = initialize(initialization_settings)
    
    # Get all objects in the scene
    objects = bpy.data.objects
    
    # Information to save
    bbox_info = {
        "whole_object": {},
        "parts": []
    }
    
    # Get bounding box for the whole object
    min_coords = Vector((float('inf'), float('inf'), float('inf')))
    max_coords = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    for obj in objects:
        if obj.type == 'MESH':
            # Get world matrix - add type check to prevent errors
            matrix_world = obj.matrix_world
            if not hasattr(matrix_world, "__matmul__"):
                print(f"Warning: matrix_world for object {obj.name} doesn't support matrix multiplication")
                continue
            
            # Calculate global bounding box
            for vertex in obj.data.vertices:
                try:
                    global_vertex = matrix_world @ vertex.co
                    min_coords.x = min(min_coords.x, global_vertex.x)
                    min_coords.y = min(min_coords.y, global_vertex.y)
                    min_coords.z = min(min_coords.z, global_vertex.z)
                    max_coords.x = max(max_coords.x, global_vertex.x)
                    max_coords.y = max(max_coords.y, global_vertex.y)
                    max_coords.z = max(max_coords.z, global_vertex.z)
                except TypeError as e:
                    print(f"Error processing vertex in {obj.name}: {e}")
                    continue
    
    # Save whole object bounding box
    bbox_info["whole_object"] = {
        "min": [min_coords.x, min_coords.y, min_coords.z],
        "max": [max_coords.x, max_coords.y, max_coords.z],
        "center": [(min_coords.x + max_coords.x) / 2,
                    (min_coords.y + max_coords.y) / 2,
                    (min_coords.z + max_coords.z) / 2],
        "size": [max_coords.x - min_coords.x,
                max_coords.y - min_coords.y,
                max_coords.z - min_coords.z]
    }
    
    # Calculate bounding box for each part
    for obj in objects:
        if obj.type == 'MESH':
            try:
                # Get world matrix
                matrix_world = obj.matrix_world
                
                # Calculate part bounding box
                part_min = Vector((float('inf'), float('inf'), float('inf')))
                part_max = Vector((float('-inf'), float('-inf'), float('-inf')))
                
                for vertex in obj.data.vertices:
                    global_vertex = matrix_world @ vertex.co
                    part_min.x = min(part_min.x, global_vertex.x)
                    part_min.y = min(part_min.y, global_vertex.y)
                    part_min.z = min(part_min.z, global_vertex.z)
                    part_max.x = max(part_max.x, global_vertex.x)
                    part_max.y = max(part_max.y, global_vertex.y)
                    part_max.z = max(part_max.z, global_vertex.z)
                
                part_info = {
                    "name": obj.name,
                    "min": [part_min.x, part_min.y, part_min.z],
                    "max": [part_max.x, part_max.y, part_max.z],
                    "center": [(part_min.x + part_max.x) / 2,
                                (part_min.y + part_max.y) / 2,
                                (part_min.z + part_max.z) / 2],
                    "size": [part_max.x - part_min.x,
                            part_max.y - part_min.y,
                            part_max.z - part_min.z]
                }
                
                bbox_info["parts"].append(part_info)
            except Exception as e:
                print(f"Error processing part {obj.name}: {e}")
                continue
    
    # Save bbox information to a JSON file
    with open(os.path.join(output_folder, 'bbox_info.json'), 'w') as f:
        json.dump(bbox_info, f, indent=2)
    
    # Check if saving was successful
    if os.path.exists(os.path.join(output_folder, 'bbox_info.json')):
        return {'sha256': sha256, 'cond_rendered': True}
    return {'sha256': sha256, 'cond_rendered': False}

if __name__ == '__main__':
    # Import dataset-specific utilities based on command line argument
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Create output directory
    os.makedirs(os.path.join(opt.output_dir, 'renders_cond'), exist_ok=True)

    # Load and filter metadata
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    
    if opt.instances is None:
        # Filter metadata based on conditions
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        # if 'cond_rendered' in metadata.columns:
        #     metadata = metadata[metadata['cond_rendered'] == False]
    else:
        # Process specific instances provided
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # Distribute work across processes using rank and world_size
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    print(f'Processing {len(metadata)} objects...')

    # Process objects in parallel
    func = partial(_render_cond, output_dir=opt.output_dir)
    cond_rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')