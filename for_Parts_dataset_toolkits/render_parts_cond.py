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
# if os.environ.get("USE_PIP_BLENDER"):
#     from argparse import ArgumentParser
# else:
#     from vrenderer.blender_utils import BlenderArgumentParser as ArgumentParser
from argparse import ArgumentParser

import importlib
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np


def get_four_cameras(initialization_output, num_views):
    """
    Get camera specifications for rendering from different viewpoints
    
    Args:
        initialization_output: Output from the vrenderer initialization process containing model information
        
    Returns:
        List of CameraSpec objects representing different camera viewpoints
    """
    from vrenderer.spec import CameraSpec
    from vrenderer.ops import polar_to_transform_matrix
    import math
    # Calculate camera field of view from lens and sensor parameters
    # Standard values for a 35mm camera
    default_camera_lens = 50
    default_camera_sensor_width = 36
    camera_angle_x = 2.0*math.atan(default_camera_sensor_width/2/default_camera_lens)
    fov_deg = math.degrees(camera_angle_x)
    
    # Define camera viewpoints (elevation and azimuth angles)
    # Using 4 views from different sides of the object (90 degrees apart)
    elev_list = [20, 20, 20, 20]
    azim_list = [270., 180., 90., 0.]

    # Calculate camera distance based on model bounding box
    # This ensures the entire object is visible in the frame
    bbox_size = np.array(initialization_output.normalization_spec.bbox_max) - np.array(initialization_output.normalization_spec.bbox_min)
    ratio = 1.0
    # Calculate the ratio of the bounding box size to the default camera sensor width
    distance = ratio * default_camera_lens / default_camera_sensor_width * \
        math.sqrt(bbox_size[0]**2 + bbox_size[1]**2+bbox_size[2]**2)
    distance_list = [distance] * len(elev_list)
    
    # Create camera specifications for each viewpoint
    assert len(elev_list) == len(azim_list) == len(distance_list)
    cameras = []
    for i in range(len(elev_list)):
        cameras.append(CameraSpec(
            projection_type="PERSP",
            transform_matrix=polar_to_transform_matrix(elev_list[i], azim_list[i], distance_list[i]),
            fov_deg=fov_deg,
        ))
        
    return cameras

def get_sphere_cameras(initialization_output, num_views):

    from utils import sphere_hammersley_sequence
    from vrenderer.spec import CameraSpec
    from vrenderer.ops import polar_to_transform_matrix
    import math
    # Calculate camera field of view
    default_camera_lens = 50
    default_camera_sensor_width = 36
    camera_angle_x = 2.0*math.atan(default_camera_sensor_width/2/default_camera_lens)
    fov_deg = math.degrees(camera_angle_x)
    
    # Calculate camera distance based on model bounding box
    bbox_size = np.array(initialization_output.normalization_spec.bbox_max) - np.array(initialization_output.normalization_spec.bbox_min)
    ratio = 1.0
    distance = ratio * default_camera_lens / default_camera_sensor_width * \
        math.sqrt(bbox_size[0]**2 + bbox_size[1]**2+bbox_size[2]**2)
    
    # Generate camera positions using Hammersley sequence
    cameras = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        
        # Convert yaw/pitch to elevation/azimuth angles
        elevation = math.degrees(pitch)
        azimuth = math.degrees(yaw)
        
        cameras.append(CameraSpec(
            projection_type="PERSP",
            transform_matrix=polar_to_transform_matrix(elevation, azimuth, distance),
            fov_deg=fov_deg,
        ))

    return cameras
    

def _render_cond(file_path, sha256, output_dir, num_views=24):
    """
    Render a 3D model for conditioning purposes using vrenderer.
    
    Args:
        file_path: Path to the 3D model file
        sha256: SHA256 hash of the model file, used as identifier
        output_dir: Directory to save rendered images

    Returns:
        Dictionary with rendering results containing status information
    """
    from vrenderer.render import initialize, render_and_save
    from vrenderer.spec import InitializationSettings, RuntimeSettings
    import math

    import bpy
    from mathutils import Vector

    # Ensure the proper GLB import process
    # bpy.ops.import_scene.gltf(filepath='/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/raw/Parts/A/ab9804f981184f8db6f1f814c2b8c169.glb')
    # Create output directory using the model's hash as identifier
    output_folder = os.path.join(output_dir, 'renders_cond', sha256)
    os.makedirs(output_folder, exist_ok=True)
    # 获取当前脚本文件的路径
    current_file_path = __file__
    # 获取当前文件的父目录
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file_path)))
    file_path = os.path.join(parent_dir, file_path)
    # print(parent_dir)
    # print(file_path)
    
    # Initialize model settings with normalized scale and merged vertices
    # Normalizing scale ensures consistent rendering across differently sized models
    initialization_settings = InitializationSettings(
        file_path=file_path,
        merge_vertices=True, 
        normalizing_scale=0.5
    )
    initialization_output = initialize(initialization_settings)
    # print("Initialization completed")
    # Configure render settings
    # Using EEVEE renderer for a good balance between quality and speed
    runtime_settings = RuntimeSettings(
        use_environment_map=False,
        frame_index=1,
        engine="BLENDER_EEVEE",  # Using BLENDER_EEVEE as per original script
        resolution_x=1024,       # Using the 1024 resolution from the original _render_cond
        resolution_y=1024,
        use_gtao=True,           # Ambient occlusion for better depth cues
        use_ssr=True,            # Screen space reflections for better material rendering
        use_high_quality_normals=True,
        use_auto_smooth=True,    # Smooth shading with angle threshold
        auto_smooth_angle_deg=30.,
        blend_mode="OPAQUE"
    )

    # Helper function to get object's Z-center position
    # obj.bound_box has 8 corners in local space
    # obj.matrix_world is the transformation matrix from local to world space
    def get_bbox_center_z(obj):
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        bbox_center = sum(bbox, Vector()) / 8 # get the center of the bounding box
        return bbox_center.z

    # Sort objects by Z position (top to bottom)
    # This helps to render parts in a consistent order
    objects = bpy.data.objects

    object_positions = [(i, get_bbox_center_z(obj)) for i, obj in enumerate(objects) if obj.type == 'MESH']
    sorted_objects = sorted(object_positions, key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_objects] # Get the original indices of the sorted objects
        
    # Hide all mesh objects initially
    # We'll show them one by one for individual rendering
    # print("Hide all objects")
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            # print("Hide object: " + obj.name)
            obj.hide_render = True
    
    # Create a list to store object names in order
    obj_name_list = []
    # Collect names of each part
    for i in range(len(sorted_indices)):
        obj = objects[sorted_indices[i]]
        # print("Render object " + str(i) + ": " + obj.name)
        obj_name_list.append(obj.name)
    
    # Create transforms.json file structure to store camera parameters
    transforms = {
        "frames": []
    }

    # Render each part separately
    for i, name in enumerate(obj_name_list):

        # print(f"Rendering part {i+1}/{len(obj_name_list)}: {name}")
        # Initialize the renderer with just this part visible
        initialization_output = initialize(initialization_settings, part_names=[name])
        # Get camera positions for multiple viewpoints
        cameras = get_four_cameras(initialization_output, num_views)
        # Perform the actual rendering and save the results
        render_outputs = render_and_save(
            settings=runtime_settings,
            cameras=cameras,
            initialization_output=initialization_output,
            save_dir=output_folder,
            name_format=f"{i:04d}_{{camera_index:04d}}.{{file_ext}}",
            render_types = {"Color"},
            overwrite=True
        )

        for camera_index, cam in enumerate(cameras):
            transforms["frames"].append({
                "file_path": f"{i:04d}_{camera_index:04d}.webp",
                "transform_matrix": cam.transform_matrix.tolist() if hasattr(cam.transform_matrix, 
                        'tolist') else [list(row) for row in cam.transform_matrix],
                "camera_angle_x": cam.fov_deg * (math.pi / 180.0)
            })
    
    # Save transforms.json with all camera parameters
    with open(os.path.join(output_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)
    
    # Check if rendering was successful
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
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
    parser.add_argument('--num_views', type=int, default=24,
                        help='Number of views to render')
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
        if 'cond_rendered' in metadata.columns:
            metadata = metadata[metadata['cond_rendered'] == False]
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

    # Filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders_cond', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'cond_rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # Process objects in parallel
    func = partial(_render_cond, output_dir=opt.output_dir, num_views=opt.num_views)
    cond_rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    cond_rendered = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
    
    # Save rendering results
    cond_rendered.to_csv(os.path.join(opt.output_dir, f'cond_rendered_{opt.rank}.csv'), index=False)
