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
import numpy as np
from utils import sphere_hammersley_sequence

def _render(file_path, sha256, output_dir, num_views):
    """
    Render a 3D model from multiple viewpoints using vrenderer.
    
    Args:
        file_path: Path to the 3D model file
        sha256: SHA256 hash of the model file, used as identifier
        output_dir: Directory to save rendered images
        num_views: Number of views to render
        
    Returns:
        Dictionary with rendering results
    """
    from vrenderer.render import initialize, render_and_save
    from vrenderer.spec import InitializationSettings, RuntimeSettings, CameraSpec
    from vrenderer.ops import polar_to_transform_matrix
    import math
    
    output_folder = os.path.join(output_dir, 'renders', sha256)
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize model settings with normalized scale and merged vertices
    initialization_settings = InitializationSettings(
        file_path=os.path.expanduser(file_path),
        merge_vertices=True, 
        normalizing_scale=0.5
    )
    initialization_output = initialize(initialization_settings)
    
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
    
    # Configure render settings
    runtime_settings = RuntimeSettings(
        use_environment_map=False,
        frame_index=1,
        engine="BLENDER_EEVEE",  # Using BLENDER_EEVEE as per original script
        resolution_x=512,
        resolution_y=512,
        use_gtao=True,
        use_ssr=True,
        use_high_quality_normals=True,
        use_auto_smooth=True,
        auto_smooth_angle_deg=30.,
        blend_mode="OPAQUE"
    )

    # Render images
    render_outputs = render_and_save(
        settings=runtime_settings,
        cameras=cameras,
        initialization_output=initialization_output,
        save_dir=output_folder,
        name_format="{camera_index:04d}.{file_ext}",
        render_types={"Color"},
        overwrite=True
    )
    
    # Create transforms.json file for compatibility with original code
    transforms = {
        "frames": [],
        "camera_params": [
            {
                "projection_type": cam.projection_type,
                "transform_matrix": cam.transform_matrix.tolist() if hasattr(cam.transform_matrix, 'tolist') 
                    else [list(row) for row in cam.transform_matrix],
                "fov_deg": cam.fov_deg
            } for cam in cameras
        ],
        "normalization": {
            "scaling_factor": initialization_output.normalization_spec.scaling_factor,
            "rotation_euler": initialization_output.normalization_spec.rotation_euler,
            "translation": initialization_output.normalization_spec.translation,
            "bbox_min": initialization_output.normalization_spec.bbox_min,
            "bbox_max": initialization_output.normalization_spec.bbox_max
        }
    }
    
    # Add frame data with file paths
    for i in range(num_views):
        transforms["frames"].append({
            "file_path": f"{i:04d}.webp",
            "transform_matrix": transforms["camera_params"][i]["transform_matrix"],
            "camera_angle_x": transforms["camera_params"][i]["fov_deg"] * (math.pi / 180.0)  # Convert fov_deg to radians
        })
    
    # Save transforms.json
    with open(os.path.join(output_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)
    
    # Check if rendering was successful
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'rendered': True}
    return {'sha256': sha256, 'rendered': False}

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
