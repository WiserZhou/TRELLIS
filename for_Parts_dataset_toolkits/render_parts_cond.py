import os
import json
import copy
import sys
sys.path.append(os.path.abspath("."))
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
import numpy as np
from utils import sphere_hammersley_sequence

from vrenderer.render import initialize, render_and_save
from vrenderer.spec import InitializationSettings, RuntimeSettings, CameraSpec
from vrenderer.ops import polar_to_transform_matrix
import math

def get_cameras(initialization_output):
    """
    Get camera specifications for rendering from different viewpoints
    
    Returns:
        List of CameraSpec objects
    """
    # Calculate camera field of view from lens and sensor parameters
    default_camera_lens = 50
    default_camera_sensor_width = 36
    camera_angle_x = 2.0*math.atan(default_camera_sensor_width/2/default_camera_lens)
    fov_deg = math.degrees(camera_angle_x)
    
    # Define camera viewpoints (elevation and azimuth angles)
    # Define camera viewpoints (elevation and azimuth angles) (20, 270. are the default values)
    elev_list = [20, 20, 20, 20]
    azim_list = [270., 180., 90., 0.]

    # Calculate camera distance based on model bounding box
    # Calculate the scale of bounding box
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

def _render_cond(file_path, sha256, output_dir, num_views):
    """
    Render a 3D model for conditioning purposes using vrenderer.
    
    Args:
        file_path: Path to the 3D model file
        sha256: SHA256 hash of the model file, used as identifier
        output_dir: Directory to save rendered images
        num_views: Number of views to render
        
    Returns:
        Dictionary with rendering results
    """
    
    output_folder = os.path.join(output_dir, 'renders_cond', sha256)
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
    
    # Calculate variable camera parameters as in single_dataset_toolkits version
    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (num_views,))
    radii = [1 / np.sqrt(k) for k in ks]
    fovs = [2 * np.arcsin(np.sqrt(3) / 2 / r) / np.pi * 180 for r in radii]
    
    for i in range(num_views):
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        
        # Convert yaw/pitch to elevation/azimuth angles
        elevation = math.degrees(pitch)
        azimuth = math.degrees(yaw)
        
        # Use variable distance and FOV instead of fixed values
        current_distance = distance * radii[i]
        current_fov_deg = fovs[i]
        
        cameras.append(CameraSpec(
            projection_type="PERSP",
            transform_matrix=polar_to_transform_matrix(elevation, azimuth, current_distance),
            fov_deg=current_fov_deg,
        ))
    
    # Configure render settings
    runtime_settings = RuntimeSettings(
        use_environment_map=False,
        frame_index=1,
        engine="BLENDER_EEVEE",  # Using BLENDER_EEVEE as per original script
        resolution_x=1024,       # Using the 1024 resolution from the original _render_cond
        resolution_y=1024,
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
    
    # Create transforms.json file
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
            "camera_angle_x": transforms["camera_params"][i]["fov_deg"] * (math.pi / 180.0)
        })
    
    # Save transforms.json
    with open(os.path.join(output_folder, 'transforms.json'), 'w') as f:
        json.dump(transforms, f, indent=2)
    
    # Check if rendering was successful
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'sha256': sha256, 'cond_rendered': True}
    return {'sha256': sha256, 'cond_rendered': False}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
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

    os.makedirs(os.path.join(opt.output_dir, 'renders_cond'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'cond_rendered' in metadata.columns:
            metadata = metadata[metadata['cond_rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'renders_cond', sha256, 'transforms.json')):
            records.append({'sha256': sha256, 'cond_rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_render_cond, output_dir=opt.output_dir, num_views=opt.num_views)
    cond_rendered = dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Rendering objects')
    cond_rendered = pd.concat([cond_rendered, pd.DataFrame.from_records(records)])
    cond_rendered.to_csv(os.path.join(opt.output_dir, f'cond_rendered_{opt.rank}.csv'), index=False)
