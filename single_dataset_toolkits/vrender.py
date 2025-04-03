import os
import sys
sys.path.append(os.path.abspath("."))
if os.environ.get("USE_PIP_BLENDER"):
    from argparse import ArgumentParser
else:
    from vrenderer.blender_utils import BlenderArgumentParser as ArgumentParser
from vrenderer.render import initialize, render_and_save
from vrenderer.spec import InitializationSettings, RuntimeSettings, CameraSpec
from vrenderer.ops import polar_to_transform_matrix
import json
import math
import numpy as np

import bpy
from mathutils import Vector
from utils import sphere_hammersley_sequence

def get_cameras(initialization_output):
    """
    Get camera specifications for rendering from different viewpoints
    
    Args:
        initialization_output: Output from the initialization process
        
    Returns:
        List of CameraSpec objects
    """
    # Calculate camera field of view from lens and sensor parameters
    default_camera_lens = 50
    default_camera_sensor_width = 36
    camera_angle_x = 2.0*math.atan(default_camera_sensor_width/2/default_camera_lens)
    fov_deg = math.degrees(camera_angle_x)
    
    # Define camera viewpoints (elevation and azimuth angles)
    elev_list = [20, 20, 20, 20]
    azim_list = [270., 180., 90., 0.]

    # Calculate camera distance based on model bounding box
    bbox_size = np.array(initialization_output.normalization_spec.bbox_max) - np.array(initialization_output.normalization_spec.bbox_min)
    ratio = 1.0
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

def get_sphere_cameras(initialization_output, num_views=150):
    """
    Generate camera positions uniformly distributed on a sphere around the object.
    
    Args:
        initialization_output: Output from the initialization process
        num_views (int, optional): Number of camera viewpoints to generate. Defaults to 150.
    
    Returns:
        List of CameraSpec objects
    """
    # Calculate camera field of view from lens and sensor parameters
    default_camera_lens = 50
    default_camera_sensor_width = 36
    camera_angle_x = 2.0*math.atan(default_camera_sensor_width/2/default_camera_lens)
    fov_deg = math.degrees(camera_angle_x)
    
    # Generate camera positions using Hammersley sequence for uniform distribution on sphere
    # Add random offset to avoid fixed patterns in the distribution
    offset = (np.random.rand(), np.random.rand())
    
    # Calculate camera distance based on model bounding box
    bbox_size = np.array(initialization_output.normalization_spec.bbox_max) - np.array(initialization_output.normalization_spec.bbox_min)
    ratio = 1.0
    distance = ratio * default_camera_lens / default_camera_sensor_width * \
        math.sqrt(bbox_size[0]**2 + bbox_size[1]**2+bbox_size[2]**2)
    
    # Create camera specifications for each viewpoint
    cameras = []
    for i in range(num_views):
        # sphere_hammersley_sequence returns yaw and pitch angles in radians
        yaw, pitch = sphere_hammersley_sequence(i, num_views, offset)
        
        # Convert yaw/pitch to elevation/azimuth angles expected by polar_to_transform_matrix
        # pitch is elevation (0 is equator, π/2 is north pole)
        elevation = math.degrees(pitch)
        # yaw is azimuth (0 is +x axis, increasing counterclockwise)
        azimuth = math.degrees(yaw)
        
        cameras.append(CameraSpec(
            projection_type="PERSP",
            transform_matrix=polar_to_transform_matrix(elevation, azimuth, distance),
            fov_deg=fov_deg,
        ))
    
    return cameras

def process(model_path, save_dir):
    """
    Process a 3D model by rendering it from different viewpoints and creating part-wise renders
    
    Args:
        model_path: Path to input 3D model file (.glb)
        save_dir: Directory to save rendered images
    """
    # Initialize model settings with normalized scale and merged vertices
    initialization_settings = InitializationSettings(
        file_path=model_path,
        merge_vertices=True, 
        normalizing_scale=0.5
    )
    initialization_output = initialize(initialization_settings)

    # Configure render settings using Blender's EEVEE engine
    runtime_settings = RuntimeSettings(
        use_environment_map=False,
        frame_index=1,
        engine="BLENDER_EEVEE",
        use_gtao=True,  # Use ground truth ambient occlusion
        use_ssr=True,   # Use screen space reflections
        use_high_quality_normals=True,
        use_auto_smooth=True,
        auto_smooth_angle_deg=30.,
        blend_mode="OPAQUE",
        resolution_x = 256,
        resolution_y = 256
    )

    cameras = get_sphere_cameras(initialization_output)

    # Render whole model first
    render_outputs = render_and_save(
        settings=runtime_settings,
        cameras=cameras,
        initialization_output=initialization_output,
        save_dir=save_dir,
        name_format="whole_{camera_index:04d}.{file_ext}",
        render_types = {"Color"},
        overwrite=True
    )

    # Helper function to get object's Z-center position
    # obj.bound_box has 8 corners in local space
    # obj.matric_world is the transformation matrix from local to world space
    def get_bbox_center_z(obj):
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        bbox_center = sum(bbox, Vector()) / 8 # get the center of the bounding box
        return bbox_center.z

    # Sort objects by Z position (top to bottom)
    objects = bpy.data.objects

    object_positions = [(i, get_bbox_center_z(obj)) for i, obj in enumerate(objects) if obj.type == 'MESH']
    sorted_objects = sorted(object_positions, key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_objects] # Get the original indices of the sorted objects
        
    # Create a dictionary to store whole model render information
    render_info = {}
    
    # Add whole model render info
    whole_model_paths = [os.path.join(save_dir, f"whole_{i:04d}.webp") for i in range(len(cameras))]
    render_info = {
        "image_paths": whole_model_paths,
        "camera_params": [
            {
                "projection_type": cam.projection_type,
                "transform_matrix": cam.transform_matrix.tolist() if hasattr(cam.transform_matrix, 'tolist') \
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
    
    # Save render information to a JSON file
    json_path = os.path.join("/mnt/pfs/users/yangyunhan/yufan/data/render_part_images", "render_info.json")
    with open(json_path, 'w') as f:
        json.dump(render_info, f, indent=2)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument("-o", type=str, default="ab9804f981184f8db6f1f814c2b8c169_2", 
                        help="Object ID from Objaverse dataset")
    args = parser.parse_args()

    # Define paths for model storage and processing
    model_root = "/mnt/pfs/users/yangyunhan/yufan/data/raw"  # Root directory for storing 3D models
    
    # Download the 3D model from Objaverse using the provided ID
    uid = args.o
    
    # Define output directory for rendered images
    save_root = "/mnt/pfs/users/yangyunhan/yufan/data/render_images"

    # Construct full paths for input model and output directory
    model_path = os.path.join(model_root, f"{uid}.glb")  # Path to the downloaded .glb file
    save_dir = os.path.join(save_root, uid)  # Directory where renders will be saved

    # Process the model: render views and create part-wise visualizations
    process(model_path, save_dir)

    # Note: Commented out sections below show alternative usage patterns:
    # - Processing multiple files from a directory
    # - Processing files with different formats (obj)
    # - Various input/output directory configurations
    # - Batch processing for different datasets (vroid, tripo, rodin, etc.)