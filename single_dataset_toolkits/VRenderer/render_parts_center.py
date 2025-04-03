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
from PIL import Image, ImageOps

def create_collage(image_folder, output_path):
    """
    Create a collage from multiple rendered images
    
    Args:
        image_folder: Directory containing the rendered images
        output_path: Path where the final collage will be saved
    """
    # Get all .webp files from the folder
    images = [img for img in os.listdir(image_folder) if img.endswith('.webp')]
    
    # Prioritize whole model renders by moving them to the front
    # This ensures the full model views appear first in the collage
    images.sort()
    if 'whole_0000.webp' in images:
        images.remove('whole_0000.webp')
        images.insert(0, 'whole_0000.webp') # insert the whole_0000.webp to the first position
    if 'whole_0001.webp' in images:
        images.remove('whole_0001.webp')
        images.insert(1, 'whole_0001.webp')
    
    # Process and load all images
    image_objects = []
    border_size = 3  # Size of border in pixels
    for img in images:
        image_path = os.path.join(image_folder, img)
        # Load image and convert from RGBA to RGB if necessary
        img = np.array(Image.open(image_path))
        if img.shape[-1] == 4:  # If image has alpha channel
            # Replace transparent pixels with white background
            mask_img = img[..., 3] == 0
            img[mask_img] = [255, 255, 255, 255]
            img = img[..., :3]  # Keep only RGB channels
            img = Image.fromarray(img.astype('uint8'))
        # Add black border around each image
        img_with_border = ImageOps.expand(img, border=border_size, fill='black')
        image_objects.append(img_with_border)
    
    # Get dimensions of a single image (assuming all images are same size)
    img_width, img_height = image_objects[0].size
    
    # Calculate dimensions for the final collage
    num_images = len(image_objects)
    num_columns = 8  # Number of images per row
    num_rows = (num_images + num_columns - 1) // num_columns  # Ceiling division
    collage_width = num_columns * img_width
    collage_height = num_rows * img_height
    
    # Create new blank image for the collage
    collage_image = Image.new('RGB', (collage_width, collage_height))
    
    # Paste all images into the collage in a grid layout
    for index, img in enumerate(image_objects):
        # Calculate x,y position for each image
        x = (index % num_columns) * img_width
        y = (index // num_columns) * img_height
        collage_image.paste(img, (x, y))
    
    # Save the final collage
    collage_image.save(output_path)

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

    cameras = get_cameras(initialization_output)

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
        
    # Create a dictionary to store all render information
    render_info = {
        "whole_model": {},
        "parts": []
    }
    
    # Add whole model render info
    whole_model_paths = [os.path.join(save_dir, f"whole_{i:04d}.webp") for i in range(len(cameras))]
    render_info["whole_model"] = {
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

    # Hide all mesh objects initially
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.hide_render = True
    
    obj_name_list = []
    # Render each part separately
    for i in range(len(sorted_indices)):
        obj = objects[sorted_indices[i]]
        # obj.hide_render = False
        obj_name_list.append(obj.name)
        # name = obj.name
    
    def find_obj_and_hidden(obj_name):
        for obj in objects:
            if obj.name == obj_name:
                # return obj
                obj.hide_render = False
                return obj
        # return None

    # Render each part separately
    for i, name in enumerate(obj_name_list):
        # objects[i].hide_render = False
        initialization_output = initialize(initialization_settings, part_names=[name])
        cameras = get_cameras(initialization_output)
        # obj_sp = find_obj_and_hidden(name)
        part_paths = [os.path.join(save_dir, f"part_{i:04d}_{name}_{cam_idx:04d}.webp") for cam_idx in range(len(cameras))]
        
        render_outputs = render_and_save(
            settings=runtime_settings,
            cameras=cameras,
            initialization_output=initialization_output,
            save_dir=save_dir,
            name_format=f"part_{i:04d}_{name}_{{camera_index:04d}}.{{file_ext}}",
            render_types = {"Color"},
            overwrite=True
        )
        
        # obj_sp.hide_render = True
        # Add part render info
        part_info = {
            "name": name,
            "index": i,
            "image_paths": part_paths,
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
        render_info["parts"].append(part_info)
    
    # Save all render information to a JSON file
    json_path = os.path.join("/mnt/pfs/users/yangyunhan/yufan/data/render_part_images", "render_info.json")
    with open(json_path, 'w') as f:
        json.dump(render_info, f, indent=2)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    parser.add_argument("-o", type=str, default="ab9804f981184f8db6f1f814c2b8c169", 
                        help="Object ID from Objaverse dataset")
    args = parser.parse_args()

    # Define paths for model storage and processing
    model_root = "/mnt/pfs/users/yangyunhan/yufan/data"  # Root directory for storing 3D models
    
    # Download the 3D model from Objaverse using the provided ID
    uid = args.o
    
    # Define output directory for rendered images
    save_root = "/mnt/pfs/users/yangyunhan/yufan/data/render_part_images"

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