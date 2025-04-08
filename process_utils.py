import os
import imageio
from trellis.utils import render_utils, postprocessing_utils
import json
from PIL import Image

def save_outputs(outputs, output_dir="./z_output", filename_prefix="sample_image", save_video=True, save_glb=True):
    """
    Save various outputs from the model.
    
    Args:
        outputs (dict): Dictionary containing 'gaussian', 'radiance_field', and 'mesh' outputs
        output_dir (str): Directory to save outputs
        filename_prefix (str): Prefix for output filenames
    """
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if save_video:
        # Render and save the gaussian video
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        gaussian_video_path = f"{output_dir}/{filename_prefix}_gs_text.mp4"
        if os.path.exists(gaussian_video_path):
            os.remove(gaussian_video_path)
        imageio.mimsave(gaussian_video_path, video, fps=30)
        
        # Render and save the radiance field video
        video = render_utils.render_video(outputs['radiance_field'][0])['color']
        rf_video_path = f"{output_dir}/{filename_prefix}_rf_text.mp4"
        if os.path.exists(rf_video_path):
            os.remove(rf_video_path)
        imageio.mimsave(rf_video_path, video, fps=30)
        
        # Render and save the mesh video
        video = render_utils.render_video(outputs['mesh'][0])['normal']
        mesh_video_path = f"{output_dir}/{filename_prefix}_mesh_text.mp4"
        if os.path.exists(mesh_video_path):
            os.remove(mesh_video_path)
        imageio.mimsave(mesh_video_path, video, fps=30)
        
    if save_glb:
        # Create and save GLB file
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024,
        )
        glb_path = f"{output_dir}/{filename_prefix}.glb"
        if os.path.exists(glb_path):
            os.remove(glb_path)
        glb.export(glb_path)
        
        # Save Gaussians as PLY files
        ply_path = f"{output_dir}/{filename_prefix}.ply"
        if os.path.exists(ply_path):
            os.remove(ply_path)
        outputs['gaussian'][0].save_ply(ply_path)

def load_render_cond_info(sh256, json_dir="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/renders_cond"):
    """
    Load render information from JSON file and extract part names and image paths
    
    Args:
        json_path: Path to the JSON file containing render information
    
    Returns:
        A tuple containing:
        - Dictionary mapping each part name to its corresponding image paths
        - List of whole_{num_parts}_2 model image paths
    """
    data_dir = os.path.join(json_dir, sh256)

    json_path = os.path.join(data_dir, "transforms.json")

    with open(json_path, 'r') as f:
        render_info = json.load(f)
    
    cond_list = []
    for i in range(0, len(render_info["frames"]), 4):
        view_list = []
        for j in range(4):
            image_path = os.path.join(data_dir, render_info["frames"][i+j]["file_path"])
            image = Image.open(image_path)
            view_list.append(image)
        cond_list.append(view_list)
    
    return cond_list