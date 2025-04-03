"""
extract_feature.py - 3D Object Feature Extraction Tool

This script extracts deep learning features from rendered 3D objects using the DINOv2 vision transformer model.
It processes images from render_info.json, extracts visual features, and projects these features onto
3D voxel positions for whole models.
"""

import os
import json
import argparse
import torch
import numpy as np
import utils3d
import traceback
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Disable gradient computation for inference
torch.set_grad_enabled(False)

def process_image(image_path, size=518):
    """Process a single image for feature extraction
    
    Args:
        image_path: Path to the image file
        size: Target size for resizing
        
    Returns:
        Processed image as a tensor, or None if processing fails
    """
    try:
        # Open and resize the image
        image = Image.open(image_path)
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255
        
        # Process alpha channel if present
        if image.shape[-1] == 4:
            image = image[:, :, :3] * image[:, :, 3:]
        else:
            image = image[:, :, :3]
            
        # Convert to torch tensor with channels first
        return torch.from_numpy(image).permute(2, 0, 1).float()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def is_valid_transform_matrix(transform_matrix):
    """Check if a transform matrix is valid
    
    Args:
        transform_matrix: The transformation matrix to check
        
    Returns:
        Boolean indicating if the matrix is valid
    """
    if not transform_matrix or len(transform_matrix) < 4:
        return False
        
    # Check if any rows are missing or incomplete
    for i in range(4):
        if i >= len(transform_matrix) or not transform_matrix[i] or len(transform_matrix[i]) < 4:
            return False

    return True

def process_camera_params(camera_param):
    """Process camera parameters to get extrinsics and intrinsics
    
    Args:
        camera_param: Camera parameters dictionary
        
    Returns:
        Tuple of (extrinsics, intrinsics) as torch tensors
    """
    transform_matrix = camera_param["transform_matrix"]
    
    # Convert the matrix to torch tensor
    c2w = torch.tensor(transform_matrix, dtype=torch.float32)
    
    # Convert from OpenGL to camera coordinate system
    c2w[:3, 1:3] *= -1
    # Get camera extrinsics (world to camera matrix)
    extrinsics = torch.inverse(c2w)
    
    # Get field of view in radians
    fov_deg = camera_param["fov_deg"]
    fov = fov_deg * (3.14159 / 180)
    
    # Calculate camera intrinsics from the field of view
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
    
    return extrinsics, intrinsics

def load_image_data(render_info, image_size, transform):
    """Load and process all image data
    
    Args:
        render_info: Dictionary with render information
        image_size: Size for image processing
        transform: Image normalization transform
        
    Returns:
        List of processed image data
    """
    data = []
    print(f"Found {len(render_info['image_paths'])} images in render_info.json")
    
    # Process all images directly from image_paths
    for item_index in tqdm(range(len(render_info["image_paths"])), desc="Loading images"):
        # Skip invalid indices
        if item_index >= len(render_info["camera_params"]):
            print(f"Skipping index {item_index}: camera params not available")
            continue
        
        # Get image path and camera parameters
        image_path = render_info["image_paths"][item_index]
        camera_param = render_info["camera_params"][item_index]
        
        # Process the image
        image = process_image(image_path, size=image_size)
        if image is None:
            continue
            
        # Check if transform matrix is valid
        if not is_valid_transform_matrix(camera_param.get("transform_matrix", None)):
            print(f"Skipping image {item_index}: Invalid transform_matrix")
            continue
        
        # Process camera parameters
        extrinsics, intrinsics = process_camera_params(camera_param)
        
        # Apply normalization to the image
        norm_image = transform(image)
        
        # Add processed data to the list
        data.append({
            'image': norm_image,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics
        })
    
    return data

def extract_features_batch(data, batch_size, dinov2_model):
    """Extract features from images in batches
    
    Args:
        data: List of processed image data
        batch_size: Size of processing batch
        dinov2_model: The feature extraction model
        
    Returns:
        List of feature tensors
    """
    features_list = []
    n_patch = 518 // 14  # image_size // patch_size
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Extracting features"):
        batch_data = data[i:i+batch_size]
        bs = len(batch_data)
        # Prepare batch input tensors
        batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
        
        # Extract features with DINOv2
        with torch.no_grad():  # Ensure no gradients are computed
            features = dinov2_model(batch_images, is_training=True)
        
        # Extract patch tokens (excluding CLS token and register tokens)
        patchtokens = features['x_prenorm'][:, 
            dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
        features_list.append(patchtokens.cpu())
    
    return features_list

def save_whole_model_features(features_list, output_dir, feature_name, model_id):
    """Save features for the whole model
    
    Args:
        features_list: List of feature tensors
        output_dir: Directory to save features
        feature_name: Name of the feature type
        model_id: ID of the current model
    
    Returns:
        The concatenated features tensor
    """
    all_features = torch.cat(features_list, dim=0)
    save_path = os.path.join(output_dir, 'features', feature_name, f'{model_id}_whole.pt')
    torch.save(all_features, save_path)
    print(f"Saved features for whole model to {save_path}, tensor shape: {all_features.shape}")
    return all_features

def process_voxel_positions(voxel_path):
    """Load and process voxel positions
    
    Args:
        voxel_path: Path to the voxel file
        
    Returns:
        Tuple of (positions, indices) tensors or None if file not found
    """
    if not os.path.exists(voxel_path):
        print(f"Voxel file {voxel_path} not found, skipping 3D projection")
        return None, None
    
    print(f"Loading voxel positions from {voxel_path}")
    positions = utils3d.io.read_ply(voxel_path)[0]
    positions = torch.from_numpy(positions).float().cuda()
    
    # Convert continuous positions to discrete voxel indices
    indices = ((positions + 0.5) * 64).long()
    
    # Check if indices are within bounds
    if torch.all(indices >= 0) and torch.all(indices < 64):
        return positions, indices
    else:
        print("Some vertices are out of bounds, skipping 3D projection")
        return None, None

def project_features_to_3d(positions, data, features_list, batch_size):
    """Project features to 3D positions
    
    Args:
        positions: 3D point positions tensor
        data: List of processed image data
        features_list: List of feature tensors
        batch_size: Size of processing batch
        
    Returns:
        Averaged projected features
    """
    # Project features to 3D positions
    patchtokens_lst = []
    uv_lst = []
    
    # Process in batches
    for i in tqdm(range(0, len(data), batch_size), desc="Projecting features to 3D"):
        batch_data = data[i:i+batch_size]
        bs = len(batch_data)
        batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
        batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
        
        # Project 3D points to 2D for each view
        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        uv_lst.append(uv)
        
        # Get corresponding features
        patchtokens = features_list[i//batch_size][:bs].cuda()
        patchtokens_lst.append(patchtokens)
    
    # Process all projected features
    projected_features = []
    for i in range(len(patchtokens_lst)):
        patchtokens = patchtokens_lst[i]
        uv = uv_lst[i]
        
        # Project features using grid_sample
        projected = F.grid_sample(
            patchtokens,
            uv.unsqueeze(1),
            mode='bilinear',
            align_corners=False,
        ).squeeze(2).permute(0, 2, 1).cpu()
        
        projected_features.append(projected)
    
    # Concatenate and average features across views
    all_projected = torch.cat(projected_features, dim=0)
    return torch.mean(all_projected, dim=0).numpy().astype(np.float16)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="/mnt/pfs/users/yangyunhan/yufan/data/dino_features",
                        help='Directory to save the features')
    parser.add_argument('--render_info', type=str, default="/mnt/pfs/users/yangyunhan/yufan/data/render_images/render_info.json",
                        help='Path to render_info.json file')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of images to process in one batch')
    opt = parser.parse_args()
    opt = edict(vars(opt))

    # Setup output directories for features
    feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, 'features', feature_name), exist_ok=True)

    # Load the DINOv2 model for feature extraction
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()  # Set to evaluation mode and move to GPU
    
    # Define image normalization transform using ImageNet statistics
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Calculate number of patches based on image size and patch size (14x14)
    image_size = 518
    patch_size = 14
    n_patch = image_size // patch_size

    # Load render info JSON
    with open(opt.render_info, 'r') as f:
        render_info = json.load(f)
    
    # Extract model ID from the file path in render_info
    if not render_info["image_paths"]:
        print("No image paths found in render_info.json")
        exit(1)
    
    first_image_path = render_info["image_paths"][0]
    model_id = os.path.basename(os.path.dirname(first_image_path))
    
    print(f"Processing sphere whole for model: {model_id}")
    
    try:
        # Load and process image data
        data = load_image_data(render_info, image_size, transform)
        
        if not data:
            print("No valid data found")
            exit(1)
            
        print(f"Loaded {len(data)} valid images out of {len(render_info['image_paths'])} total images")
            
        # Extract features in batches
        features_list = extract_features_batch(data, opt.batch_size, dinov2_model)
        
        # Save whole model features
        all_features = save_whole_model_features(features_list, opt.output_dir, feature_name, model_id)
        
        # Load and process voxel positions
        voxel_path = os.path.join(os.path.dirname(opt.output_dir), 'voxel', f'{model_id}.ply')
        positions, indices = process_voxel_positions(voxel_path)
        
        if positions is not None:
            # Project features to 3D positions
            avg_features = project_features_to_3d(positions, data, features_list, opt.batch_size)
            
            # Save projected features
            pack = {
                'indices': indices.cpu().numpy().astype(np.uint8),  # Store voxel indices
                'patchtokens': avg_features
            }
            projected_path = os.path.join(opt.output_dir, 'features', feature_name, f'{model_id}_projected.npz')
            np.savez_compressed(projected_path, **pack)
            print(f"Saved projected features to {projected_path}, shape: {avg_features.shape}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
