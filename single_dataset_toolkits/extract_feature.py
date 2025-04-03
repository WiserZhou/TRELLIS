"""
extract_feature.py - 3D Object Feature Extraction Tool

This script extracts deep learning features from rendered 3D objects using the DINOv2 vision transformer model.
It processes images from render_info.json, extracts visual features, and projects these features onto
3D voxel positions for both whole models and individual parts.
"""

import os
import copy
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from torchvision import transforms
from PIL import Image

# Disable gradient computation for inference
torch.set_grad_enabled(False)

def get_data_from_render_info(render_info, target="whole"):
    """
    Extract data from render_info.json file.
    
    Args:
        render_info: Loaded render_info.json content
        target: "whole" for whole model or "parts" for individual parts
        
    Yields:
        Dictionary containing processed image and camera parameters
    """
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(item_info):
            """
            Worker function to process a single view.
            
            Args:
                item_info: Dictionary containing image path and camera parameters
                
            Returns:
                Dictionary with processed image and camera parameters, or None if error occurs
            """
            image_path = item_info["image_path"]
            camera_param = item_info["camera_param"]
            
            try:
                # Open the image file
                image = Image.open(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None
            
            # Resize image and process alpha channel
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            
            # Check if image has alpha channel
            if image.shape[-1] == 4:
                # Apply alpha channel masking to keep only visible parts of the object
                image = image[:, :, :3] * image[:, :, 3:]
            else:
                # If no alpha channel, just use RGB
                image = image[:, :, :3]
                
            # Convert to torch tensor with channels first
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            # Process camera transformation matrix
            c2w = torch.tensor(camera_param["transform_matrix"])
            # Convert from OpenGL to camera coordinate system
            c2w[:3, 1:3] *= -1
            # Get camera extrinsics (world to camera matrix)
            extrinsics = torch.inverse(c2w)
            
            # Get field of view in radians (convert from degrees if needed)
            fov_deg = camera_param["fov_deg"]
            fov = fov_deg * (3.14159 / 180)
            
            # Calculate camera intrinsics from the field of view
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }
        
        # Prepare list of items to process
        items_to_process = []
        
        if target == "whole":
            # Process whole model images
            for i, image_path in enumerate(render_info["whole_model"]["image_paths"]):
                items_to_process.append({
                    "image_path": image_path,
                    "camera_param": render_info["whole_model"]["camera_params"][i]
                })
        else:
            # Process part images
            for part in render_info["parts"]:
                part_name = part["name"]
                part_index = part["index"]
                for i, image_path in enumerate(part["image_paths"]):
                    items_to_process.append({
                        "image_path": image_path,
                        "camera_param": part["camera_params"][i],
                        "part_name": part_name,
                        "part_index": part_index
                    })
        
        # Process all items in parallel and yield valid results
        datas = executor.map(worker, items_to_process)
        for data in datas:
            if data is not None:
                yield data

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the features')
    parser.add_argument('--render_info', type=str, required=True,
                        help='Path to render_info.json file')
    parser.add_argument('--process_parts', action='store_true',
                        help='Process individual parts instead of whole model')
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
    n_patch = 518 // 14

    # Load render info JSON
    with open(opt.render_info, 'r') as f:
        render_info = json.load(f)
    
    # Extract model ID from the file path in render_info
    model_id = os.path.basename(os.path.dirname(render_info["whole_model"]["image_paths"][0]))
    
    # Determine what to process: whole model or parts
    target = "parts" if opt.process_parts else "whole"
    
    print(f"Processing {target} for model: {model_id}")
    
    try:
        # Process the data
        data = []
        for datum in get_data_from_render_info(render_info, target):
            # Apply normalization to the image
            datum['image'] = transform(datum['image'])
            data.append(datum)
        
        if not data:
            print("No valid data found")
            exit(1)
            
        # Batch processing of images
        features_list = []
        
        # Process in batches
        for i in range(0, len(data), opt.batch_size):
            batch_data = data[i:i+opt.batch_size]
            bs = len(batch_data)
            # Prepare batch input tensors
            batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
            
            # Extract features with DINOv2
            features = dinov2_model(batch_images, is_training=True)
            
            # Extract patch tokens (excluding CLS token and register tokens)
            patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
            features_list.append(patchtokens.cpu())
            
        # Concatenate all batch results
        all_features = torch.cat(features_list, dim=0)
        
        # Save features
        if target == "whole":
            # Save whole model features
            save_path = os.path.join(opt.output_dir, 'features', feature_name, f'{model_id}_whole.pt')
            torch.save(all_features, save_path)
            print(f"Saved features for whole model to {save_path}")
        else:
            # Save part features - group by part
            # This assumes parts are processed in order and have their metadata
            for part in render_info["parts"]:
                part_name = part["name"]
                part_index = part["index"]
                part_id = f"{model_id}_{part_index}_{part_name.replace(' ', '_')}"
                # Get indices in the data corresponding to this part
                start_idx = part_index * len(part["image_paths"])
                end_idx = start_idx + len(part["image_paths"])
                
                if start_idx < len(all_features):
                    part_features = all_features[start_idx:end_idx]
                    save_path = os.path.join(opt.output_dir, 'features', feature_name, f'{part_id}.pt')
                    torch.save(part_features, save_path)
                    print(f"Saved features for part {part_name} to {save_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")

