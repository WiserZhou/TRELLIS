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


def get_data(frames, sha256):
    """
    Extract data from rendered frames using parallel processing.
    
    Args:
        frames: List of frame data containing camera information and file paths
        sha256: The unique identifier of the object
        
    Yields:
        Dictionary containing processed image and camera parameters
    """
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            """
            Worker function to process a single view.
            
            Args:
                view: Dictionary containing view information
                
            Returns:
                Dictionary with processed image and camera parameters, or None if error occurs
            """
            image_path = os.path.join(opt.output_dir, 'renders', sha256, view['file_path'])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            
            # Resize image and process alpha channel
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]  # Apply alpha channel
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            # Process camera transformation matrix
            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1  # Convert to camera coordinate system
            extrinsics = torch.inverse(c2w)
            fov = view['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }
        
        # Process all frames in parallel and yield valid results
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--model', type=str, default='dinov2_vitl14_reg',
                        help='Feature extraction model')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    # Setup output directories for features
    feature_name = opt.model
    os.makedirs(os.path.join(opt.output_dir, 'features', feature_name), exist_ok=True)

    # Load the DINOv2 model for feature extraction
    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.model)
    dinov2_model.eval().cuda()  # Set to evaluation mode and move to GPU
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])
    n_patch = 518 // 14  # Calculate number of patches for the ViT model

    # Load metadata for processing
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    
    # Filter objects based on input parameters
    if opt.instances is not None:
        # Only process specific instances listed in a file
        with open(opt.instances, 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        # Apply filters based on other criteria
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if f'feature_{feature_name}' in metadata.columns:
            metadata = metadata[metadata[f'feature_{feature_name}'] == False]  # Skip already processed items
        metadata = metadata[metadata['voxelized'] == True]  # Only process voxelized objects
        metadata = metadata[metadata['rendered'] == True]  # Only process rendered objects

    # Distribute work across multiple processes (for parallel processing)
    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
    records = []

    # Skip objects that have already been processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'feature_{feature_name}' : True})
            sha256s.remove(sha256)

    # Setup thread pools for parallel data loading and saving
    load_queue = Queue(maxsize=4)  # Buffer for loaded data
    try:
        with ThreadPoolExecutor(max_workers=8) as loader_executor, \
            ThreadPoolExecutor(max_workers=8) as saver_executor:
            
            def loader(sha256):
                """
                Worker function to load data for a single object.
                
                Args:
                    sha256: Object identifier
                """
                try:
                    # Load camera transforms
                    with open(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json'), 'r') as f:
                        metadata = json.load(f)
                    frames = metadata['frames']
                    
                    # Process all frames for this object
                    data = []
                    for datum in get_data(frames, sha256):
                        datum['image'] = transform(datum['image'])  # Apply normalization
                        data.append(datum)
                    
                    # Load voxel positions
                    positions = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
                    load_queue.put((sha256, data, positions))
                except Exception as e:
                    print(f"Error loading data for {sha256}: {e}")

            # Start loading all objects in parallel
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack, patchtokens, uv):
                """
                Save extracted features for a single object.
                
                Args:
                    sha256: Object identifier
                    pack: Dictionary containing feature data
                    patchtokens: Extracted patch features
                    uv: UV coordinates for projection
                """
                # Project features to 3D positions
                pack['patchtokens'] = F.grid_sample(
                    patchtokens,
                    uv.unsqueeze(1),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1).cpu().numpy()
                
                # Average features across views
                pack['patchtokens'] = np.mean(pack['patchtokens'], axis=0).astype(np.float16)
                
                # Save compressed features
                save_path = os.path.join(opt.output_dir, 'features', feature_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'feature_{feature_name}' : True})
            
            # Process each object as it becomes available in the queue
            for _ in tqdm(range(len(sha256s)), desc="Extracting features"):
                sha256, data, positions = load_queue.get()
                positions = torch.from_numpy(positions).float().cuda()
                
                # Convert continuous positions to discrete voxel indices
                indices = ((positions + 0.5) * 64).long()
                assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
                
                n_views = len(data)
                N = positions.shape[0]
                pack = {
                    'indices': indices.cpu().numpy().astype(np.uint8),  # Store voxel indices
                }
                
                # Process views in batches to avoid GPU memory issues
                patchtokens_lst = []
                uv_lst = []
                for i in range(0, n_views, opt.batch_size):
                    batch_data = data[i:i+opt.batch_size]
                    bs = len(batch_data)
                    batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
                    batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
                    batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
                    
                    # Extract features with DINOv2
                    features = dinov2_model(batch_images, is_training=True)
                    
                    # Project 3D points to 2D for each view
                    uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
                    
                    # Extract patch tokens (excluding CLS token)
                    patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
                    patchtokens_lst.append(patchtokens)
                    uv_lst.append(uv)
                
                # Concatenate results from all batches
                patchtokens = torch.cat(patchtokens_lst, dim=0)
                uv = torch.cat(uv_lst, dim=0)

                # Save features asynchronously
                saver_executor.submit(saver, sha256, pack, patchtokens, uv)
                
            # Wait for all saving tasks to complete
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    # Save processing records to CSV
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.output_dir, f'feature_{feature_name}_{opt.rank}.csv'), index=False)