"""
Dataset Metadata Builder Script

This script builds and maintains metadata for a 3D model dataset processing pipeline. It tracks 
the processing status of individual 3D models through various stages such as downloading, rendering,
voxelization, feature extraction, and latent representation generation. The script merges processing
records from separate CSV files and can also build metadata directly from file existence checks.

Usage:
    python build_metadata.py dataset_name --output_dir /path/to/output [options]

Example:
    python build_metadata.py shapenet --output_dir /data/shapenet_processed --field rendered,voxelized
"""

import os
import shutil
import sys
import time
import importlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
import utils3d

def get_first_directory(path):  
    """
    Find and return the name of the first directory in the given path.
    
    Args:
        path: Path to search for directories
        
    Returns:
        Name of the first directory found, or None if no directories exist
    """
    with os.scandir(path) as it:  
        for entry in it:  
            if entry.is_dir():  
                return entry.name  
    return None

def need_process(key):
    """
    Check if a specific processing field needs to be processed based on user options.
    
    Args:
        key: The processing field name to check
        
    Returns:
        Boolean indicating whether the field should be processed
    """
    return key in opt.field or opt.field == ['all']

if __name__ == '__main__':
    # Import the dataset-specific utilities module based on the first argument
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    # Set up the command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--field', type=str, default='all',
                        help='Fields to process, separated by commas')
    parser.add_argument('--from_file', action='store_true',
                        help='Build metadata from file instead of from records of processings.' +
                             'Useful when some processing fail to generate records but file already exists.')
    # Add dataset-specific arguments
    dataset_utils.add_args(parser)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    # Create necessary output directories
    os.makedirs(opt.output_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.output_dir, 'merged_records'), exist_ok=True)

    # Split comma-separated fields into a list
    opt.field = opt.field.split(',')
    
    # Generate a timestamp for record-keeping
    timestamp = str(int(time.time()))

    # Load or create the metadata dataframe
    if os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        print('Loading previous metadata...')
        metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    else:
        # Generate new metadata from dataset-specific function
        metadata = dataset_utils.get_metadata(**opt)
    
    # set index with sha256 hash
    metadata.set_index('sha256', inplace=True)
    
    # Merge downloaded files metadata from separate CSV files
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('downloaded_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        if 'local_path' in metadata.columns:
            metadata.update(df, overwrite=True)
        else:
            metadata = metadata.join(df, on='sha256', how='left')
        # Move processed files to merged_records directory
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
            
    # Detect available models for different processing types
    image_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'features')):
        image_models = os.listdir(os.path.join(opt.output_dir, 'features'))
    latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'latents')):
        latent_models = os.listdir(os.path.join(opt.output_dir, 'latents'))
    ss_latent_models = []
    if os.path.exists(os.path.join(opt.output_dir, 'ss_latents')):
        ss_latent_models = os.listdir(os.path.join(opt.output_dir, 'ss_latents'))
    print(f'Image models: {image_models}')
    print(f'Latent models: {latent_models}')
    print(f'Sparse Structure latent models: {ss_latent_models}')

    # Initialize default values for processing status columns
    if 'rendered' not in metadata.columns:
        metadata['rendered'] = [False] * len(metadata)
    if 'voxelized' not in metadata.columns:
        metadata['voxelized'] = [False] * len(metadata)
    if 'num_voxels' not in metadata.columns:
        metadata['num_voxels'] = [0] * len(metadata)
    if 'cond_rendered' not in metadata.columns:
        metadata['cond_rendered'] = [False] * len(metadata)
    # Initialize feature extraction status columns
    for model in image_models:
        if f'feature_{model}' not in metadata.columns:
            metadata[f'feature_{model}'] = [False] * len(metadata)
    # Initialize latent model status columns
    for model in latent_models:
        if f'latent_{model}' not in metadata.columns:
            metadata[f'latent_{model}'] = [False] * len(metadata)
    # Initialize sparse structure latent model status columns
    for model in ss_latent_models:
        if f'ss_latent_{model}' not in metadata.columns:
            metadata[f'ss_latent_{model}'] = [False] * len(metadata)
    
    # Merge rendered model processing records
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('rendered_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # Merge voxelized model processing records
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('voxelized_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # Merge conditional rendered model processing records
    df_files = [f for f in os.listdir(opt.output_dir) if f.startswith('cond_rendered_') and f.endswith('.csv')]
    df_parts = []
    for f in df_files:
        try:
            df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
        except:
            pass
    if len(df_parts) > 0:
        df = pd.concat(df_parts)
        df.set_index('sha256', inplace=True)
        metadata.update(df, overwrite=True)
        for f in df_files:
            shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
    
    # Merge feature extraction records for each image model
    for model in image_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'feature_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
                
    # Merge latent model extraction records
    for model in latent_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'latent_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))
                
    # Merge sparse structure latent model extraction records
    for model in ss_latent_models:
        df_files = [f for f in os.listdir(opt.output_dir) if f.startswith(f'ss_latent_{model}_') and f.endswith('.csv')]
        df_parts = []
        for f in df_files:
            try:
                df_parts.append(pd.read_csv(os.path.join(opt.output_dir, f)))
            except:
                pass
        if len(df_parts) > 0:
            df = pd.concat(df_parts)
            df.set_index('sha256', inplace=True)
            metadata.update(df, overwrite=True)
            for f in df_files:
                shutil.move(os.path.join(opt.output_dir, f), os.path.join(opt.output_dir, 'merged_records', f'{timestamp}_{f}'))

    # If from_file option is set, verify processing status by checking file existence
    if opt.from_file:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor, \
            tqdm(total=len(metadata), desc="Building metadata") as pbar:
            def worker(sha256):
                """
                Worker function to check file existence and update metadata for a specific model
                
                Args:
                    sha256: The model identifier to process
                """
                try:
                    # Check for rendered files
                    if need_process('rendered') and metadata.loc[sha256, 'rendered'] == False and \
                        os.path.exists(os.path.join(opt.output_dir, 'renders', sha256, 'transforms.json')):
                        metadata.loc[sha256, 'rendered'] = True
                    
                    # Check for voxelized files and count voxels
                    if need_process('voxelized') and metadata.loc[sha256, 'rendered'] == True and metadata.loc[sha256, 'voxelized'] == False and \
                        os.path.exists(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply')):
                        try:
                            pts = utils3d.io.read_ply(os.path.join(opt.output_dir, 'voxels', f'{sha256}.ply'))[0]
                            metadata.loc[sha256, 'voxelized'] = True
                            metadata.loc[sha256, 'num_voxels'] = len(pts)
                        except Exception as e:
                            pass
                    
                    # Check for conditional renders
                    if need_process('cond_rendered') and metadata.loc[sha256, 'cond_rendered'] == False and \
                        os.path.exists(os.path.join(opt.output_dir, 'renders_cond', sha256, 'transforms.json')):
                        metadata.loc[sha256, 'cond_rendered'] = True
                    
                    # Check for image feature extractions
                    for model in image_models:
                        if need_process(f'feature_{model}') and \
                            metadata.loc[sha256, f'feature_{model}'] == False and \
                            metadata.loc[sha256, 'rendered'] == True and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'features', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'feature_{model}'] = True
                    
                    # Check for latent model extractions
                    for model in latent_models:
                        if need_process(f'latent_{model}') and \
                            metadata.loc[sha256, f'latent_{model}'] == False and \
                            metadata.loc[sha256, 'rendered'] == True and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'latents', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'latent_{model}'] = True
                    
                    # Check for sparse structure latent model extractions
                    for model in ss_latent_models:
                        if need_process(f'ss_latent_{model}') and \
                            metadata.loc[sha256, f'ss_latent_{model}'] == False and \
                            metadata.loc[sha256, 'voxelized'] == True and \
                            os.path.exists(os.path.join(opt.output_dir, 'ss_latents', model, f'{sha256}.npz')):
                            metadata.loc[sha256, f'ss_latent_{model}'] = True
                    pbar.update()
                except Exception as e:
                    print(f'Error processing {sha256}: {e}')
                    pbar.update()
            
            # Process all models in parallel using thread pool
            executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    # Save the updated metadata and generate dataset statistics
    metadata.to_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    num_downloaded = metadata['local_path'].count() if 'local_path' in metadata.columns else 0
    
    # Write detailed statistics to a file
    with open(os.path.join(opt.output_dir, 'statistics.txt'), 'w') as f:
        f.write('Statistics:\n')
        f.write(f'  - Number of assets: {len(metadata)}\n')
        f.write(f'  - Number of assets downloaded: {num_downloaded}\n')
        f.write(f'  - Number of assets rendered: {metadata["rendered"].sum()}\n')
        f.write(f'  - Number of assets voxelized: {metadata["voxelized"].sum()}\n')
        if len(image_models) != 0:
            f.write(f'  - Number of assets with image features extracted:\n')
            for model in image_models:
                f.write(f'    - {model}: {metadata[f"feature_{model}"].sum()}\n')
        if len(latent_models) != 0:
            f.write(f'  - Number of assets with latents extracted:\n')
            for model in latent_models:
                f.write(f'    - {model}: {metadata[f"latent_{model}"].sum()}\n')
        if len(ss_latent_models) != 0:
            f.write(f'  - Number of assets with sparse structure latents extracted:\n')
            for model in ss_latent_models:
                f.write(f'    - {model}: {metadata[f"ss_latent_{model}"].sum()}\n')
        f.write(f'  - Number of assets with captions: {metadata["captions"].count()}\n')
        f.write(f'  - Number of assets with image conditions: {metadata["cond_rendered"].sum()}\n')
        
    # Print statistics to console
    with open(os.path.join(opt.output_dir, 'statistics.txt'), 'r') as f:
        print(f.read())