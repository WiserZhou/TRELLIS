"""
Amazon Berkeley Objects (ABO) Dataset Module

This file provides functionality to download, extract, and process the Amazon Berkeley Objects (ABO) dataset.
It handles the downloading of the dataset from Amazon S3, extraction of 3D models, validation of file integrity
using SHA256 hashes, and parallel processing of dataset instances.

The ABO dataset contains 3D models and is used for research in computer vision and 3D modeling.
"""

import os
import argparse
import tarfile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    """
    Add dataset-specific command line arguments to the parser.
    Currently a placeholder for future implementation.
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to
    """
    pass


def get_metadata(**kwargs):
    """
    Load the metadata for the ABO dataset from a pre-defined CSV file.
    
    Returns:
        pandas.DataFrame: DataFrame containing metadata for the ABO dataset
    """
    metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ABO.csv")
    return metadata
        

def download(metadata, output_dir, **kwargs):
    """
    Download and extract the ABO dataset.
    
    Args:
        metadata (pandas.DataFrame): DataFrame containing dataset metadata
        output_dir (str): Directory to save the downloaded dataset
        **kwargs: Additional keyword arguments
        
    Returns:
        pandas.DataFrame: DataFrame mapping SHA256 hashes to local file paths
    """
    # Create the raw directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # Download the dataset archive if it doesn't exist locally
    if not os.path.exists(os.path.join(output_dir, 'raw', 'abo-3dmodels.tar')):
        try:
            os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)
            os.system(f"wget -O {output_dir}/raw/abo-3dmodels.tar https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-3dmodels.tar")
        except:
            print("\033[93m")
            print("Error downloading ABO dataset. Please check your internet connection and try again.")
            print("Or, you can manually download the abo-3dmodels.tar file and place it in the {output_dir}/raw directory")
            print("Visit https://amazon-berkeley-objects.s3.amazonaws.com/index.html for more information")
            print("\033[0m")
            raise FileNotFoundError("Error downloading ABO dataset")
    
    # Prepare to track downloaded files
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    
    # Extract files from the tarball and verify their integrity
    with tarfile.open(os.path.join(output_dir, 'raw', 'abo-3dmodels.tar')) as tar:
        with ThreadPoolExecutor(max_workers=1) as executor, \
            tqdm(total=len(metadata), desc="Extracting") as pbar:
            def worker(instance: str) -> str:
                """
                Worker function to extract a file and compute its SHA256 hash.
                
                Args:
                    instance (str): File identifier to extract
                    
                Returns:
                    str: SHA256 hash of the extracted file, or None if extraction fails
                """
                try:
                    tar.extract(f"3dmodels/original/{instance}", path=os.path.join(output_dir, 'raw'))
                    sha256 = get_file_hash(os.path.join(output_dir, 'raw/3dmodels/original', instance))
                    pbar.update()
                    return sha256
                except Exception as e:
                    pbar.update()
                    print(f"Error extracting for {instance}: {e}")
                    return None
                
            # Map worker function to all instances in the metadata
            sha256s = executor.map(worker, metadata.index)
            executor.shutdown(wait=True)

    # Verify hash values match the expected ones from metadata
    for k, sha256 in zip(metadata.index, sha256s):
        if sha256 is not None:
            if sha256 == metadata.loc[k, "sha256"]:
                downloaded[sha256] = os.path.join('raw/3dmodels/original', k)
            else:
                print(f"Error downloading {k}: sha256s do not match")

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    """
    Process each instance in the dataset using the provided function with parallel execution.
    
    Args:
        metadata (pandas.DataFrame): DataFrame containing dataset metadata
        output_dir (str): Directory containing the dataset files
        func (callable): Function to apply to each instance
        max_workers (int, optional): Maximum number of worker threads. Defaults to CPU count.
        desc (str, optional): Description for the progress bar. Defaults to 'Processing objects'.
        
    Returns:
        pandas.DataFrame: DataFrame containing the results of processing
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    # Convert metadata to list of dictionaries for easier processing
    metadata = metadata.to_dict('records')

    # Prepare for parallel processing
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                """
                Worker function to process a single dataset instance.
                
                Args:
                    metadatum (dict): Metadata for a single instance
                """
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    file = os.path.join(output_dir, local_path)
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            # Map worker function to all instances in the metadata
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)
