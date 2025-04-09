import os
import glob
import torch
from safetensors import safe_open

def load_safetensors_content(safetensors_file_path):
    """
    Load and return the content from a safetensors file without saving
    
    Args:
        safetensors_file_path (str): Path to the input safetensors file
        
    Returns:
        dict: Dictionary containing the tensors from the safetensors file
    """
    try:
        # Load the safetensors file
        tensors = {}
        with safe_open(safetensors_file_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        
        print(f"Successfully loaded content from {safetensors_file_path}")
        return tensors
    except Exception as e:
        print(f"Error loading {safetensors_file_path}: {str(e)}")
        return None

def convert_safetensors_to_pt(safetensors_file_path, pt_file_path):
    """
    Convert a safetensors file to PyTorch (.pt) format
    
    Args:
        safetensors_file_path (str): Path to the input safetensors file
        pt_file_path (str): Path where the output PyTorch file will be saved
        
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        tensors = load_safetensors_content(safetensors_file_path)
        if tensors is None:
            return False
        
        # Save as PyTorch file
        torch.save(tensors, pt_file_path)
        
        print(f"Successfully converted {safetensors_file_path} to {pt_file_path}")
        return True
    except Exception as e:
        print(f"Error converting {safetensors_file_path} to {pt_file_path}: {str(e)}")
        return False

# Define the source directory for safetensors files
source_dir = "/mnt/pfs/users/yangyunhan/yufan/TRELLIS/pretrained/TRELLIS-image-large/ckpts"
# Define output directory (current directory)
output_dir = os.getcwd()

# Find all .safetensors files
safetensors_files = glob.glob(os.path.join(source_dir, "*.safetensors"))

print(f"Found {len(safetensors_files)} safetensors files")

for safetensor_file in safetensors_files:
    # Get just the filename without path
    filename = os.path.basename(safetensor_file)
    # Change extension from .safetensors to .pt
    pt_filename = os.path.splitext(filename)[0] + ".pt"
    # Create full output path
    output_path = os.path.join(output_dir, pt_filename)
    
    print(f"Converting {filename} to {pt_filename}...")
    convert_safetensors_to_pt(safetensor_file, output_path)
    print(f"Conversion complete: {output_path}")

print("All conversions completed successfully")