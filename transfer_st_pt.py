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

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert safetensors file to PyTorch format or return content")
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path")
    parser.add_argument("--output", type=str, help="Output PyTorch file path (if not provided, content will just be loaded)")
    parser.add_argument("--print-keys", action="store_true", help="Print the keys of tensors in the file")
    
    args = parser.parse_args()
    
    if args.output:
        convert_safetensors_to_pt(args.input, args.output)
    else:
        tensors = load_safetensors_content(args.input)
        if tensors and args.print_keys:
            print("Keys in the safetensors file:")
            for key in tensors.keys():
                print(f"- {key}: {tensors[key].shape}")
