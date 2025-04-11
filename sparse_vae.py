"""
Sparse Structure Variational Autoencoder for 3D volumes.
This module provides functionality to encode and decode 3D structures using a VAE.
"""
import os
import argparse
import time
import json
from typing import Dict, Tuple, Optional, Any, Union

# Third-party imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import utils

# Local imports
from trellis.models.sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
import utils3d
from trellis.representations.octree import DfsOctree as Octree
from trellis.renderers import OctreeRenderer
from trellis.utils.render_utils import get_part_bbox

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VAE encoding and decoding.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="VAE Encoding and Decoding")
    
    # Model settings
    parser.add_argument("--config_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/configs/vae/ss_vae_conv3d_16l8_fp16.json",
                        help="Path to VAE configuration file")
    parser.add_argument("--encoder_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/pretrained/TRELLIS-image-large-pt/ss_enc_conv3d_16l8_fp16.pt",
                        help="Path to encoder checkpoint")
    parser.add_argument("--decoder_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/pretrained/TRELLIS-image-large-pt/ss_dec_conv3d_16l8_fp16.pt",
                        help="Path to decoder checkpoint")
    
    # Processing settings
    parser.add_argument("--input_path", type=str,
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/voxels/a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee.ply",
                        help="Path to input 3D volume file (if None, will generate random data)")
    parser.add_argument("--input_glb", type=str,
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/raw/Parts/A/ab9804f981184f8db6f1f814c2b8c169.glb",
                        help="Path to input GLB file")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Resolution of the 3D volume")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="./vae_outputs",
                        help="Directory to save outputs")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Visualize results")
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dict containing the loaded configuration
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the config file is not valid JSON
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")


def load_vae_from_config(
    config: Dict[str, Any], 
    encoder_path: Optional[str] = None, 
    decoder_path: Optional[str] = None, 
    device: str = "cuda"
) -> Tuple[SparseStructureEncoder, SparseStructureDecoder]:
    """
    Load VAE encoder and decoder from configuration and checkpoints.
    
    Args:
        config: Dictionary containing model configuration
        encoder_path: Path to the encoder checkpoint file
        decoder_path: Path to the decoder checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (encoder, decoder) loaded models
    """
    # Create encoder and decoder from config
    encoder_config = config["models"]["encoder"]["args"]
    decoder_config = config["models"]["decoder"]["args"]
    
    encoder = SparseStructureEncoder(
        in_channels=encoder_config["in_channels"],
        latent_channels=encoder_config["latent_channels"],
        num_res_blocks=encoder_config["num_res_blocks"],
        num_res_blocks_middle=encoder_config["num_res_blocks_middle"],
        channels=encoder_config["channels"],
        use_fp16=encoder_config.get("use_fp16", False)
    )
    
    decoder = SparseStructureDecoder(
        out_channels=decoder_config["out_channels"],
        latent_channels=decoder_config["latent_channels"],
        num_res_blocks=decoder_config["num_res_blocks"],
        num_res_blocks_middle=decoder_config["num_res_blocks_middle"],
        channels=decoder_config["channels"],
        use_fp16=decoder_config.get("use_fp16", False)
    )
    
    # Load encoder weights if provided
    if encoder_path and os.path.exists(encoder_path):
        print(f"Loading encoder from {encoder_path}")
        encoder_checkpoint = torch.load(encoder_path, map_location=device)
        
        # Handle different checkpoint formats
        if "model" in encoder_checkpoint:
            encoder.load_state_dict(encoder_checkpoint["model"])
        elif "encoder" in encoder_checkpoint:
            encoder.load_state_dict(encoder_checkpoint["encoder"])
        else:
            # Try loading directly
            try:
                encoder.load_state_dict(encoder_checkpoint)
            except Exception as e:
                print(f"Could not load encoder weights: {e}")
    else:
        print("No encoder checkpoint found or provided, using randomly initialized weights")
    
    # Load decoder weights if provided
    if decoder_path and os.path.exists(decoder_path):
        print(f"Loading decoder from {decoder_path}")
        decoder_checkpoint = torch.load(decoder_path, map_location=device)
        
        # Handle different checkpoint formats
        if "model" in decoder_checkpoint:
            decoder.load_state_dict(decoder_checkpoint["model"])
        elif "decoder" in decoder_checkpoint:
            decoder.load_state_dict(decoder_checkpoint["decoder"])
        else:
            # Try loading directly
            try:
                decoder.load_state_dict(decoder_checkpoint)
            except Exception as e:
                print(f"Could not load decoder weights: {e}")
    else:
        print("No decoder checkpoint found or provided, using randomly initialized weights")
    
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    return encoder, decoder


def load_ply_voxels(file_path: str, resolution: int, device: str = "cuda") -> torch.Tensor:
    """
    Load voxel data from a PLY file.
    
    Args:
        file_path: Path to the PLY file
        resolution: Target resolution for the voxel grid
        device: Device to put the tensor on
        
    Returns:
        Tensor of shape [1, 1, resolution, resolution, resolution]
        
    Raises:
        ImportError: If open3d is not installed
        FileNotFoundError: If the PLY file doesn't exist
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D is required to load PLY files. Install with: pip install open3d")
    
    print(f"Loading PLY data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PLY file not found: {file_path}")
    
    # Load the PLY point cloud
    pcd = o3d.io.read_point_cloud(file_path)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
    
    # Normalize points to [0, 1] range
    min_coords, _ = torch.min(points, dim=0)
    max_coords, _ = torch.max(points, dim=0)
    points = (points - min_coords) / (max_coords - min_coords + 1e-8)
    
    # Scale to resolution and convert to voxel indices
    points = (points * (resolution - 1)).round().long()
    
    # Create empty voxel grid
    voxels = torch.zeros((1, 1, resolution, resolution, resolution), device=device)
    
    # Fill voxels with points
    for p in points:
        if 0 <= p[0] < resolution and 0 <= p[1] < resolution and 0 <= p[2] < resolution:
            voxels[0, 0, p[0], p[1], p[2]] = 1.0
    
    print(f"Loaded voxel grid of shape {voxels.shape}, with {points.shape[0]} points")
    return voxels


def visualize_3d(
    volume: torch.Tensor, 
    output_path: Optional[str] = None, 
    device: str = "cuda"
) -> None:
    """
    Visualizes a 3D volume using Octree rendering from multiple viewpoints.
    
    Args:
        volume: 3D or 4D tensor to visualize [C, D, H, W] or [D, H, W]
        output_path: Path to save the visualization (optional)
        device: Device to perform rendering on
    """
    # Take first channel if multi-channel (4D input)
    if len(volume.shape) == 4:
        volume = volume[0]  # Now volume shape is [D, H, W]
    
    # Set up the renderer with options
    renderer = OctreeRenderer()
    # Configure rendering quality and appearance settings
    renderer.rendering_options.resolution = 512  # Output image resolution
    renderer.rendering_options.near = 0.8        # Near clipping plane distance
    renderer.rendering_options.far = 1.6         # Far clipping plane distance
    renderer.rendering_options.bg_color = (1, 1, 1)  # White background
    renderer.rendering_options.ssaa = 2  # Super sampling anti-aliasing factor for smoother edges
    renderer.pipe.primitive = 'voxel'    # Set rendering primitive type to voxel
    
    # Create octree representation of the volume
    resolution = volume.shape[0]
    volume = volume.to(device)  # Ensure volume is on the correct device
    
    # Extract coordinates of occupied voxels (where value > 0)
    # Returns a tensor of shape [N, 3] where N is the number of occupied voxels
    coords = torch.nonzero(volume > 0, as_tuple=False)
    
    # Create octree representation for efficient rendering of sparse 3D data
    representation = Octree(
        depth=10,                          # Maximum tree depth
        aabb=[-0.5, -0.5, -0.5, 1, 1, 1], # Axis-aligned bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
        device=device,
        primitive='voxel',                 # Render as voxels
        sh_degree=0,                       # Spherical harmonics degree (0 for flat shading)
        primitive_config={'solid': True},  # Configure voxels to be solid rather than transparent
    )
    
    # Set positions of occupied voxels, normalized to [0,1] range
    representation.position = coords.float() / resolution
    
    # Set depth of each voxel based on resolution
    # This determines the level of detail in the octree representation
    representation.depth = torch.full(
        (representation.position.shape[0], 1), 
        int(np.log2(resolution))-2,   # Calculate appropriate depth based on resolution 
        dtype=torch.uint8,          # Depth is stored as uint8
        device=device
    )
    
    # Create a 2x2 grid of rendered views
    image = torch.zeros(3, 1024, 1024).to(device)  # Create empty RGB image tensor (3 channels)
    tile = [2, 2]  # Grid layout configuration: 2 rows, 2 columns
    
    # Set up camera positions from 4 different angles
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 0, 90, 180, 270 degrees around object
    yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)  # Add random rotation offset for variety
    yaws = [y + yaws_offset for y in yaws]  # Apply offset to all camera positions
    pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]  # Random up/down angle for each view

    exts = []  # List to store camera extrinsics (position and orientation)
    ints = []  # List to store camera intrinsics (internal parameters like focal length)

    # Calculate camera parameters for each viewpoint
    for yaw, pitch in zip(yaws, pitch):
        # Calculate camera origin position using spherical coordinates
        # Places camera on a sphere around the center point
        orig = torch.tensor([
            np.sin(yaw) * np.cos(pitch),   # x coordinate
            np.cos(yaw) * np.cos(pitch),   # y coordinate
            np.sin(pitch),                 # z coordinate
        ]).float().cuda() * 3              # Multiply by 2 to set camera distance
        
        fov = torch.deg2rad(torch.tensor(30)).cuda()  # 30 degree field of view (in radians)
        
        # Create camera extrinsics (position and orientation in world space)
        # look_at function creates a transformation that points camera from orig toward target [0,0,0]
        extrinsics = utils3d.torch.extrinsics_look_at(
            orig,                             # Camera position
            torch.tensor([0, 0, 0]).float().cuda(),  # Look at center
            torch.tensor([0, 0, 1]).float().cuda()   # Up vector
        )
        
        # Create camera intrinsics (internal camera parameters)
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        
        # Store camera parameters for later use
        exts.append(extrinsics)
        ints.append(intrinsics)

    # Render the object from each viewpoint and assemble the grid
    for j, (ext, intr) in enumerate(zip(exts, ints)):
        # Render the octree representation using current camera settings
        res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
        
        # Place each rendered view in the appropriate position in the grid
        # Calculate grid position based on j and tile configuration
        image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 
              512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
    
    # Set value range for normalization when saving
    value_range = (0, 1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the assembled grid image
    utils.save_image(
            image,
            output_path,
            nrow=int(np.sqrt(4)),  # Arrange images in a square grid (2×2)
            normalize=True,        # Normalize pixel values
            value_range=value_range,  # Use specified value range for normalization
        )

def generate_random_input(
    batch_size: int, 
    in_channels: int, 
    resolution: int, 
    device: str = "cuda"
) -> torch.Tensor:
    """
    Generate random input tensor for the VAE.
    
    Args:
        batch_size: Batch size
        in_channels: Number of input channels
        resolution: Resolution of the 3D volume
        device: Device to put the tensor on
        
    Returns:
        Random tensor of shape [batch_size, in_channels, resolution, resolution, resolution]
    """
    print("Generating random input data")
    return torch.randn(
        batch_size, in_channels, resolution, resolution, resolution
    ).to(device)

def mask_bbox_latent(latent: torch.Tensor) -> torch.Tensor:
    """
    Mask the latent tensor based on the bounding box information.
    
    Args:
        latent: Latent tensor of shape [batch_size, channels, depth, height, width]
        
    Returns:
        Masked latent tensor with values outside the bounding box set to zero
    """
    print(f" latent shape {latent.shape}") # torch.Size([1, 1, 64, 64, 64])
    # latent shape: [1, 8, 16, 16, 16]
    bbox_info = get_part_bbox("body", 
        "/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/renders_cond/a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee/bbox_info.json")
    bbox_min = bbox_info["min"]
    bbox_max = bbox_info["max"]
    print(f"original bbox from {bbox_min} to {bbox_max}")
    
    # Get spatial dimensions of the latent tensor
    _, _, D, H, W = latent.shape
    
    # Convert normalized bbox coordinates to latent space indices
    min_d = int(bbox_min[0] * D) + 32
    min_h = int(bbox_min[1] * H) + 32
    min_w = int(bbox_min[2] * W) + 32
    max_d = int(bbox_max[0] * D) + 32
    max_h = int(bbox_max[1] * H) + 32
    max_w = int(bbox_max[2] * W) + 32
    print(f"Applied mask with bbox {[min_d, min_h, min_w]} to {[max_d, max_h, max_w]}")
    # Ensure indices are within bounds
    min_d = max(0, min_d)
    min_h = max(0, min_h)
    min_w = max(0, min_w)
    max_d = min(D, max_d)
    max_h = min(H, max_h)
    max_w = min(W, max_w)
    
    # Initialize mask with ones
    full_latent = torch.zeros_like(latent)

    full_latent[:, :, min_d:max_d, min_h:max_h, min_w:max_w] = latent[:, :, min_d:max_d, min_h:max_h, min_w:max_w]

    print(f"latent min: {latent.min().item()}, max: {latent.max().item()}")
    print(f"full latent min: {full_latent.min().item()}, max: {full_latent.max().item()}")

    # Apply mask to the latent tensor

    print(f"Applied mask with bbox {[min_d, min_h, min_w]} to {[max_d, max_h, max_w]}")
    
    return full_latent

def main() -> None:
    """Main execution function for VAE encoding and decoding."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Load encoder and decoder from separate files
    encoder, decoder = load_vae_from_config(
        config, 
        encoder_path=args.encoder_path,
        decoder_path=args.decoder_path, 
        device=args.device
    )
    
    # Create or load input data
    if args.input_path and os.path.exists(args.input_path):
        # Check file extension to determine loading method
        if args.input_path.lower().endswith('.ply'):
            input_tensor = load_ply_voxels(args.input_path, args.resolution, args.device)
        else:
            print(f"Unsupported file format for {args.input_path}")
            input_tensor = generate_random_input(
                args.batch_size,
                config["models"]["encoder"]["args"]["in_channels"],
                args.resolution,
                args.device
            )
    else:
        # Generate random data
        input_tensor = generate_random_input(
            args.batch_size,
            config["models"]["encoder"]["args"]["in_channels"],
            args.resolution,
            args.device
        )
    
    # Apply threshold to create binary volume (for visualization)
    binary_input = (input_tensor > 0).float()
    # print(f"encoder latent min: {input_tensor.min().item()}, max: {input_tensor.max().item()}")
    # Encode input to latent space
    input_tensor = mask_bbox_latent(input_tensor)
    print("Encoding input to latent space...")
    start_time = time.time()
    with torch.no_grad():
        latent, mean, logvar = encoder(input_tensor, sample_posterior=True, return_raw=True)
    encode_time = time.time() - start_time
    print(f"Encoding completed in {encode_time:.3f} seconds")

    # print(latent.shape) # torch.Size([1, 8, 16, 16, 16])
    # Decode latent representation back to 3D space
    print("Decoding latent representation...")
    start_time = time.time()
    with torch.no_grad():
        output_tensor = decoder(latent)
    decode_time = time.time() - start_time
    print(f"Decoding completed in {decode_time:.3f} seconds")
    
    # Apply threshold to create binary volume (for visualization)
    binary_output = (output_tensor > 0).float()
    
    # Calculate reconstruction metrics
    mse_loss = torch.nn.functional.mse_loss(output_tensor, input_tensor).item()
    binary_accuracy = (binary_output == binary_input).float().mean().item()
    
    print(f"Reconstruction MSE: {mse_loss:.6f}")
    print(f"Binary accuracy: {binary_accuracy:.6f}")
    
    # Save latent representation
    latent_path = os.path.join(args.output_dir, "latent.pt")
    torch.save({
        "latent": latent.cpu(), 
        "mean": mean.cpu(), 
        "logvar": logvar.cpu()
    }, latent_path)
    print(f"Saved latent representation to {latent_path}")
    
    # Visualize results if requested
    if args.visualize:
        # 3D visualizations
        input_3d_path = os.path.join(args.output_dir, "input_3d_visualization.png")
        visualize_3d(binary_input[0], input_3d_path, device=args.device)
        
        output_3d_path = os.path.join(args.output_dir, "output_3d_visualization.png")
        visualize_3d(binary_output[0], output_3d_path, device=args.device)

if __name__ == "__main__":
    main()