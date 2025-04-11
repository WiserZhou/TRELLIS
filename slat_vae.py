"""
Structured Latent Variational Autoencoder for 3D volumes.
This module provides functionality to encode and decode 3D structures using the SLAT VAE
with multiple decoders (Gaussian, Mesh, Radiance Field).
"""
import os
import argparse
import time
import json
from typing import Dict, Tuple, Optional, Any, Union, List, Literal

# Third-party imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision import utils

# Local imports
from trellis.models.structured_latent_vae.encoder import SLatEncoder
from trellis.models.structured_latent_vae.decoder_gs import SLatGaussianDecoder
from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder
from trellis.models.structured_latent_vae.decoder_rf import SLatRadianceFieldDecoder
from trellis.modules.sparse import SparseTensor
import utils3d
from trellis.representations import Gaussian, MeshExtractResult, Strivec
from trellis.renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from trellis.utils.render_utils import get_part_bbox


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VAE encoding and decoding.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="SLAT VAE Encoding and Decoding")
    
    # Model settings
    parser.add_argument("--encoder_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/pretrained/TRELLIS-image-large-pt/slat_enc_swin8_B_64l8_fp16.pt",
                        help="Path to encoder checkpoint")
    parser.add_argument("--gs_decoder_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/pretrained/TRELLIS-image-large-pt/slat_dec_gs_swin8_B_64l8gs32_fp16.pt",
                        help="Path to Gaussian decoder checkpoint")
    parser.add_argument("--mesh_decoder_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/pretrained/TRELLIS-image-large-pt/slat_dec_mesh_swin8_B_64l8m256c_fp16.pt",
                        help="Path to Mesh decoder checkpoint")
    parser.add_argument("--rf_decoder_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/pretrained/TRELLIS-image-large-pt/slat_dec_rf_swin8_B_64l8r16_fp16.pt",
                        help="Path to Radiance Field decoder checkpoint")
    parser.add_argument("--encoder_config_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json",
                        help="Path to encoder configuration file")
    parser.add_argument("--mesh_config_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json",
                        help="Path to mesh decoder configuration file")
    parser.add_argument("--rf_config_path", type=str, 
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/configs/vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json",
                        help="Path to RF decoder configuration file")
    
    # Processing settings
    parser.add_argument("--input_path", type=str,
                        default="/mnt/pfs/users/yangyunhan/yufan/TRELLIS/datasets/Parts/ss_latents/ss_enc_conv3d_16l8_fp16/a36a48897cc62006f0b49bc12d30cd38f28c83704feb203815365405247295ee.npz",
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
    
    # Decoder options
    parser.add_argument("--decoders", type=str, default="all",
                        choices=["all", "gaussian", "mesh", "rf"],
                        help="Which decoders to use")
    
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


def load_slat_vae_from_config(
    config: Dict[str, Any], 
    encoder_path: Optional[str] = None, 
    gs_decoder_path: Optional[str] = None,
    mesh_decoder_path: Optional[str] = None,
    rf_decoder_path: Optional[str] = None,
    device: str = "cuda"
) -> Tuple[SLatEncoder, Dict[str, Union[SLatGaussianDecoder, SLatMeshDecoder, SLatRadianceFieldDecoder]]]:
    """
    Load SLAT VAE encoder and decoders from configuration and checkpoints.
    
    Args:
        config: Dictionary containing model configuration
        encoder_path: Path to the encoder checkpoint file
        gs_decoder_path: Path to the gaussian decoder checkpoint file
        mesh_decoder_path: Path to the mesh decoder checkpoint file
        rf_decoder_path: Path to the radiance field decoder checkpoint file
        device: Device to load the model on
        
    Returns:
        Tuple of (encoder, decoder_dict) loaded models
    """
    # Create encoder from config
    encoder_config = config["models"]["encoder"]["args"]
    
    encoder = SLatEncoder(
        resolution=encoder_config["resolution"],
        in_channels=encoder_config["in_channels"],
        model_channels=encoder_config["model_channels"],
        latent_channels=encoder_config["latent_channels"],
        num_blocks=encoder_config["num_blocks"],
        num_heads=encoder_config.get("num_heads"),
        num_head_channels=encoder_config.get("num_head_channels", 64),
        mlp_ratio=encoder_config.get("mlp_ratio", 4.0),
        attn_mode=encoder_config.get("attn_mode", "swin"),
        window_size=encoder_config.get("window_size", 8),
        pe_mode=encoder_config.get("pe_mode", "ape"),
        use_fp16=encoder_config.get("use_fp16", False),
        use_checkpoint=encoder_config.get("use_checkpoint", False),
        qk_rms_norm=encoder_config.get("qk_rms_norm", False)
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
    
    encoder.to(device)
    encoder.eval()
    
    # Create and load decoders
    decoders = {}
    
    # Create and load Gaussian decoder
    if gs_decoder_path:
        gs_config = config["models"]["decoder_gs"]["args"] if "decoder_gs" in config["models"] else config["models"]["decoder"]["args"]
        gs_decoder = SLatGaussianDecoder(
            resolution=gs_config["resolution"],
            model_channels=gs_config["model_channels"],
            latent_channels=gs_config["latent_channels"],
            num_blocks=gs_config["num_blocks"],
            num_heads=gs_config.get("num_heads"),
            num_head_channels=gs_config.get("num_head_channels", 64),
            mlp_ratio=gs_config.get("mlp_ratio", 4.0),
            attn_mode=gs_config.get("attn_mode", "swin"),
            window_size=gs_config.get("window_size", 8),
            pe_mode=gs_config.get("pe_mode", "ape"),
            use_fp16=gs_config.get("use_fp16", False),
            use_checkpoint=gs_config.get("use_checkpoint", False),
            qk_rms_norm=gs_config.get("qk_rms_norm", False),
            representation_config=gs_config.get("representation_config", {})
        )
        
        if os.path.exists(gs_decoder_path):
            print(f"Loading Gaussian decoder from {gs_decoder_path}")
            gs_checkpoint = torch.load(gs_decoder_path, map_location=device)
            
            # Handle different checkpoint formats
            if "model" in gs_checkpoint:
                gs_decoder.load_state_dict(gs_checkpoint["model"])
            elif "decoder" in gs_checkpoint:
                gs_decoder.load_state_dict(gs_checkpoint["decoder"])
            else:
                # Try loading directly
                try:
                    gs_decoder.load_state_dict(gs_checkpoint)
                except Exception as e:
                    print(f"Could not load Gaussian decoder weights: {e}")
        
        gs_decoder.to(device)
        gs_decoder.eval()
        decoders["gaussian"] = gs_decoder
    
    # Create and load Mesh decoder
    if mesh_decoder_path:
        try:
            mesh_config = config["models"]["decoder_mesh"]["args"] if "decoder_mesh" in config["models"] else config["models"].get("decoder_mesh_args", {})
            mesh_decoder = SLatMeshDecoder(
                resolution=mesh_config.get("resolution", encoder_config["resolution"]),
                model_channels=mesh_config.get("model_channels", encoder_config["model_channels"]),
                latent_channels=mesh_config.get("latent_channels", encoder_config["latent_channels"]),
                num_blocks=mesh_config.get("num_blocks", encoder_config["num_blocks"]),
                num_heads=mesh_config.get("num_heads"),
                num_head_channels=mesh_config.get("num_head_channels", 64),
                mlp_ratio=mesh_config.get("mlp_ratio", 4.0),
                attn_mode=mesh_config.get("attn_mode", "swin"),
                window_size=mesh_config.get("window_size", 8),
                pe_mode=mesh_config.get("pe_mode", "ape"),
                use_fp16=mesh_config.get("use_fp16", False),
                use_checkpoint=mesh_config.get("use_checkpoint", False),
                qk_rms_norm=mesh_config.get("qk_rms_norm", False),
                representation_config=mesh_config.get("representation_config", {"use_color": False})
            )
            
            if os.path.exists(mesh_decoder_path):
                print(f"Loading Mesh decoder from {mesh_decoder_path}")
                mesh_checkpoint = torch.load(mesh_decoder_path, map_location=device)
                
                # Handle different checkpoint formats
                if "model" in mesh_checkpoint:
                    mesh_decoder.load_state_dict(mesh_checkpoint["model"])
                elif "decoder" in mesh_checkpoint:
                    mesh_decoder.load_state_dict(mesh_checkpoint["decoder"])
                else:
                    # Try loading directly
                    try:
                        mesh_decoder.load_state_dict(mesh_checkpoint)
                    except Exception as e:
                        print(f"Could not load Mesh decoder weights: {e}")
            
            mesh_decoder.to(device)
            mesh_decoder.eval()
            decoders["mesh"] = mesh_decoder
        except Exception as e:
            print(f"Error loading Mesh decoder: {e}")
    
    # Create and load Radiance Field decoder
    if rf_decoder_path:
        try:
            rf_config = config["models"]["decoder_rf"]["args"] if "decoder_rf" in config["models"] else config["models"].get("decoder_rf_args", {})
            rf_decoder = SLatRadianceFieldDecoder(
                resolution=rf_config.get("resolution", encoder_config["resolution"]),
                model_channels=rf_config.get("model_channels", encoder_config["model_channels"]),
                latent_channels=rf_config.get("latent_channels", encoder_config["latent_channels"]),
                num_blocks=rf_config.get("num_blocks", encoder_config["num_blocks"]),
                num_heads=rf_config.get("num_heads"),
                num_head_channels=rf_config.get("num_head_channels", 64),
                mlp_ratio=rf_config.get("mlp_ratio", 4.0),
                attn_mode=rf_config.get("attn_mode", "swin"),
                window_size=rf_config.get("window_size", 8),
                pe_mode=rf_config.get("pe_mode", "ape"),
                use_fp16=rf_config.get("use_fp16", False),
                use_checkpoint=rf_config.get("use_checkpoint", False),
                qk_rms_norm=rf_config.get("qk_rms_norm", False),
                representation_config=rf_config.get("representation_config", {"rank": 16, "dim": 32})
            )
            
            if os.path.exists(rf_decoder_path):
                print(f"Loading Radiance Field decoder from {rf_decoder_path}")
                rf_checkpoint = torch.load(rf_decoder_path, map_location=device)
                
                # Handle different checkpoint formats
                if "model" in rf_checkpoint:
                    rf_decoder.load_state_dict(rf_checkpoint["model"])
                elif "decoder" in rf_checkpoint:
                    rf_decoder.load_state_dict(rf_checkpoint["decoder"])
                else:
                    # Try loading directly
                    try:
                        rf_decoder.load_state_dict(rf_checkpoint)
                    except Exception as e:
                        print(f"Could not load RF decoder weights: {e}")
            
            rf_decoder.to(device)
            rf_decoder.eval()
            decoders["rf"] = rf_decoder
        except Exception as e:
            print(f"Error loading Radiance Field decoder: {e}")
    
    return encoder, decoders


def voxels_to_sparse_tensor(voxel_grid: torch.Tensor, device: str = "cuda") -> SparseTensor:
    """
    Convert a dense voxel grid to a sparse tensor representation.
    
    Args:
        voxel_grid: A 5D tensor [B, C, D, H, W]
        device: Device to put the tensor on
        
    Returns:
        SparseTensor representation
    """
    batch_size, channels, depth, height, width = voxel_grid.shape
    
    # Find occupied voxels (non-zero values)
    coords = []
    feats = []
    layout = []
    
    start = 0
    for i in range(batch_size):
        # Get occupied voxel indices in this batch item
        # Convert to int32 immediately
        occupied = torch.nonzero(voxel_grid[i].sum(dim=0) > 0, as_tuple=False).to(dtype=torch.int32)  # [N, 3]
        
        if occupied.shape[0] > 0:
            # Add batch dimension to coordinates - IMPORTANT: create on the same device as occupied
            batch_idx = torch.full((occupied.shape[0], 1), i, dtype=torch.int32, device=occupied.device)
            occupied_coords = torch.cat([batch_idx, occupied], dim=1)  # [N, 4] (B, D, H, W)
            
            # Get feature values at occupied positions
            occupied_feats = voxel_grid[i, :, occupied[:, 0], occupied[:, 1], occupied[:, 2]].transpose(0, 1)  # [N, C]
            
            coords.append(occupied_coords)
            feats.append(occupied_feats)
            layout.append(slice(start, start + occupied.shape[0]))
            start += occupied.shape[0]
    
    if not coords:  # Handle empty case
        # Create a single dummy point if no occupied voxels were found
        dummy_coords = torch.zeros((1, 4), dtype=torch.int32, device=voxel_grid.device)
        dummy_feats = torch.zeros((1, channels), dtype=voxel_grid.dtype, device=voxel_grid.device)
        coords = [dummy_coords]
        feats = [dummy_feats]
        layout = [slice(0, 1)]
    
    # Concatenate all coordinates and features
    # Ensure coords are int32 when moving to the specified device
    coords = torch.cat(coords, dim=0).to(device=device, dtype=torch.int32)
    feats = torch.cat(feats, dim=0).to(device)
    
    # Create sparse tensor
    sparse_tensor = SparseTensor(coords=coords, feats=feats)
    sparse_tensor._shape = torch.Size([batch_size, channels])
    sparse_tensor.register_spatial_cache('layout', layout)
    
    return sparse_tensor


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


def visualize_gaussian_3d(
    representations: List[Gaussian], 
    output_path: Optional[str] = None, 
    device: str = "cuda"
) -> None:
    """
    Visualizes a 3D Gaussian representation using GaussianRenderer from multiple viewpoints.
    
    Args:
        representations: List of Gaussian representations
        output_path: Path to save the visualization (optional)
        device: Device to perform rendering on
    """
    # Set up the renderer with options
    renderer = GaussianRenderer()
    # Configure rendering quality and appearance settings
    renderer.rendering_options.resolution = 512  # Output image resolution
    renderer.rendering_options.near = 0.8       # Near clipping plane distance
    renderer.rendering_options.far = 1.6         # Far clipping plane distance
    renderer.rendering_options.bg_color = (1, 1, 1)  # White background
    
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
        orig = torch.tensor([
            np.sin(yaw) * np.cos(pitch),   # x coordinate
            np.cos(yaw) * np.cos(pitch),   # y coordinate
            np.sin(pitch),                 # z coordinate
        ]).float().to(device) * 3          # Multiply by 3 to set camera distance
        
        fov = torch.deg2rad(torch.tensor(30)).to(device)  # 30 degree field of view (in radians)
        
        # Create camera extrinsics (position and orientation in world space)
        extrinsics = utils3d.torch.extrinsics_look_at(
            orig,                                        # Camera position
            torch.tensor([0, 0, 0]).float().to(device),  # Look at center
            torch.tensor([0, 0, 1]).float().to(device)   # Up vector
        )
        
        # Create camera intrinsics (internal camera parameters)
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        
        # Store camera parameters for later use
        exts.append(extrinsics)
        ints.append(intrinsics)

    # Render each representation and assemble them into a grid
    for rep_idx, representation in enumerate(representations):
        # Render the object from each viewpoint and assemble the grid
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            # Render the representation using current camera settings
            res = renderer.render(representation, ext, intr)
            
            # Place each rendered view in the appropriate position in the grid
            image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 
                512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
    
    # Set value range for normalization when saving
    value_range = (0, 1)
    
    # Create output directory if it doesn't exist
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the assembled grid image
        utils.save_image(
            image,
            output_path,
            normalize=True,        # Normalize pixel values
            value_range=value_range,  # Use specified value range for normalization
        )


def visualize_mesh_3d(
    representations: List[MeshExtractResult], 
    output_path: Optional[str] = None, 
    device: str = "cuda"
) -> None:
    """
    Visualizes a 3D mesh representation using MeshRenderer from multiple viewpoints.
    
    Args:
        representations: List of mesh representations
        output_path: Path to save the visualization (optional)
        device: Device to perform rendering on
    """
    # Set up the renderer with options
    renderer = MeshRenderer({"near": 1.0, "far": 3.0, "bg_color": (1, 1, 1)}, device=device)
    renderer.rendering_options.resolution = 512
    
    # Create a 2x2 grid of rendered views
    image = torch.zeros(3, 1024, 1024).to(device)
    tile = [2, 2]
    
    # Set up camera positions from 4 different angles
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
    yaws = [y + yaws_offset for y in yaws]
    pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

    exts = []
    ints = []

    # Calculate camera parameters for each viewpoint
    for yaw, pitch in zip(yaws, pitch):
        orig = torch.tensor([
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
        ]).float().to(device) * 3
        
        fov = torch.deg2rad(torch.tensor(30)).to(device)
        
        extrinsics = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().to(device),
            torch.tensor([0, 0, 1]).float().to(device)
        )
        
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        
        exts.append(extrinsics)
        ints.append(intrinsics)

    # Render each representation and assemble them into a grid
    for rep_idx, representation in enumerate(representations):
        if not representation.success:
            print("Warning: Mesh extraction was not successful")
            continue
            
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = renderer.render(representation, ext, intr, return_types=['normal'])
            
            # Place each rendered view in the appropriate position in the grid
            image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 
                512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['normal']
    
    # Create output directory if it doesn't exist
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the assembled grid image
        utils.save_image(
            image,
            output_path,
            normalize=True,
            value_range=(0, 1),
        )


def visualize_rf_3d(
    representations: List[Strivec], 
    output_path: Optional[str] = None, 
    device: str = "cuda"
) -> None:
    """
    Visualizes a 3D radiance field representation using OctreeRenderer from multiple viewpoints.
    
    Args:
        representations: List of Strivec representations
        output_path: Path to save the visualization (optional)
        device: Device to perform rendering on
    """
    # Set up the renderer with options
    renderer = OctreeRenderer()
    renderer.pipe.primitive = 'trivec'  # Set primitive type for the renderer
    renderer.rendering_options.resolution = 512
    renderer.rendering_options.near = 0.8
    renderer.rendering_options.far = 1.6
    renderer.rendering_options.bg_color = (1, 1, 1)
    
    # Create a 2x2 grid of rendered views
    image = torch.zeros(3, 1024, 1024).to(device)
    tile = [2, 2]
    
    # Set up camera positions from 4 different angles
    yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
    yaws = [y + yaws_offset for y in yaws]
    pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

    exts = []
    ints = []

    # Calculate camera parameters for each viewpoint
    for yaw, pitch in zip(yaws, pitch):
        orig = torch.tensor([
            np.sin(yaw) * np.cos(pitch),
            np.cos(yaw) * np.cos(pitch),
            np.sin(pitch),
        ]).float().to(device) * 3
        
        fov = torch.deg2rad(torch.tensor(30)).to(device)
        
        extrinsics = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().to(device),
            torch.tensor([0, 0, 1]).float().to(device)
        )
        
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        
        exts.append(extrinsics)
        ints.append(intrinsics)

    # Render each representation and assemble them into a grid
    for rep_idx, representation in enumerate(representations):
        for j, (ext, intr) in enumerate(zip(exts, ints)):
            res = renderer.render(representation, ext, intr)
            
            # Place each rendered view in the appropriate position in the grid
            image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 
                512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
    
    # Create output directory if it doesn't exist
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the assembled grid image
        utils.save_image(
            image,
            output_path,
            normalize=True,
            value_range=(0, 1),
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


def main() -> None:
    """Main execution function for SLAT VAE encoding and decoding."""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the separate configuration files
    print("Loading configuration files...")
    encoder_config = load_config(args.encoder_config_path)
    
    # Create a combined config for model initialization
    combined_config = {"models": {"encoder": encoder_config["models"]["encoder"]}}
    
    # Add decoder configs as needed
    if args.decoders in ["all", "gaussian"]:
        # Gaussian decoder is also in the encoder config file
        combined_config["models"]["decoder_gs"] = encoder_config["models"]["decoder"]
    
    # Load mesh decoder config if needed
    if args.decoders in ["all", "mesh"] and os.path.exists(args.mesh_config_path):
        mesh_config = load_config(args.mesh_config_path)
        combined_config["models"]["decoder_mesh"] = mesh_config["models"]["decoder"]
    
    # Load RF decoder config if needed
    if args.decoders in ["all", "rf"] and os.path.exists(args.rf_config_path):
        rf_config = load_config(args.rf_config_path)
        combined_config["models"]["decoder_rf"] = rf_config["models"]["decoder"]
    
    # Determine which decoders to use
    use_gs_decoder = args.decoders in ["all", "gaussian"] and args.gs_decoder_path
    use_mesh_decoder = args.decoders in ["all", "mesh"] and args.mesh_decoder_path
    use_rf_decoder = args.decoders in ["all", "rf"] and args.rf_decoder_path
    
    # Load encoder and decoders
    encoder, decoders = load_slat_vae_from_config(
        combined_config, 
        encoder_path=args.encoder_path,
        gs_decoder_path=args.gs_decoder_path if use_gs_decoder else None,
        mesh_decoder_path=args.mesh_decoder_path if use_mesh_decoder else None,
        rf_decoder_path=args.rf_decoder_path if use_rf_decoder else None,
        device=args.device
    )
    
    # Rest of the function remains the same
    # Load input data
    input_tensor = None
    if args.input_path and os.path.exists(args.input_path):
        # Check file extension to determine loading method
        if args.input_path.lower().endswith('.ply'):
            voxels = load_ply_voxels(args.input_path, args.resolution, args.device)
            # Convert voxel grid to sparse tensor
            input_tensor = voxels_to_sparse_tensor(voxels, args.device)
        else:
            print(f"Unsupported file format for {args.input_path}")
    
    if input_tensor is None:
        # Generate random data if no input was loaded
        voxels = generate_random_input(
            args.batch_size,
            encoder_config["models"]["encoder"]["args"]["in_channels"],
            args.resolution,
            args.device
        )
        # Convert voxel grid to sparse tensor
        input_tensor = voxels_to_sparse_tensor(voxels, args.device)
    
    # Encode input to latent space
    print("Encoding input to latent space...")
    start_time = time.time()
    with torch.no_grad():
        latent, mean, logvar = encoder(input_tensor, sample_posterior=True, return_raw=True)
    encode_time = time.time() - start_time
    print(f"Encoding completed in {encode_time:.3f} seconds")
    
    # Save latent representation
    latent_path = os.path.join(args.output_dir, "latent.pt")
    torch.save({
        "latent": latent,
        "mean": mean, 
        "logvar": logvar
    }, latent_path)
    print(f"Saved latent representation to {latent_path}")
    
    # Decode with each available decoder and visualize results
    
    # Gaussian decoder
    if "gaussian" in decoders:
        print("\nDecoding with Gaussian decoder...")
        start_time = time.time()
        with torch.no_grad():
            gaussian_reps = decoders["gaussian"](latent)
        decode_time = time.time() - start_time
        print(f"Gaussian decoding completed in {decode_time:.3f} seconds")
        
        # Visualize Gaussian representation
        if args.visualize:
            output_vis_path = os.path.join(args.output_dir, "gaussian_3d_visualization.png")
            visualize_gaussian_3d(gaussian_reps, output_vis_path, device=args.device)
            print(f"Saved Gaussian visualization to {output_vis_path}")
    
    # Mesh decoder
    if "mesh" in decoders:
        print("\nDecoding with Mesh decoder...")
        start_time = time.time()
        with torch.no_grad():
            try:
                mesh_reps = decoders["mesh"](latent)
                decode_time = time.time() - start_time
                print(f"Mesh decoding completed in {decode_time:.3f} seconds")
                
                # Visualize mesh representation
                if args.visualize:
                    output_vis_path = os.path.join(args.output_dir, "mesh_3d_visualization.png")
                    visualize_mesh_3d(mesh_reps, output_vis_path, device=args.device)
                    print(f"Saved Mesh visualization to {output_vis_path}")
            except Exception as e:
                print(f"Error during mesh decoding: {e}")
    
    # Radiance Field decoder
    if "rf" in decoders:
        print("\nDecoding with Radiance Field decoder...")
        start_time = time.time()
        with torch.no_grad():
            try:
                rf_reps = decoders["rf"](latent)
                decode_time = time.time() - start_time
                print(f"Radiance Field decoding completed in {decode_time:.3f} seconds")
                
                # Visualize RF representation
                if args.visualize:
                    output_vis_path = os.path.join(args.output_dir, "rf_3d_visualization.png")
                    visualize_rf_3d(rf_reps, output_vis_path, device=args.device)
                    print(f"Saved Radiance Field visualization to {output_vis_path}")
            except Exception as e:
                print(f"Error during radiance field decoding: {e}")


if __name__ == "__main__":
    main()