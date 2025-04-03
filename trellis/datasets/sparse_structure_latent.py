'''
sparse_structure_latent.py - Sparse Structure Latent Datasets for 3D Generation

This file defines dataset classes for handling sparse structure latent representations in the TRELLIS framework.
It provides functionality for loading, processing, and visualizing sparse structure latents, which are compact
representations of 3D structures. The file includes:
1. SparseStructureLatentVisMixin - Visualization capabilities for sparse structure latents
2. SparseStructureLatent - Base dataset class for sparse structure latents
3. TextConditionedSparseStructureLatent - Dataset for text-to-3D generation
4. ImageConditionedSparseStructureLatent - Dataset for image-to-3D generation
'''

import os
import json
from typing import *
import numpy as np
import torch
import utils3d
from ..representations.octree import DfsOctree as Octree
from ..renderers import OctreeRenderer
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from .. import models
from ..utils.dist_utils import read_file_dist


class SparseStructureLatentVisMixin:
    """
    A mixin class that provides visualization capabilities for sparse structure latents.
    It handles loading decoders and rendering the decoded structures.
    """
    def __init__(
        self,
        *args,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        # Initialize with decoder configuration options
        super().__init__(*args, **kwargs)
        self.ss_dec = None  # Decoder model will be loaded on demand
        self.pretrained_ss_dec = pretrained_ss_dec  # Path to pretrained decoder
        self.ss_dec_path = ss_dec_path  # Optional custom decoder path
        self.ss_dec_ckpt = ss_dec_ckpt  # Optional custom decoder checkpoint name
        
    def _loading_ss_dec(self):
        """
        Lazy-loads the sparse structure decoder when needed.
        Prioritizes loading from a specified path if provided, otherwise uses the pretrained model.
        """
        if self.ss_dec is not None:
            return
        if self.ss_dec_path is not None:
            # Load decoder from a specific path and checkpoint
            cfg = json.load(open(os.path.join(self.ss_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.ss_dec_path, 'ckpts', f'decoder_{self.ss_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(read_file_dist(ckpt_path), map_location='cpu', weights_only=True))
        else:
            # Load from pretrained model repository
            decoder = models.from_pretrained(self.pretrained_ss_dec)
        self.ss_dec = decoder.cuda().eval()  # Move to GPU and set to evaluation mode

    def _delete_ss_dec(self):
        """
        Frees GPU memory by deleting the decoder when it's no longer needed.
        """
        del self.ss_dec
        self.ss_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        """
        Decodes latent vectors into sparse structure representations.
        
        Args:
            z: The latent vectors to decode
            batch_size: Batch size for decoding to manage memory usage
            
        Returns:
            Decoded sparse structure tensors
        """
        self._loading_ss_dec()  # Load decoder if not already loaded
        ss = []
        if self.normalization is not None:
            # Denormalize the latents if normalization was applied
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        # Process in batches to avoid OOM errors
        for i in range(0, z.shape[0], batch_size):
            ss.append(self.ss_dec(z[i:i+batch_size]))
        ss = torch.cat(ss, dim=0)
        self._delete_ss_dec()  # Free memory
        return ss

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[torch.Tensor, dict]):
        """
        Visualizes a sample by decoding it and rendering from multiple viewpoints.
        
        Args:
            x_0: Latent tensor or dictionary containing latent tensor
            
        Returns:
            Tensor of rendered images from multiple viewpoints
        """
        # Extract and decode the latent
        x_0 = x_0 if isinstance(x_0, torch.Tensor) else x_0['x_0']
        x_0 = self.decode_latent(x_0.cuda())
        
        # Set up the renderer with options
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = 512
        renderer.rendering_options.near = 0.8
        renderer.rendering_options.far = 1.6
        renderer.rendering_options.bg_color = (0, 0, 0)
        renderer.rendering_options.ssaa = 4  # Super sampling anti-aliasing
        renderer.pipe.primitive = 'voxel'
        
        # Set up camera positions from 4 different angles
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 0, 90, 180, 270 degrees
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)  # Add some randomness
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]  # Random pitch for each view

        exts = []  # Extrinsics (camera position and orientation)
        ints = []  # Intrinsics (camera internal parameters)
        for yaw, pitch in zip(yaws, pitch):
            # Calculate camera origin based on spherical coordinates
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(30)).cuda()  # 30 degree field of view
            # Create camera extrinsics and intrinsics
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        images = []
        
        # Process each structure in the batch
        x_0 = x_0.cuda()
        for i in range(x_0.shape[0]):
            # Create an octree representation
            representation = Octree(
                depth=10,
                aabb=[-0.5, -0.5, -0.5, 1, 1, 1],  # Axis-aligned bounding box
                device='cuda',
                primitive='voxel',
                sh_degree=0,
                primitive_config={'solid': True},
            )
            # Extract coordinates of occupied voxels
            coords = torch.nonzero(x_0[i, 0] > 0, as_tuple=False)
            resolution = x_0.shape[-1]
            representation.position = coords.float() / resolution  # Normalize positions
            # Set depth of each voxel based on resolution
            representation.depth = torch.full((representation.position.shape[0], 1), int(np.log2(resolution)), dtype=torch.uint8, device='cuda')

            # Create a 2x2 grid of rendered views
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr, colors_overwrite=representation.position)
                # Place each rendered view in the appropriate position in the grid
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
            
        return torch.stack(images)
       

class SparseStructureLatent(SparseStructureLatentVisMixin, StandardDatasetBase):
    """
    Sparse structure latent dataset
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        normalization (dict): normalization stats
        pretrained_ss_dec (str): name of the pretrained sparse structure decoder
        ss_dec_path (str): path to the sparse structure decoder, if given, will override the pretrained_ss_dec
        ss_dec_ckpt (str): name of the sparse structure decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        normalization: Optional[dict] = None,
        pretrained_ss_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16',
        ss_dec_path: Optional[str] = None,
        ss_dec_ckpt: Optional[str] = None,
    ):
        # Configure dataset parameters
        self.latent_model = latent_model  # Which latent model to use
        self.min_aesthetic_score = min_aesthetic_score  # Quality threshold
        self.normalization = normalization  # Stats for normalizing latents
        self.value_range = (0, 1)  # Expected value range
        
        super().__init__(
            roots,
            pretrained_ss_dec=pretrained_ss_dec,
            ss_dec_path=ss_dec_path,
            ss_dec_ckpt=ss_dec_ckpt,
        )
        
        # Set up normalization tensors if provided
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(-1, 1, 1, 1)
            self.std = torch.tensor(self.normalization['std']).reshape(-1, 1, 1, 1)
  
    def filter_metadata(self, metadata):
        """
        Filters dataset metadata based on criteria like existence of latents and aesthetic score.
        
        Args:
            metadata: DataFrame containing dataset metadata
            
        Returns:
            Tuple of (filtered metadata, statistics dictionary)
        """
        stats = {}
        # Filter for entries that have sparse structure latents
        metadata = metadata[metadata[f'ss_latent_{self.latent_model}']]
        stats['With sparse structure latents'] = len(metadata)
        # Filter for entries with high enough aesthetic scores
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        return metadata, stats
                
    def get_instance(self, root, instance):
        """
        Loads a specific instance from the dataset.
        
        Args:
            root: Dataset root path
            instance: Instance identifier
            
        Returns:
            Dictionary containing the latent tensor
        """
        # Load the latent vector from disk
        latent = np.load(os.path.join(root, 'ss_latents', self.latent_model, f'{instance}.npz'))
        z = torch.tensor(latent['mean']).float()
        # Normalize if required
        if self.normalization is not None:
            z = (z - self.mean) / self.std

        pack = {
            'x_0': z,
        }
        return pack
    

class TextConditionedSparseStructureLatent(TextConditionedMixin, SparseStructureLatent):
    """
    Text-conditioned sparse structure dataset
    
    Extends the base SparseStructureLatent dataset with text conditioning capabilities,
    allowing for text-to-3D generation tasks.
    """
    pass


class ImageConditionedSparseStructureLatent(ImageConditionedMixin, SparseStructureLatent):
    """
    Image-conditioned sparse structure dataset
    
    Extends the base SparseStructureLatent dataset with image conditioning capabilities,
    allowing for image-to-3D generation tasks.
    """
    pass