"""
This file defines dataset classes for handling structured latent representations (SLat) in the TRELLIS framework.
It provides functionality for loading, processing, and visualizing 3D structures represented as sparse tensors.
The file includes classes for standard datasets as well as text-conditioned and image-conditioned variants.
"""

import json
import os
from typing import *
import numpy as np
import torch
import utils3d.torch
from .components import StandardDatasetBase, TextConditionedMixin, ImageConditionedMixin
from ..modules.sparse.basic import SparseTensor
from .. import models
from ..utils.render_utils import get_renderer
from ..utils.dist_utils import read_file_dist
from ..utils.data_utils import load_balanced_group_indices


class SLatVisMixin:
    """
    Mixin class that adds visualization capabilities for structured latent representations.
    Handles loading of latent decoders and rendering 3D structures from latent codes.
    """
    def __init__(
        self,
        *args,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the visualization mixin.
        
        Args:
            pretrained_slat_dec: Identifier for pretrained decoder model
            slat_dec_path: Optional path to custom decoder model
            slat_dec_ckpt: Optional checkpoint name for custom decoder
        """
        super().__init__(*args, **kwargs)
        self.slat_dec = None  # Decoder model (loaded on demand)
        self.pretrained_slat_dec = pretrained_slat_dec
        self.slat_dec_path = slat_dec_path
        self.slat_dec_ckpt = slat_dec_ckpt
        
    def _loading_slat_dec(self):
        """
        Load the structured latent decoder model if not already loaded.
        Uses either a custom path or pretrained model based on initialization parameters.
        """
        if self.slat_dec is not None:
            return
        if self.slat_dec_path is not None:
            # Load from custom path
            cfg = json.load(open(os.path.join(self.slat_dec_path, 'config.json'), 'r'))
            decoder = getattr(models, cfg['models']['decoder']['name'])(**cfg['models']['decoder']['args'])
            ckpt_path = os.path.join(self.slat_dec_path, 'ckpts', f'decoder_{self.slat_dec_ckpt}.pt')
            decoder.load_state_dict(torch.load(read_file_dist(ckpt_path), map_location='cpu', weights_only=True))
        else:
            # Load pretrained model
            decoder = models.from_pretrained(self.pretrained_slat_dec)
        self.slat_dec = decoder.cuda().eval()

    def _delete_slat_dec(self):
        """
        Delete the decoder model to free up memory.
        """
        del self.slat_dec
        self.slat_dec = None

    @torch.no_grad()
    def decode_latent(self, z, batch_size=4):
        """
        Decode latent vectors into 3D representations.
        
        Args:
            z: Latent vectors to decode
            batch_size: Batch size for processing
            
        Returns:
            List of 3D representations
        """
        self._loading_slat_dec()
        reps = []
        if self.normalization is not None:
            # Apply normalization if needed
            z = z * self.std.to(z.device) + self.mean.to(z.device)
        for i in range(0, z.shape[0], batch_size):
            reps.append(self.slat_dec(z[i:i+batch_size]))
        reps = sum(reps, [])
        self._delete_slat_dec()
        return reps

    @torch.no_grad()
    def visualize_sample(self, x_0: Union[SparseTensor, dict]):
        """
        Generate multi-view renderings of a 3D representation.
        
        Args:
            x_0: Input sparse tensor or dictionary containing sparse tensor
            
        Returns:
            Tensor of rendered images from multiple viewpoints
        """
        x_0 = x_0 if isinstance(x_0, SparseTensor) else x_0['x_0']
        reps = self.decode_latent(x_0.cuda())
        
        # Build camera parameters for multiple viewpoints
        yaws = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
        yaws_offset = np.random.uniform(-np.pi / 4, np.pi / 4)
        yaws = [y + yaws_offset for y in yaws]
        pitch = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(4)]

        exts = []
        ints = []
        for yaw, pitch in zip(yaws, pitch):
            # Calculate camera position based on spherical coordinates
            orig = torch.tensor([
                np.sin(yaw) * np.cos(pitch),
                np.cos(yaw) * np.cos(pitch),
                np.sin(pitch),
            ]).float().cuda() * 2
            fov = torch.deg2rad(torch.tensor(40)).cuda()
            extrinsics = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
            exts.append(extrinsics)
            ints.append(intrinsics)

        # Render images for each representation
        renderer = get_renderer(reps[0])
        images = []
        for representation in reps:
            image = torch.zeros(3, 1024, 1024).cuda()
            tile = [2, 2]
            for j, (ext, intr) in enumerate(zip(exts, ints)):
                res = renderer.render(representation, ext, intr)
                # Place each view in a grid position
                image[:, 512 * (j // tile[1]):512 * (j // tile[1] + 1), 512 * (j % tile[1]):512 * (j % tile[1] + 1)] = res['color']
            images.append(image)
        images = torch.stack(images)
            
        return images
    
    
class SLat(SLatVisMixin, StandardDatasetBase):
    """
    Structured latent dataset class.
    Handles loading and processing of structured latent representations.
    
    Args:
        roots (str): path to the dataset
        latent_model (str): name of the latent model
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
        normalization (dict): normalization stats
        pretrained_slat_dec (str): name of the pretrained slat decoder
        slat_dec_path (str): path to the slat decoder, if given, will override the pretrained_slat_dec
        slat_dec_ckpt (str): name of the slat decoder checkpoint
    """
    def __init__(self,
        roots: str,
        *,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16',
        slat_dec_path: Optional[str] = None,
        slat_dec_ckpt: Optional[str] = None,
    ):
        """
        Initialize the structured latent dataset.
        """
        self.normalization = normalization
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score
        self.max_num_voxels = max_num_voxels
        self.value_range = (0, 1)
        
        super().__init__(
            roots,
            pretrained_slat_dec=pretrained_slat_dec,
            slat_dec_path=slat_dec_path,
            slat_dec_ckpt=slat_dec_ckpt,
        )

        # Store voxel counts for each instance
        self.loads = [self.metadata.loc[sha256, 'num_voxels'] for _, sha256 in self.instances]
        
        # Set up normalization parameters if provided
        if self.normalization is not None:
            self.mean = torch.tensor(self.normalization['mean']).reshape(1, -1)
            self.std = torch.tensor(self.normalization['std']).reshape(1, -1)
      
    def filter_metadata(self, metadata):
        """
        Filter dataset metadata based on criteria.
        
        Args:
            metadata: DataFrame containing dataset metadata
            
        Returns:
            Tuple of (filtered metadata, statistics)
        """
        stats = {}
        # Filter instances with available latent representation
        metadata = metadata[metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)
        # Filter based on aesthetic score
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        # Filter based on voxel count
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        return metadata, stats

    def get_instance(self, root, instance):
        """
        Load a specific instance from the dataset.
        
        Args:
            root: Dataset root path
            instance: Instance identifier
            
        Returns:
            Dictionary containing coordinates and features
        """
        data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()
        feats = torch.tensor(data['feats']).float()
        if self.normalization is not None:
            # Apply normalization if needed
            feats = (feats - self.mean) / self.std
        return {
            'coords': coords,
            'feats': feats,
        }
        
    @staticmethod
    def collate_fn(batch, split_size=None):
        """
        Collate function for creating batches from individual samples.
        Handles sparse tensor construction and layout information.
        
        Args:
            batch: List of data samples
            split_size: Optional size for splitting large batches
            
        Returns:
            Collated batch or list of batches if split_size is provided
        """
        if split_size is None:
            group_idx = [list(range(len(batch)))]
        else:
            # Create balanced groups based on voxel counts
            group_idx = load_balanced_group_indices([b['coords'].shape[0] for b in batch], split_size)
        packs = []
        for group in group_idx:
            sub_batch = [batch[i] for i in group]
            pack = {}
            coords = []
            feats = []
            layout = []
            start = 0
            # Combine coordinates and features from all samples
            for i, b in enumerate(sub_batch):
                coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
                feats.append(b['feats'])
                layout.append(slice(start, start + b['coords'].shape[0]))
                start += b['coords'].shape[0]
            coords = torch.cat(coords)
            feats = torch.cat(feats)
            # Create sparse tensor
            pack['x_0'] = SparseTensor(
                coords=coords,
                feats=feats,
            )
            pack['x_0']._shape = torch.Size([len(group), *sub_batch[0]['feats'].shape[1:]])
            pack['x_0'].register_spatial_cache('layout', layout)
            
            # Collate other data fields
            keys = [k for k in sub_batch[0].keys() if k not in ['coords', 'feats']]
            for k in keys:
                if isinstance(sub_batch[0][k], torch.Tensor):
                    pack[k] = torch.stack([b[k] for b in sub_batch])
                elif isinstance(sub_batch[0][k], list):
                    pack[k] = sum([b[k] for b in sub_batch], [])
                else:
                    pack[k] = [b[k] for b in sub_batch]
                    
            packs.append(pack)
          
        if split_size is None:
            return packs[0]
        return packs
        
    
class TextConditionedSLat(TextConditionedMixin, SLat):
    """
    Text conditioned structured latent dataset.
    Extends the base SLat class with text conditioning capabilities.
    """
    pass


class ImageConditionedSLat(ImageConditionedMixin, SLat):
    """
    Image conditioned structured latent dataset.
    Extends the base SLat class with image conditioning capabilities.
    """
    pass
