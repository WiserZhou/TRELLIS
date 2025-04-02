import os
from PIL import Image
import json
import numpy as np
import torch
import utils3d.torch
from ..modules.sparse.basic import SparseTensor
from .components import StandardDatasetBase


class SLat2Render(StandardDatasetBase):
    """
    Dataset for Structured Latent and rendered images.
    
    Args:
        roots (str): paths to the dataset
        image_size (int): size of the image
        latent_model (str): latent model name
        min_aesthetic_score (float): minimum aesthetic score
        max_num_voxels (int): maximum number of voxels
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
    ):
        # Store configuration parameters
        self.image_size = image_size
        self.latent_model = latent_model
        self.min_aesthetic_score = min_aesthetic_score  # Filter threshold for aesthetic quality
        self.max_num_voxels = max_num_voxels  # Upper limit on voxel count
        self.value_range = (0, 1)  # Normalized pixel value range
        
        # Initialize parent class with the dataset roots
        super().__init__(roots)
        
    def filter_metadata(self, metadata):
        """
        Filter dataset metadata based on defined criteria.
        
        Args:
            metadata: DataFrame containing dataset metadata
            
        Returns:
            tuple: (filtered metadata, statistics dictionary)
        """
        stats = {}
        # Filter samples that have latent representations
        metadata = metadata[metadata[f'latent_{self.latent_model}']]
        stats['With latent'] = len(metadata)
        
        # Filter by aesthetic score threshold
        metadata = metadata[metadata['aesthetic_score'] >= self.min_aesthetic_score]
        stats[f'Aesthetic score >= {self.min_aesthetic_score}'] = len(metadata)
        
        # Filter by voxel count threshold
        metadata = metadata[metadata['num_voxels'] <= self.max_num_voxels]
        stats[f'Num voxels <= {self.max_num_voxels}'] = len(metadata)
        
        return metadata, stats

    def _get_image(self, root, instance):
        """
        Load and process a rendered image for a given instance.
        
        Args:
            root: Dataset root path
            instance: Instance identifier
            
        Returns:
            dict: Dictionary containing image data, camera parameters
        """
        # Load camera transformation metadata
        with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
            metadata = json.load(f)
            
        # Select a random view from available camera angles
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        
        # Extract camera parameters
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1  # Adjust coordinate system
        extrinsics = torch.inverse(c2w)

        # Load and process the image
        image_path = os.path.join(root, 'renders', instance, metadata['file_path'])
        image = Image.open(image_path)
        alpha = image.getchannel(3)  # Extract alpha channel
        image = image.convert('RGB')  # Convert to RGB
        
        # Resize image and alpha channel to target size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Convert to normalized torch tensors
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }
    
    def _get_latent(self, root, instance):
        """
        Load latent representation for a given instance.
        
        Args:
            root: Dataset root path
            instance: Instance identifier
            
        Returns:
            dict: Dictionary containing latent coordinates and features
        """
        # Load latent representation data
        data = np.load(os.path.join(root, 'latents', self.latent_model, f'{instance}.npz'))
        coords = torch.tensor(data['coords']).int()  # Spatial coordinates
        feats = torch.tensor(data['feats']).float()  # Feature vectors
        
        return {
            'coords': coords,
            'feats': feats,
        }

    @torch.no_grad()
    def visualize_sample(self, sample: dict):
        """
        Extract image from sample for visualization.
        
        Args:
            sample: Dictionary containing sample data
            
        Returns:
            torch.Tensor: Image tensor for visualization
        """
        return sample['image']

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for batching samples.
        
        Args:
            batch: List of samples to collate
            
        Returns:
            dict: Collated batch dictionary
        """
        pack = {}
        
        # Prepare sparse tensor from batch of latent representations
        coords = []
        for i, b in enumerate(batch):
            # Add batch index as first dimension to coordinates
            coords.append(torch.cat([torch.full((b['coords'].shape[0], 1), i, dtype=torch.int32), b['coords']], dim=-1))
        coords = torch.cat(coords)
        feats = torch.cat([b['feats'] for b in batch])
        
        # Create sparse tensor representation
        pack['latents'] = SparseTensor(
            coords=coords,
            feats=feats,
        )
        
        # Collate other data fields
        keys = [k for k in batch[0].keys() if k not in ['coords', 'feats']]
        for k in keys:
            if isinstance(batch[0][k], torch.Tensor):
                pack[k] = torch.stack([b[k] for b in batch])
            elif isinstance(batch[0][k], list):
                pack[k] = sum([b[k] for b in batch], [])
            else:
                pack[k] = [b[k] for b in batch]

        return pack

    def get_instance(self, root, instance):
        """
        Load complete data for a specific instance.
        
        Args:
            root: Dataset root path
            instance: Instance identifier
            
        Returns:
            dict: Combined dictionary with image and latent data
        """
        image = self._get_image(root, instance)
        latent = self._get_latent(root, instance)
        return {
            **image,
            **latent,
        }
        

class Slat2RenderGeo(SLat2Render):
    """
    Extended dataset class that includes geometric data alongside
    structured latents and rendered images.
    """
    def __init__(
        self,
        roots: str,
        image_size: int,
        latent_model: str,
        min_aesthetic_score: float = 5.0,
        max_num_voxels: int = 32768,
    ):
        # Initialize parent class with the same parameters
        super().__init__(
            roots,
            image_size,
            latent_model,
            min_aesthetic_score,
            max_num_voxels,
        )
        
    def _get_geo(self, root, instance):
        """
        Load geometric data (mesh) for a given instance.
        
        Args:
            root: Dataset root path
            instance: Instance identifier
            
        Returns:
            dict: Dictionary containing mesh data
        """
        # Load mesh vertices and faces from PLY file
        verts, face = utils3d.io.read_ply(os.path.join(root, 'renders', instance, 'mesh.ply'))
        mesh = {
            "vertices": torch.from_numpy(verts),
            "faces": torch.from_numpy(face),
        }
        return {
            "mesh": mesh,
        }
        
    def get_instance(self, root, instance):
        """
        Load complete data for a specific instance including geometric data.
        
        Args:
            root: Dataset root path
            instance: Instance identifier
            
        Returns:
            dict: Combined dictionary with image, latent, and geometric data
        """
        # Get image and latent data from parent class
        image = self._get_image(root, instance)
        latent = self._get_latent(root, instance)
        # Get additional geometric data
        geo = self._get_geo(root, instance)
        return {
            **image,
            **latent,
            **geo,
        }