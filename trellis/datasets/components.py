from typing import *  # Import all type annotations
from abc import abstractmethod  # Import for creating abstract methods
import os  # Operating system functionality
import json  # JSON parsing utilities
import torch  # PyTorch deep learning framework
import numpy as np  # Numerical computation library
import pandas as pd  # Data manipulation and analysis library
from PIL import Image  # Python Imaging Library for image processing
from torch.utils.data import Dataset  # Base PyTorch dataset class


class StandardDatasetBase(Dataset):
    """
    Base class for standard datasets.
    
    This class provides the foundation for dataset handling with common functionality
    like loading metadata from multiple data roots and filtering instances.

    Args:
        roots (str): Comma-separated paths to the dataset directories
    """

    def __init__(self,
        roots: str,
    ):
        super().__init__()
        self.roots = roots.split(',')  # Split the comma-separated paths into a list
        self.instances = []  # List to store dataset instances
        self.metadata = pd.DataFrame()  # DataFrame to store metadata for all instances
        
        # Dictionary to store statistics about each dataset root
        self._stats = {}
        for root in self.roots:
            key = os.path.basename(root)  # Use the directory name as the key
            self._stats[key] = {}
            metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))  # Load metadata file
            self._stats[key]['Total'] = len(metadata)  # Record total number of instances
            
            # Filter metadata according to implementation-specific criteria
            metadata, stats = self.filter_metadata(metadata)
            self._stats[key].update(stats)  # Update statistics with filter results
            
            # Add instances to the dataset with their root path
            self.instances.extend([(root, sha256) for sha256 in metadata['sha256'].values])
            
            # Set SHA256 as index and merge with existing metadata
            metadata.set_index('sha256', inplace=True)
            self.metadata = pd.concat([self.metadata, metadata])

    @abstractmethod
    def filter_metadata(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Filter metadata according to implementation-specific criteria.
        
        Args:
            metadata (pd.DataFrame): The raw metadata to filter
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]: Filtered metadata and statistics dictionary
        """
        pass
    
    @abstractmethod
    def get_instance(self, root: str, instance: str) -> Dict[str, Any]:
        """
        Get a specific instance from the dataset.
        
        Args:
            root (str): The root directory of the instance
            instance (str): The instance identifier (SHA256)
            
        Returns:
            Dict[str, Any]: Dictionary containing the instance data
        """
        pass
        
    def __len__(self):
        """Return the total number of instances in the dataset"""
        return len(self.instances)

    def __getitem__(self, index) -> Dict[str, Any]:
        """
        Get an item from the dataset by index.
        
        Includes error handling - if fetching an instance fails, returns a random instance.
        
        Args:
            index: Index of the item to retrieve
            
        Returns:
            Dict[str, Any]: Dictionary containing the instance data
        """
        try:
            root, instance = self.instances[index]
            return self.get_instance(root, instance)
        except Exception as e:
            print(e)
            # On error, return a random instance instead
            return self.__getitem__(np.random.randint(0, len(self)))
        
    def __str__(self):
        """
        Return a string representation of the dataset, including statistics.
        
        Returns:
            str: Formatted dataset information
        """
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(f'  - Total instances: {len(self)}')
        lines.append(f'  - Sources:')
        for key, stats in self._stats.items():
            lines.append(f'    - {key}:')
            for k, v in stats.items():
                lines.append(f'      - {k}: {v}')
        return '\n'.join(lines)


class TextConditionedMixin:
    """
    Mixin class for datasets conditioned on text/captions.
    
    Adds functionality for loading and accessing text captions associated with instances.
    """
    def __init__(self, roots, **kwargs):
        super().__init__(roots, **kwargs)
        self.captions = {}  # Dictionary to store captions for each instance
        
        # Load captions for all instances from metadata
        for instance in self.instances:
            sha256 = instance[1]
            self.captions[sha256] = json.loads(self.metadata.loc[sha256]['captions'])
    
    def filter_metadata(self, metadata):
        """
        Filter metadata to only include instances with captions.
        
        Args:
            metadata (pd.DataFrame): The metadata to filter
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]: Filtered metadata and updated statistics
        """
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata['captions'].notna()]  # Keep only instances with captions
        stats['With captions'] = len(metadata)  # Record count of instances with captions
        return metadata, stats
    
    def get_instance(self, root, instance):
        """
        Extends the base get_instance method to include a randomly chosen caption.
        
        Args:
            root (str): The root directory of the instance
            instance (str): The instance identifier (SHA256)
            
        Returns:
            Dict[str, Any]: Dictionary with instance data plus caption as 'cond'
        """
        pack = super().get_instance(root, instance)
        text = np.random.choice(self.captions[instance])  # Select a random caption
        pack['cond'] = text  # Add the caption as a condition
        return pack
    
    
class ImageConditionedMixin:
    """
    Mixin class for datasets conditioned on images.
    
    Adds functionality for loading and preprocessing conditional images.
    """
    def __init__(self, roots, *, image_size=518, **kwargs):
        """
        Initialize the mixin with image size parameter.
        
        Args:
            roots (str): Dataset roots
            image_size (int): Size to resize images to (square)
            **kwargs: Additional arguments to pass to parent
        """
        self.image_size = image_size
        super().__init__(roots, **kwargs)
    
    def filter_metadata(self, metadata):
        """
        Filter metadata to only include instances with rendered conditional images.
        
        Args:
            metadata (pd.DataFrame): The metadata to filter
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, int]]: Filtered metadata and updated statistics
        """
        metadata, stats = super().filter_metadata(metadata)
        metadata = metadata[metadata[f'cond_rendered']]  # Keep only instances with conditional renders
        stats['Cond rendered'] = len(metadata)  # Record count of instances with conditional renders
        return metadata, stats
    
    def get_instance(self, root, instance):
        """
        Extends the base get_instance method to include a processed conditional image.
        
        Includes sophisticated image preprocessing:
        1. Loads a random view from available renders
        2. Crops the image around the non-transparent content
        3. Resizes to the target resolution
        4. Applies alpha compositing
        
        Args:
            root (str): The root directory of the instance
            instance (str): The instance identifier (SHA256)
            
        Returns:
            Dict[str, Any]: Dictionary with instance data plus image as 'cond'
        """
        # Call the second father class from MRO sequence to get the base instance
        pack = super().get_instance(root, instance)

        # Load conditional image metadata
        image_root = os.path.join(root, 'renders_cond', instance)
        with open(os.path.join(image_root, 'transforms.json')) as f:
            metadata = json.load(f)
        
        # Select a random view from available frames
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]

        # Load the image from the selected view
        image_path = os.path.join(image_root, metadata['file_path'])
        image = Image.open(image_path)  # Opens RGBA image

        # Calculate bounding box from alpha channel to center the object
        alpha = np.array(image.getchannel(3))
        bbox = np.array(alpha).nonzero()
        bbox = [bbox[1].min(), bbox[0].min(), bbox[1].max(), bbox[0].max()]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        # Calculate padding for augmentation
        hsize = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        aug_size_ratio = 1.2  # Add 20% padding around the object
        aug_hsize = hsize * aug_size_ratio
        aug_center_offset = [0, 0]  # No offset from center
        aug_center = [center[0] + aug_center_offset[0], center[1] + aug_center_offset[1]]
        
        # Crop image to the augmented bounding box
        aug_bbox = [int(aug_center[0] - aug_hsize), int(aug_center[1] - aug_hsize), 
                   int(aug_center[0] + aug_hsize), int(aug_center[1] + aug_hsize)]
        image = image.crop(aug_bbox)

        # Resize image to target size
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Extract alpha channel and convert image to RGB
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        
        # Convert to PyTorch tensors and normalize
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        # Apply alpha compositing
        image = image * alpha.unsqueeze(0)
        pack['cond'] = image

        return pack