from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


BSDF_RENDER_TYPES = {"Base Color", "Roughness", "Metallic", "Normal Map"}
AUXILIARY_RENDER_TYPES = {"Depth", "Normal", "Mask"} | BSDF_RENDER_TYPES
ALL_RENDER_TYPES = {"Color"} | AUXILIARY_RENDER_TYPES
RENDER_TYPE_TO_FILE_EXT = {
    "Color": "webp",
    "Depth": "exr",
    "Base Color": "webp",
    "Roughness": "webp",
    "Metallic": "webp",
    "Normal": "webp",
    "Normal Map": "webp",
    "Mask": "exr",
}
RENDER_TYPE_TO_SAVE_NAME = {
    "Color": "color",
    "Depth": "depth",
    "Base Color": "base_color",
    "Roughness": "roughness",
    "Metallic": "metallic",
    "Normal": "normal",
    "Normal Map": "normal_map",
    "Mask": "mask",
}

@dataclass
class NormalizationSpec:
    scaling_factor: float
    rotation_euler: Tuple[float, float, float]
    translation: Tuple[float, float, float]
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]


@dataclass
class InitializationSettings:
    # Path to the input mesh file
    file_path: str
    # Format of the input file (e.g., 'obj', 'fbx', etc.). If None, will be inferred from extension
    file_format: Optional[str] = None
    # Axis pointing forward in the coordinate system ('X', 'Y', 'Z', 'NEGATIVE_X', etc.)
    forward_axis: str = "NEGATIVE_Z"
    # Axis pointing upward in the coordinate system ('X', 'Y', 'Z', 'NEGATIVE_X', etc.)
    up_axis: str = "Y"
    # Whether to merge duplicate vertices during mesh import
    merge_vertices: bool = True
    # Method to compute bounding box ('bound_box' or 'render_box')
    bbox_compute_method: str = "bound_box"
    # Scale factor for normalizing the model size
    normalizing_scale: float = 0.5
    # Rotation angles in radians around x, y, z axes to be applied during initialization
    rotation_euler: Tuple[float, float, float] = (0, 0, 0)
    # Default animation frame to use for static models
    default_frame: int = 1
    # Whether to clear existing normal map data during initialization
    clear_normal_map: bool = False


@dataclass
class InitializationOutput:
    normalization_spec: NormalizationSpec
    valid_render_types: set[str]
    index_to_name: dict[int, str]

@dataclass
class RuntimeSettings:
    # environment map
    use_environment_map: bool = False
    environment_map_path: Optional[str] = None
    light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    light_strength: float = 1.0
    environment_map_rotation_deg: float = 0.

    # animation
    frame_index: int = 1

    # renderer
    engine: str = "BLENDER_EEVEE"
    # for BLENDER_EEVEE
    taa_render_samples: int = 64 # Sampling -> Render
    use_gtao: bool = False # Ambient Occlusion
    gtao_distance: float = 0.2 # Ambient Occlusion -> Distance
    gtao_factor: float = 1.0 # Ambient Occlusion -> Factor
    gtao_quality: float = 0.25 # Ambient Occlusion -> Trace Precision
    use_gtao_bent_normals: bool = True # Ambient Occlusion -> Bent Normals
    use_gtao_bounce: bool = True # Ambient Occlusion -> Bounces Approximation
    use_ssr: bool = False # Screen Space Reflections
    use_ssr_refraction: bool = False # Screen Space Reflections -> Refraction
    # for CYCLES
    cycles_device: str = "NONE" # Device
    cycles_compute_backend: str = "NONE"
    preview_samples: int = 1 # Sampling -> Viewport -> Max Samples
    render_samples: int = 128 # Sampling -> Render -> Max Samples
    use_denoising: bool = True # Sampling -> Denoise
    max_bounces: int = 12 # Light Paths -> Max Bounces -> Total
    diffuse_bounces: int = 4 # Light Paths -> Max Bounces -> Diffuse
    glossy_bounces: int = 4 # Light Paths -> Max Bounces -> Glossy
    transmission_bounces: int = 12 # Light Paths -> Max Bounces -> Transmission
    volume_bounces: int = 0 # Light Paths -> Max Bounces -> Volume
    transparent_max_bounces: int = 8 # Light Paths -> Max Bounces -> Transparent
    caustics_reflective: bool = True # Light Paths -> Caustics -> Reflective
    caustics_refractive: bool = True # Light Paths -> Caustics -> Refractive

    use_high_quality_normals: bool = False # Performance -> High Quality Normals
    film_transparent: bool = True # Film -> Transparent    

    # auto smooth
    use_auto_smooth: bool = False
    auto_smooth_angle_deg: float = 30.
    # material blend and shadow
    blend_mode: Optional[str] = None
    shadow_mode: Optional[str] = None
    show_transparent_back: bool = False

    # resolution
    resolution_x: int = 1024
    resolution_y: int = 1024


@dataclass
class RenderOutput:
    render_type: str
    file_path: str


@dataclass
class CameraSpec:
    projection_type: str
    transform_matrix: np.ndarray
    # for perspective camera
    focal_length_mm: Optional[float] = None
    fov_deg: Optional[float] = None
    # for othographic camera
    ortho_scale: Optional[float] = None
    near_clip: float = 0.1
    far_clip: float = 100

