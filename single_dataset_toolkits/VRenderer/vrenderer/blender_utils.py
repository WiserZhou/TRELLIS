from typing import Optional, List, Tuple, Dict
import os
import sys
import math
import shutil
import numpy as np
from contextlib import contextmanager
import argparse
import uuid

import bpy
from mathutils import Vector, Matrix

from .spec import (
    ALL_RENDER_TYPES,
    BSDF_RENDER_TYPES,
    AUXILIARY_RENDER_TYPES,
    RENDER_TYPE_TO_FILE_EXT,
    RENDER_TYPE_TO_SAVE_NAME,
    NormalizationSpec,
    RenderOutput,
)
from .ops import transform_points



class BlenderArgumentParser(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash()) 


@contextmanager
def use_single_sample_renderer():
    scene = bpy.context.scene
    if scene.render.engine == "BLENDER_EEVEE":
        taa_render_samples = scene.eevee.taa_render_samples
        scene.eevee.taa_render_samples = 1
        try:
            yield
        finally:
            scene.eevee.taa_render_samples = taa_render_samples
    elif scene.render.engine == "CYCLES":
        render_samples = scene.cycles.samples
        scene.cycles.samples = 1
        try:
            yield
        finally:
            scene.cycles.samples = render_samples
    else:
        raise NotImplementedError


@contextmanager
def use_opaque_material():
    material_blend_method = []
    for material in bpy.data.materials:
        material.use_nodes = True
        material_blend_method.append((material, material.blend_method))
    try:
        for material in bpy.data.materials:
            material.blend_method = 'OPAQUE'
        yield
    finally:
        for material, blend_method in material_blend_method:
            material.blend_method = blend_method


@contextmanager
def use_flat_shading(
    settings_use_auto_smooth,
    settings_auto_smooth_angle_deg
):
    try:
        set_shading_type(use_auto_smooth=False)
        yield
    finally:
        set_shading_type(
            use_auto_smooth=settings_use_auto_smooth,
            auto_smooth_angle_deg=settings_auto_smooth_angle_deg
        )


def make_link_func(node_tree):
    def link(from_socket, to_socket):
        return node_tree.links.new(from_socket, to_socket)
    return link


def set_shading_type(
    use_auto_smooth: bool,
    auto_smooth_angle_deg: float = 30.
):
    for obj in scene_mesh_objects():
        """
        # suppose to work on 4.1, but doesn't
        with object_context(obj):        
            if use_auto_smooth:
                bpy.ops.object.shade_smooth_by_angle(angle=auto_smooth_angle_deg * math.pi / 180.)
            else:
                bpy.ops.object.shade_flat()
        """
        if use_auto_smooth:
            obj.data.shade_smooth()
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(auto_smooth_angle_deg)
        else:
            obj.data.shade_flat()
            obj.data.use_auto_smooth = False


def clear_scene():
    """
    Clears the entire Blender scene by removing all objects and nodes.

    This function performs two main operations:
    1. Deletes all objects in the scene by first selecting everything and then deleting the selection
    2. Removes all nodes from the scene's node tree

    The function requires an active Blender context and will affect the current scene.

    Note:
        - This is a destructive operation that cannot be undone
        - After execution, the scene will be completely empty
        - The scene's use_nodes property will be set to True

    Returns:
        None
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()  

    bpy.context.scene.use_nodes = True
    scene_node_tree = bpy.context.scene.node_tree
    for node in scene_node_tree.nodes:
        scene_node_tree.nodes.remove(node)


def load_model(
    file_path: str,
    file_format: Optional[str] = None,
    forward_axis: str = "NEGATIVE_Z",  # The forward direction when importing the model
    up_axis: str = "Y",  # The up direction when importing the model 
    merge_vertices: bool = True,  # Whether to merge duplicate vertices during import
    # normalization parameters
    bbox_compute_method: str = "bound_box",  # Method to compute bounding box: "bound_box" or "vertex"
    normalizing_scale: float = 0.5,  # Scale factor for normalization (half of target object dimension)
    rotation_euler: Tuple[float, float, float] = (0, 0, 0),  # Rotation angles in radians to apply after normalization
    default_frame: int = 1,  # Default animation frame to use
    # Part selection parameters
    part_names: Optional[List[str]] = None,  # List of part names to keep
    part_filter: Optional[callable] = None,  # Custom filter function for objects
) -> NormalizationSpec:
    """
    Load a 3D model file into Blender and normalize its scale/position.
    Can selectively load parts of the model based on name or custom filter.

    Args:
        file_path: Path to the model file
        file_format: Format of the model file (glb/obj/ply/vrm). If None, inferred from file extension
        forward_axis: Forward axis direction for import
        up_axis: Up axis direction for import 
        merge_vertices: Whether to merge duplicate vertices during import
        bbox_compute_method: Method to compute bounding box for normalization
        normalizing_scale: Target scale after normalization (half of object dimension)
        rotation_euler: Rotation to apply after normalization
        default_frame: Animation frame to use
        part_names: List of part names to keep (others will be removed)
        part_filter: Custom function taking a Blender object and returning True if it should be kept

    Returns:
        NormalizationSpec: Contains normalization parameters like scale factor, rotation, translation
    """
    # Infer file format from extension if not specified
    if file_format is None:
        file_format = os.path.basename(file_path).split('.')[-1]

    # Store objects before import to identify new objects
    objects_before = set(bpy.data.objects)

    # Import model based on file format
    if file_format == "glb":
        bpy.ops.import_scene.gltf(filepath=file_path, merge_vertices=merge_vertices)
    elif file_format == "obj":
        # OBJ files use Z-up convention
        up_axis = "Z"
        bpy.ops.wm.obj_import(filepath=file_path, 
                             directory=os.path.dirname(file_path),
                             forward_axis=forward_axis, 
                             up_axis=up_axis)
    elif file_format == "ply":
        bpy.ops.wm.ply_import(filepath=file_path, 
                             directory=os.path.dirname(file_path),
                             forward_axis=forward_axis, 
                             up_axis=up_axis)
        # Setup vertex color material for PLY files
        preprocess_ply(bpy.context.selected_objects[0])
    elif file_format == "vrm":
        bpy.ops.import_scene.vrm(filepath=file_path)   
    else:
        raise NotImplementedError
        
    # Identify newly imported objects
    new_objects = [obj for obj in bpy.data.objects if obj not in objects_before]
    
    # Filter objects based on part names or custom filter
    objects_to_remove = []
    for obj in new_objects:
        keep_object = True
        
        if part_names is not None:
            # Keep only objects with names in part_names
            if obj.name not in part_names:
                keep_object = False
                
        if part_filter is not None:
            # Apply custom filter function
            keep_object = part_filter(obj)
            
        if not keep_object:
            objects_to_remove.append(obj)
    
    # Remove filtered objects
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Set animation frame
    set_current_frame(default_frame)
    
    # Normalize model scale and position
    normalization_spec = normalize(bbox_compute_method, normalizing_scale, rotation_euler)

    return normalization_spec


def preprocess_ply(obj):
    # TODO: check this
    material = bpy.data.materials.new(name="VertexColors")
    material.use_nodes = True
    obj.data.materials.append(material)

    nodes = material.node_tree.nodes
    links = material.node_tree.links

    principled_bsdf_node = nodes.get("Principled BSDF")
    if principled_bsdf_node:
        nodes.remove(principled_bsdf_node)

    emission_node = nodes.new(type="ShaderNodeEmission")
    emission_node.location = 0, 0

    attribute_node = nodes.new(type="ShaderNodeAttribute")
    attribute_node.location = -300, 0
    attribute_node.attribute_name = "Col"  # 顶点颜色属性名称

    output_node = nodes.get("Material Output")

    links.new(attribute_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], output_node.inputs["Surface"])
 

def set_file_output_non_color(node):
    if int(bpy.app.version_string[0]) >= 4:
        node.format.color_management = "OVERRIDE"
        node.format.view_settings.view_transform = "Raw"
    else:
        node.format.color_management = "OVERRIDE"
        node.format.display_settings.display_device = "None" 

def make_normal_to_rgb_node_group(node_tree, editor_type):
    assert editor_type in ["Shader", "Compositor"]
    link = make_link_func(node_tree)
    sep_color_node = node_tree.nodes.new(f"{editor_type}NodeSeparateColor")
    def create_normal_to_rgb_map_node():
        node = node_tree.nodes.new(f"{editor_type}NodeMapRange")
        if editor_type == "Shader":
            node.clamp = True # Clamp
        elif editor_type == "Compositor":
            node.use_clamp = True # Clamp
        node.inputs["From Min"].default_value = -1. # From Min
        node.inputs["From Max"].default_value = 1. # From Max
        node.inputs["To Min"].default_value = 0. # To Min
        node.inputs["To Max"].default_value = 1. # To Max
        return node

    if editor_type == "Shader":
        map_range_node_output_socket_name = "Result"
        converter_io_node_socket_name = "Color"
    elif editor_type == "Compositor":
        map_range_node_output_socket_name = "Value"
        converter_io_node_socket_name = "Image"
    map_range_nodes = {k: create_normal_to_rgb_map_node() for k in ["R", "G", "B"]}
    comb_color_node = node_tree.nodes.new(f"{editor_type}NodeCombineColor")
    link(sep_color_node.outputs["Red"], map_range_nodes["R"].inputs["Value"])
    link(sep_color_node.outputs["Green"], map_range_nodes["G"].inputs["Value"])
    link(sep_color_node.outputs["Blue"], map_range_nodes["B"].inputs["Value"])
    link(map_range_nodes["R"].outputs[map_range_node_output_socket_name], comb_color_node.inputs["Red"])
    link(map_range_nodes["G"].outputs[map_range_node_output_socket_name], comb_color_node.inputs["Green"])
    link(map_range_nodes["B"].outputs[map_range_node_output_socket_name], comb_color_node.inputs["Blue"])
    # return (input socket, output socket)
    return sep_color_node.inputs[converter_io_node_socket_name], comb_color_node.outputs[converter_io_node_socket_name]

def configure_renderer(
    engine: str = "BLENDER_EEVEE",
    # for BLENDER_EEVEE
    taa_render_samples: int = 64, # Sampling -> Render
    use_gtao: bool = False, # Ambient Occlusion
    gtao_distance: float = 0.2, # Ambient Occlusion -> Distance
    gtao_factor: float = 1.0, # Ambient Occlusion -> Factor
    gtao_quality: float = 0.25, # Ambient Occlusion -> Trace Precision
    use_gtao_bent_normals: bool = True, # Ambient Occlusion -> Bent Normals
    use_gtao_bounce: bool = True, # Ambient Occlusion -> Bounces Approximation
    use_ssr: bool = False, # Screen Space Reflections
    use_ssr_refraction: bool = False, # Screen Space Reflections -> Refraction
    # for CYCLES
    cycles_device: str = "CPU", # DEVICE
    cycles_compute_backend: str = "NONE", # in ["NONE", "CUDA", "METAL", ...]
    preview_samples: int = 1, # Sampling -> Viewport -> Max Samples
    render_samples: int = 128, # Sampling -> Render -> Max Samples
    use_denoising: bool = True, # Sampling -> Denoise
    max_bounces: int = 12, # Light Paths -> Max Bounces -> Total
    diffuse_bounces: int = 4, # Light Paths -> Max Bounces -> Diffuse
    glossy_bounces: int = 4, # Light Paths -> Max Bounces -> Glossy
    transmission_bounces: int = 12, # Light Paths -> Max Bounces -> Transmission
    volume_bounces: int = 0, # Light Paths -> Max Bounces -> Volume
    transparent_max_bounces: int = 8, # Light Paths -> Max Bounces -> Transparent
    caustics_reflective: bool = True, # Light Paths -> Caustics -> Reflective
    caustics_refractive: bool = True, # Light Paths -> Caustics -> Refractive

    use_high_quality_normals: bool = False, # Performance -> High Quality Normals
    film_transparent: bool = True, # Film -> Transparent
    resolution_x: int = 1024,
    resolution_y: int = 1024
) -> None:
    scene = bpy.context.scene
    scene.render.engine = engine    

    if engine == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = taa_render_samples
        scene.eevee.use_gtao = use_gtao
        scene.eevee.gtao_distance = gtao_distance
        scene.eevee.gtao_factor = gtao_factor
        scene.eevee.gtao_quality = gtao_quality
        scene.eevee.use_gtao_bent_normals = use_gtao_bent_normals
        scene.eevee.use_gtao_bounce = use_gtao_bounce
        scene.eevee.use_ssr = use_ssr
        scene.eevee.use_ssr_refraction = use_ssr_refraction
    else:
        # GPU rendering for CYCLES
        bpy.context.scene.cycles.device = cycles_device
        if cycles_device == "GPU":
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = cycles_compute_backend
            for device in bpy.context.preferences.addons['cycles'].preferences.get_devices_for_type(cycles_compute_backend):
                device.use = True

        scene.cycles.preview_samples = preview_samples
        scene.cycles.samples = render_samples
        scene.cycles.use_denoising = use_denoising
        scene.cycles.max_bounces = max_bounces
        scene.cycles.diffuse_bounces = diffuse_bounces
        scene.cycles.glossy_bounces = glossy_bounces
        scene.cycles.transmission_bounces = transmission_bounces
        scene.cycles.volume_bounces = volume_bounces
        scene.cycles.transparent_max_bounces = transparent_max_bounces
        scene.cycles.caustics_reflective = caustics_reflective
        scene.cycles.caustics_refractive = caustics_refractive

    scene.render.use_high_quality_normals = use_high_quality_normals
    scene.render.film_transparent = film_transparent

    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y    


def configure_scene(
    # auto smooth settings
    use_auto_smooth: bool = False,  # Whether to enable auto smooth shading
    auto_smooth_angle_deg: float = 30.,  # Angle threshold in degrees for auto smooth shading
    # material blend and shadow settings 
    blend_mode: Optional[str] = None,  # Blend mode for materials (e.g. 'OPAQUE', 'BLEND', etc)
    shadow_mode: Optional[str] = None,  # Shadow mode for materials
    show_transparent_back: bool = False,  # Whether to show back faces for transparent materials
) -> None:
    """
    Configure scene-wide settings for mesh shading and material properties.

    Args:
        use_auto_smooth: If True, enables auto smooth shading for meshes to reduce sharp edges
        auto_smooth_angle_deg: Angle threshold (in degrees) above which edges will be smoothed
        blend_mode: Sets the blend mode for all materials. Options include:
                   - 'OPAQUE': No transparency
                   - 'BLEND': Standard alpha blending
                   - 'CLIP': Binary transparency (1 or 0)
                   etc.
        shadow_mode: Sets how materials cast shadows. Options include:
                    - 'OPAQUE': Fully opaque shadows
                    - 'CLIP': Binary transparency shadows
                    - 'NONE': No shadows
                    etc.
        show_transparent_back: If True and blend_mode='BLEND', shows back faces of transparent materials.
                             Should be False when using EEVEE renderer.
    """
    # Configure mesh shading type for all objects
    set_shading_type(
        use_auto_smooth=use_auto_smooth,
        auto_smooth_angle_deg=auto_smooth_angle_deg
    )
        
    # Configure all materials in the scene
    for material in bpy.data.materials:
        # Enable material nodes for all materials
        material.use_nodes = True

        # Set blend mode if specified
        if blend_mode is not None:
            material.blend_method = blend_mode
            # For transparent materials, configure back face visibility
            if material.blend_method == 'BLEND':
                material.show_transparent_back = show_transparent_back    

        # Set shadow mode if specified
        if shadow_mode is not None:
            material.shadow_method = shadow_mode 


def configure_render_output(
   clear_normal_map: bool = False,
) -> set[str]:
    # configure material custom output
    has_bsdf = False
    for material in bpy.data.materials:
        material.use_nodes = True
        link = make_link_func(material.node_tree)
        bsdf_node = material.node_tree.nodes.get("Principled BSDF")
        if bsdf_node is not None:
            has_bsdf = True
            if clear_normal_map:
                if bsdf_node.inputs["Normal"].is_linked:
                    for link in bsdf_node.inputs["Normal"].links:
                        material.node_tree.links.remove(link)
            
            for attr_name in ["Base Color", "Roughness", "Metallic"]:
                attr_input = bsdf_node.inputs[attr_name]
                if attr_input.is_linked:
                    linked_socket = attr_input.links[0].from_socket

                    aov_output = material.node_tree.nodes.new("ShaderNodeOutputAOV")
                    aov_output.name = attr_name
                    link(linked_socket, aov_output.inputs[0])
                else:
                    fixed_attr_value = attr_input.default_value
                    if isinstance(fixed_attr_value, float):
                        fixed_attr_input = material.node_tree.nodes.new("ShaderNodeValue")
                    else:
                        fixed_attr_input = material.node_tree.nodes.new("ShaderNodeRGB")

                    fixed_attr_input.outputs[0].default_value = fixed_attr_value

                    aov_output = material.node_tree.nodes.new("ShaderNodeOutputAOV")
                    aov_output.name = attr_name
                    link(fixed_attr_input.outputs[0], aov_output.inputs[0])

            normal_input = bsdf_node.inputs["Normal"]              
            if normal_input.is_linked:
                aov_output = material.node_tree.nodes.new("ShaderNodeOutputAOV")
                aov_output.name = "Normal Map"
                link(normal_input.links[0].from_socket, aov_output.inputs[0])
            else:
                normal_map_node = material.node_tree.nodes.new("ShaderNodeNormalMap")
                aov_output = material.node_tree.nodes.new("ShaderNodeOutputAOV")                  
                aov_output.name = "Normal Map"
                material.node_tree.links.new(normal_map_node.outputs["Normal"], aov_output.inputs[0])       
    
    render_types = ALL_RENDER_TYPES.copy()
    if not has_bsdf:
        render_types = render_types - BSDF_RENDER_TYPES

    scene = bpy.context.scene
    scene.render.use_compositing = True
    scene.use_nodes = True
    if "Render Layers" not in scene.node_tree.nodes:
        # this node is removed during clear_scene(), create again
        render_layers = scene.node_tree.nodes.new('CompositorNodeRLayers')
    else:
        render_layers = scene.node_tree.nodes["Render Layers"]
    link = make_link_func(scene.node_tree)

    RESOLUTION_X, RESOLUTION_Y = 1024, 1024
    WEBP_QUALITY = 100

    temp_dir = f"/tmp/{uuid.uuid4()}/"
    os.makedirs(temp_dir, exist_ok=True)
    scene.render.filepath = temp_dir
    scene.render.resolution_x = RESOLUTION_X
    scene.render.resolution_y = RESOLUTION_Y
    scene.render.image_settings.file_format = "WEBP"
    scene.render.image_settings.quality = WEBP_QUALITY
    scene.render.image_settings.color_mode = "RGBA" 

    # depth
    bpy.context.view_layer.use_pass_z = True
    depth_output_node = scene.node_tree.nodes.new('CompositorNodeOutputFile')
    depth_output_node.name = 'Depth Output'
    depth_output_node.format.file_format = 'OPEN_EXR'
    depth_output_node.format.color_depth = '32'
    depth_output_node.base_path = temp_dir
    depth_output_node.file_slots.values()[0].path = "depth_"
    link(render_layers.outputs["Depth"], depth_output_node.inputs['Image'])

    # normal
    bpy.context.view_layer.use_pass_normal = True

    normal_trans_input_socket, normal_trans_output_socket = make_normal_to_rgb_node_group(scene.node_tree, editor_type="Compositor")
    set_normal_alpha_node = scene.node_tree.nodes.new("CompositorNodeSetAlpha")
    set_normal_alpha_node.mode = "REPLACE_ALPHA"
    link(render_layers.outputs["Normal"], normal_trans_input_socket)
    link(normal_trans_output_socket, set_normal_alpha_node.inputs["Image"])
    link(render_layers.outputs["Alpha"], set_normal_alpha_node.inputs["Alpha"])

    normal_output_node = scene.node_tree.nodes.new('CompositorNodeOutputFile')
    normal_output_node.name = 'Normal Output'
    normal_output_node.format.file_format = 'WEBP'
    normal_output_node.format.quality = WEBP_QUALITY
    normal_output_node.format.color_depth = '8'
    set_file_output_non_color(normal_output_node)

    normal_output_node.base_path = temp_dir
    normal_output_node.file_slots.values()[0].path = "normal_"
    link(set_normal_alpha_node.outputs["Image"], normal_output_node.inputs['Image'])    

    if has_bsdf:
        # roughness
        bpy.ops.scene.view_layer_add_aov()
        bpy.context.scene.view_layers["ViewLayer"].active_aov.name = "Roughness"

        roughness_output_node = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        roughness_output_node.name = "Roughness Output"
        roughness_output_node.format.file_format = "WEBP"
        roughness_output_node.format.quality = WEBP_QUALITY
        roughness_output_node.format.color_depth = '8'
        set_file_output_non_color(roughness_output_node)

        roughness_output_node.base_path = temp_dir
        roughness_output_node.file_slots.values()[0].path = "roughness_"

        set_roughness_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_roughness_alpha_node.mode = "REPLACE_ALPHA"
        link(render_layers.outputs["Roughness"], set_roughness_alpha_node.inputs["Image"])
        link(render_layers.outputs["Alpha"], set_roughness_alpha_node.inputs["Alpha"])
        link(set_roughness_alpha_node.outputs['Image'], roughness_output_node.inputs['Image'])

        # metallic
        bpy.ops.scene.view_layer_add_aov()
        bpy.context.scene.view_layers["ViewLayer"].active_aov.name = "Metallic"

        metallic_output_node = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        metallic_output_node.name = "Metallic Output"
        metallic_output_node.format.file_format = "WEBP"
        metallic_output_node.format.quality = WEBP_QUALITY
        metallic_output_node.format.color_depth = '8'
        set_file_output_non_color(metallic_output_node)   

        metallic_output_node.base_path = temp_dir
        metallic_output_node.file_slots.values()[0].path = "metallic_"

        set_metallic_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_metallic_alpha_node.mode = "REPLACE_ALPHA"
        link(render_layers.outputs["Metallic"], set_metallic_alpha_node.inputs["Image"])
        link(render_layers.outputs["Alpha"], set_metallic_alpha_node.inputs["Alpha"])
        link(set_metallic_alpha_node.outputs['Image'], metallic_output_node.inputs['Image'])
        
        # base color
        bpy.ops.scene.view_layer_add_aov()
        bpy.context.scene.view_layers["ViewLayer"].active_aov.name = "Base Color"

        base_color_output_node = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        base_color_output_node.name = "Base Color Output"
        base_color_output_node.format.file_format = "WEBP"
        base_color_output_node.format.quality = WEBP_QUALITY
        base_color_output_node.format.color_depth = '8'    

        base_color_output_node.base_path = temp_dir
        base_color_output_node.file_slots.values()[0].path = "base_color_"

        set_base_color_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_base_color_alpha_node.mode = "REPLACE_ALPHA"
        link(render_layers.outputs["Base Color"], set_base_color_alpha_node.inputs["Image"])
        link(render_layers.outputs["Alpha"], set_base_color_alpha_node.inputs["Alpha"])
        link(set_base_color_alpha_node.outputs['Image'], base_color_output_node.inputs['Image'])

        # normal map
        bpy.ops.scene.view_layer_add_aov()
        bpy.context.scene.view_layers["ViewLayer"].active_aov.name = "Normal Map"

        normal_map_output_node = scene.node_tree.nodes.new(type="CompositorNodeOutputFile")
        normal_map_output_node.name = "Normal Map Output"
        normal_map_output_node.format.file_format = "WEBP"
        normal_map_output_node.format.quality = WEBP_QUALITY
        normal_map_output_node.format.color_depth = '8'
        set_file_output_non_color(normal_map_output_node)

        normal_map_output_node.base_path = temp_dir
        normal_map_output_node.file_slots.values()[0].path = "normal_map_"

        normal_map_trans_input_socket, normal_map_trans_output_socket = make_normal_to_rgb_node_group(scene.node_tree, editor_type="Compositor")
        set_normal_map_alpha_node = scene.node_tree.nodes.new(type="CompositorNodeSetAlpha")
        set_normal_map_alpha_node.mode = "REPLACE_ALPHA"
        link(render_layers.outputs["Normal Map"], normal_map_trans_input_socket)
        link(render_layers.outputs["Alpha"], set_normal_map_alpha_node.inputs["Alpha"])
        link(normal_map_trans_output_socket, set_normal_map_alpha_node.inputs['Image'])   
        link(set_normal_map_alpha_node.outputs['Image'], normal_map_output_node.inputs['Image']) 
    
    return render_types


def get_or_create_camera():
    if bpy.context.scene.camera is None:
        # create a new camera
        camera_data = bpy.data.cameras.new(name="Camera")
        camera_object = bpy.data.objects.new("Camera", camera_data)
        scene = bpy.context.scene
        scene.collection.objects.link(camera_object)
        scene.camera = camera_object
    return bpy.context.scene.camera


def configure_camera(
    projection_type: str,  # Type of camera projection: "PERSP" for perspective or "ORTHO" for orthographic
    transform_matrix: np.ndarray,  # 4x4 transformation matrix defining camera pose
    # Perspective camera parameters
    focal_length_mm: Optional[float] = None,  # Focal length in millimeters for perspective camera
    fov_deg: Optional[float] = None,  # Field of view in degrees for perspective camera
    # Orthographic camera parameters  
    ortho_scale: Optional[float] = None,  # Scale factor for orthographic camera viewport
    near_clip: float = 0.1,  # Near clipping distance - objects closer than this won't be rendered
    far_clip: float = 100,  # Far clipping distance - objects further than this won't be rendered
):
    """
    Configure a Blender camera with the specified parameters.

    This function either retrieves an existing camera or creates a new one, then
    configures its projection type, transformation, and other parameters.

    Args:
        projection_type: Either "PERSP" for perspective or "ORTHO" for orthographic projection
        transform_matrix: 4x4 numpy array defining camera position and orientation
        focal_length_mm: For perspective camera - focal length in millimeters
        fov_deg: For perspective camera - field of view in degrees
        ortho_scale: For orthographic camera - scale of the viewport
        near_clip: Distance to near clipping plane
        far_clip: Distance to far clipping plane

    Note:
        - For perspective camera, exactly one of focal_length_mm or fov_deg must be provided
        - For orthographic camera, ortho_scale must be provided
    """

    # Get existing camera or create a new one
    camera = get_or_create_camera()
    
    # Set camera transformation matrix
    camera.matrix_world = Matrix(transform_matrix)

    # Validate and set projection type
    assert projection_type in ["PERSP", "ORTHO"]
    camera.data.type = projection_type

    if projection_type == "PERSP":
        # For perspective camera, ensure exactly one of focal length or FOV is given
        assert (focal_length_mm is None) + (fov_deg is None) == 1, "Only one of focal_length_mm and fov_deg should be provided!"
        
        # Configure using focal length
        if focal_length_mm is not None:
            camera.data.lens_unit = "MILLIMETERS"
            camera.data.lens = focal_length_mm
            
        # Configure using field of view
        if fov_deg is not None:
            camera.data.lens_unit = "FOV"
            camera.data.angle = fov_deg * math.pi / 180  # Convert degrees to radians
            
    elif projection_type == "ORTHO":
        # For orthographic camera, ensure scale is provided
        assert ortho_scale is not None
        camera.data.ortho_scale = ortho_scale
    
    # Set clipping planes
    camera.data.clip_start = near_clip
    camera.data.clip_end = far_clip


def compute_object_bbox_np(
    method: str,  # Method to compute bounding box: "bound_box" or "vertex"
    obj          # The Blender object to compute bounding box for
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the axis-aligned bounding box for a single Blender object.

    Args:
        method: Method to use for computing bounding box:
               - "bound_box": Uses object's predefined bounding box (faster but may not be tight)
               - "vertex": Computes from mesh vertices (more accurate but slower)
        obj: Blender object to compute bounding box for

    Returns:
        Tuple containing:
        - bbox_min: numpy array [x,y,z] of minimum coordinates
        - bbox_max: numpy array [x,y,z] of maximum coordinates

    Note:
        The bounding box is computed in world space coordinates
    """
    # Get object's world transformation matrix
    matrix_world = np.array(obj.matrix_world)

    if method == "bound_box":
        # Using object's predefined bounding box
        # Note: This may not be axis-aligned in world space since it's defined in local coordinates
        # Example issue: model 0004170416344b7797497cd34eed5940 
        # Potential consequence: computed bounding box may be larger than necessary
        bbox_coords = np.array([np.array(pt) for pt in obj.bound_box]) # Shape: (8, 3)
        
        # Transform bounding box corners to world space
        bbox_coords = transform_points(matrix_world, bbox_coords)
        
        # Get min/max coordinates
        bbox_min, bbox_max = bbox_coords.min(0), bbox_coords.max(0)

    elif method == "vertex":
        # Using mesh vertices for more accurate bounding box
        # Note: This doesn't account for modifiers 
        # Example issue: model 0362307da6ad4e639348c5d5e5c1b420
        
        # Get vertex coordinates from mesh data
        vertices = np.array([np.array(v.co) for v in obj.data.vertices])
        
        # Transform vertices to world space
        vertices = transform_points(matrix_world, vertices)
        
        # Get min/max coordinates
        bbox_min, bbox_max = vertices.min(0), vertices.max(0)

    return bbox_min, bbox_max


def compute_objects_bbox_np(
    method: str,  # Method to compute bbox: "bound_box" or "vertex"
    objs         # Iterator of Blender objects to compute bbox for
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the combined axis-aligned bounding box for multiple Blender objects.

    Args:
        method: Method to compute bounding box, either "bound_box" (faster) or "vertex" (more accurate)
        objs: Iterator of Blender objects to include in bbox calculation

    Returns:
        Tuple containing:
            - bbox_min: numpy array [x,y,z] of minimum coordinates
            - bbox_max: numpy array [x,y,z] of maximum coordinates 
    """
    # Lists to store individual object bboxes
    bbox_min, bbox_max = [], []

    # Compute bbox for each object
    for obj in objs:
        # Get individual object's bbox
        bbox_min_, bbox_max_ = compute_object_bbox_np(method, obj)
        bbox_min.append(bbox_min_)
        bbox_max.append(bbox_max_)

    # Convert lists to numpy arrays - shape (N, 3) where N is number of objects
    bbox_min = np.array(bbox_min) 
    bbox_max = np.array(bbox_max)

    # Get overall min/max across all objects
    bbox_min, bbox_max = bbox_min.min(0), bbox_max.max(0)

    return bbox_min, bbox_max


def scene_root_objects(override_context: bool = False):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_mesh_objects(override_context: bool = False):
    """
    Generator function that yields all visible mesh objects in the current scene.

    Args:
        override_context (bool): Flag to indicate if context override is needed. 
            Default is False.

    Yields:
        bpy.types.Object: Visible mesh objects in the scene that meet the following criteria:
            - Type is 'MESH'
            - Is visible in viewport (visible_get() is True) 
            - Is not hidden (hide_get() is False)

    Note:
        - First deselects all objects to ensure clean selection state
        - Checks both viewport visibility and hide status
        - Only returns mesh objects, ignoring other types like lights, cameras etc
        - Objects are yielded one at a time to be memory efficient
    """
    # First deselect all objects to ensure clean selection state
    bpy.ops.object.select_all(action="DESELECT")

    # Iterate through all objects in the scene
    for obj in bpy.context.scene.objects:
        # Check if object is:
        # 1. A mesh type
        # 2. Visible in viewport 
        # 3. Not hidden
        if obj.type == 'MESH' and obj.visible_get() is True and obj.hide_get() is False:
            yield obj



def object_context(obj):
    override = bpy.context.copy()
    override["selected_objects"] = [obj]
    override["selected_editable_objects"] = [obj]
    override["active_object"] = obj
    return bpy.context.temp_override(**override)


def normalize(
    bbox_compute_method: str,  # Method to compute bounding box: "bound_box" or "vertex"
    normalizing_scale: float,  # Target scale (half of object dimension)
    rotation_euler: Tuple[float, float, float]  # Rotation angles in radians to apply
) -> NormalizationSpec:    
    # Compute initial bounding box of all mesh objects in scene
    bbox_min, bbox_max = compute_objects_bbox_np(bbox_compute_method, scene_mesh_objects())
    
    # Calculate center point and scaling factor needed to achieve target scale
    center = (bbox_min + bbox_max) / 2.
    # Scale factor = (target size) / (current size)
    # normalizing_scale is half the target object dimension
    scaling_factor = normalizing_scale * 2. / (bbox_max - bbox_min).max()
    
    # Translation vector to center object at origin 
    translation = -center
    
    # Create empty object to serve as parent for all scene objects
    bpy.ops.object.empty_add(type='PLAIN_AXES')
    root_object = bpy.context.object
    
    # Parent all root objects to the empty while preserving their world transforms
    for obj in scene_root_objects():
        if obj != root_object:
            _matrix_world = obj.matrix_world.copy()
            obj.parent = root_object
            obj.matrix_world = _matrix_world
    
    # Apply transformations in order: translate -> rotate -> scale
    # First translate to center
    root_object.location = Vector(translation)
    
    # Apply translation so it becomes "baked in"
    with object_context(root_object):
        bpy.ops.object.transform_apply(location=True)
        
    # Apply rotation
    root_object.rotation_euler = rotation_euler
    
    # Apply scaling
    root_object.scale = (scaling_factor, scaling_factor, scaling_factor)

    # Compute final bounding box after transformations
    bbox_min, bbox_max = compute_objects_bbox_np(bbox_compute_method, scene_mesh_objects())

    # Return normalization parameters 
    return NormalizationSpec(
        scaling_factor=scaling_factor,
        rotation_euler=rotation_euler,
        translation=translation.tolist(),
        bbox_min=bbox_min.tolist(),
        bbox_max=bbox_max.tolist()
    )


def set_environment_light(
    use_environment_map: bool = False,        # Whether to use an environment map for lighting
    environment_map_path: Optional[str] = None,  # Path to the environment map texture file 
    light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),  # RGB color for ambient light
    light_strength: float = 1.0,              # Overall strength/intensity of the lighting
    environment_map_rotation_deg: float = 0.   # Rotation of environment map in degrees
):
    """
    Configure the environment lighting in the Blender scene.
    
    This function sets up either a solid color ambient light or an environment map light.
    For environment maps, it supports rotation and strength adjustment.
    
    Args:
        use_environment_map: If True, uses an HDR/EXR environment map for lighting
        environment_map_path: Path to environment map image file (required if use_environment_map is True)
        light_color: RGB tuple defining color of ambient light when not using environment map
        light_strength: Overall multiplier for light intensity
        environment_map_rotation_deg: Rotation angle in degrees to apply to environment map
    """

    # Get or create world settings
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
    
    # Enable and clear existing nodes
    world.use_nodes = True
    world.node_tree.nodes.clear()

    # Create basic shader nodes
    bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = light_strength
    output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')

    # Helper function to create node links
    link = make_link_func(world.node_tree)

    if use_environment_map:
        # Verify environment map path exists
        assert environment_map_path is not None and os.path.exists(environment_map_path)
        
        # Setup environment texture node
        env_texture_node = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
        bpy.ops.image.open(filepath=environment_map_path)
        env_texture_node.image = bpy.data.images.get(os.path.basename(environment_map_path))
        link(env_texture_node.outputs['Color'], bg_node.inputs['Color'])

        # Setup coordinate mapping for environment rotation
        tex_coord_node = world.node_tree.nodes.new(type='ShaderNodeTexCoord')
        mapping_node = world.node_tree.nodes.new(type='ShaderNodeMapping')
        link(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
        link(mapping_node.outputs['Vector'], env_texture_node.inputs['Vector'])
        mapping_node.inputs["Rotation"].default_value[2] = math.radians(environment_map_rotation_deg)
    else:
        # Use solid color for ambient light
        bg_node.inputs["Color"].default_value = (light_color[0], light_color[1], light_color[2], 1.)

    # Connect background shader to world output
    link(bg_node.outputs['Background'], output_node.inputs['Surface'])

def set_current_frame(frame_idx: int = 1):
    """
    Sets the current frame in Blender's timeline to the specified frame index.

    Args:
        frame_idx (int, optional): The frame number to set as the current frame. 
            Defaults to 1.

    Note:
        This function uses Blender's Python API (bpy) to modify the scene's current frame.
        The frame index should be a positive integer representing the desired frame number
        in the timeline.

    Example:
        >>> set_current_frame(10)  # Sets the current frame to frame 10
        >>> set_current_frame()    # Sets the current frame to frame 1 (default)
    """
    bpy.context.scene.frame_set(frame_idx)


def execute_render_and_save(
    render_types: set[str],  # Set of render types to generate (e.g. "Color", "Depth", "Normal" etc.)
    save_dir: str,  # Directory to save final render outputs
    name_format: str,  # Format string for output filenames
    overwrite: bool,  # Whether to overwrite existing files
    # Additional kwargs used for file naming
    **kwargs  
) -> Dict[str, str]:
    """
    Execute Blender render and save outputs to specified directory.

    Args:
        render_types: Set of render types to generate (e.g. "Color", "Depth", "Normal" etc.)
        save_dir: Directory to save final render outputs
        name_format: Format string for output filenames
        overwrite: Whether to overwrite existing files
        **kwargs: Additional keyword arguments used in name_format

    Returns:
        List of RenderOutput objects containing render type and file path information

    This function:
    1. Sets up temporary cache directory for storing render outputs
    2. Configures output paths for color render and auxiliary render types (depth, normal etc.)
    3. Executes Blender render
    4. Moves rendered files from cache to final save directory with proper naming
    5. Cleans up temporary cache directory
    """

    # Setup directories - cache for needed renders, dump for unneeded ones
    cache_dir = os.path.join(save_dir, ".cache") 
    dump_dir = "/tmp"  # Temporary dir for render types we don't need to keep

    scene = bpy.context.scene

    # Track paths of rendered files in cache
    cache_output_files = {}

    # Configure color render output path
    scene.render.filepath = os.path.join(cache_dir if "Color" in render_types else dump_dir, RENDER_TYPE_TO_SAVE_NAME['Color'])
    if "Color" in render_types:
        cache_output_files["Color"] = scene.render.filepath + f".{RENDER_TYPE_TO_FILE_EXT['Color']}"
    
    # Configure auxiliary render outputs (depth, normal etc.)
    for render_type in AUXILIARY_RENDER_TYPES:
        # Skip if this render type's output node doesn't exist
        if f"{render_type} Output" not in scene.node_tree.nodes:
            continue

        node = bpy.context.scene.node_tree.nodes[f"{render_type} Output"]
        render_type_name = RENDER_TYPE_TO_SAVE_NAME[render_type]
        file_ext = RENDER_TYPE_TO_FILE_EXT[render_type]
        
        # Set output filename pattern
        node.file_slots.values()[0].path = render_type_name

        if render_type in render_types:
            # Save to cache if we need this render type
            node.base_path = cache_dir
            cache_output_files[render_type] = os.path.join(cache_dir, f"{render_type_name}{bpy.context.scene.frame_current:04d}.{file_ext}")
        else:
            # Otherwise save to dump directory
            node.base_path = dump_dir

    # Execute render
    bpy.ops.render.render(animation=False, write_still=True)

    # Process rendered files
    render_outputs = []
    for render_type, cache_file_path in cache_output_files.items():
        # Generate final filename using format string
        file_name = name_format.format(
            render_type=RENDER_TYPE_TO_SAVE_NAME[render_type], 
            file_ext=RENDER_TYPE_TO_FILE_EXT[render_type], 
            **kwargs
        )
        dest_path = os.path.join(save_dir, file_name)

        # Create render output object
        render_outputs.append(RenderOutput(
            render_type=render_type,
            file_path=dest_path
        ))

        # Move file from cache to final location if needed
        if not os.path.exists(dest_path) or overwrite:
            shutil.move(cache_file_path, dest_path)
        else:
            # TODO: should be a warning
            print(f"File {dest_path} already exists, skip.")

    # Cleanup cache directory
    shutil.rmtree(cache_dir)

    return render_outputs
