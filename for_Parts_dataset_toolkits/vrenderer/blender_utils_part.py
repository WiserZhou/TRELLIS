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
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()  

    bpy.context.scene.use_nodes = True
    scene_node_tree = bpy.context.scene.node_tree
    for node in scene_node_tree.nodes:
        scene_node_tree.nodes.remove(node)


def load_model(
    file_path: str,
    file_format: Optional[str] = None,
    forward_axis: str = "NEGATIVE_Z",
    up_axis: str = "Y",
    merge_vertices: bool = True,
    # normalization
    bbox_compute_method: str = "bound_box",
    normalizing_scale: float = 0.5,
    rotation_euler: Tuple[float, float, float] = (0, 0, 0),
    default_frame: int = 1,
) -> NormalizationSpec:
    if file_format is None:
        file_format = os.path.basename(file_path).split('.')[-1]

    if file_format == "glb":
        bpy.ops.import_scene.gltf(filepath=file_path, merge_vertices=merge_vertices)
    elif file_format == "obj":
        up_axis = "Z"
        # forward_axis = ""
        bpy.ops.wm.obj_import(filepath=file_path, directory=os.path.dirname(file_path),forward_axis=forward_axis, up_axis=up_axis)
    elif file_format == "ply":
        bpy.ops.wm.ply_import(filepath=file_path, directory=os.path.dirname(file_path),forward_axis=forward_axis, up_axis=up_axis)
        preprocess_ply(bpy.context.selected_objects[0])
    elif file_format == "vrm":
        bpy.ops.import_scene.vrm(filepath=file_path)   
    else:
        raise NotImplementedError
    
    set_current_frame(default_frame)
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
    # auto smooth
    use_auto_smooth: bool = False,
    auto_smooth_angle_deg: float = 30.,
    # material blend and shadow
    blend_mode: Optional[str] = None,
    shadow_mode: Optional[str] = None,
    show_transparent_back: bool = False,
) -> None:
    set_shading_type(
        use_auto_smooth=use_auto_smooth,
        auto_smooth_angle_deg=auto_smooth_angle_deg
    )
        
    for material in bpy.data.materials:
        material.use_nodes = True

        if blend_mode is not None:
            material.blend_method = blend_mode
            if material.blend_method == 'BLEND':
                # should be set to False when using EEVEE
                material.show_transparent_back = show_transparent_back    

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

    # print("\nAvailable outputs after enabling object index:")
    # for output in render_layers.outputs:
    #     print(f"- {output.name}")

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

    # ObjectID mask
    index_to_name = assign_object_indices()
    bpy.context.view_layer.use_pass_object_index = True
    
    id_mask_output = scene.node_tree.nodes.new('CompositorNodeOutputFile')
    id_mask_output.name = 'Mask Output'
    id_mask_output.format.file_format = 'OPEN_EXR'  # Use EXR to preserve exact index values
    id_mask_output.format.color_depth = '16'
    id_mask_output.base_path = temp_dir
    id_mask_output.file_slots.values()[0].path = "mask_"
    # print("Available render layer outputs:", render_layers.outputs.keys())
    # breakpoint()
    link(render_layers.outputs['IndexOB'], id_mask_output.inputs['Image'])

    # render_types = ALL_RENDER_TYPES.copy() | {'ObjectID'}  # Add ObjectID to render types

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
    
    return render_types, index_to_name


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
    projection_type: str,
    transform_matrix: np.ndarray,
    # for perspective camera
    focal_length_mm: Optional[float] = None,
    fov_deg: Optional[float] = None,
    # for othographic camera
    ortho_scale: Optional[float] = None,
    near_clip: float = 0.1,
    far_clip: float = 100,
):
    camera = get_or_create_camera()
    camera.matrix_world = Matrix(transform_matrix)

    assert projection_type in ["PERSP", "ORTHO"]
    camera.data.type = projection_type
    if projection_type == "PERSP":
        assert (focal_length_mm is None) + (fov_deg is None) == 1, "Only one of focal_length_mm and fov_deg should be provided!"
        if focal_length_mm is not None:
            camera.data.lens_unit = "MILLIMETERS"
            camera.data.lens = focal_length_mm
        if fov_deg is not None:
            camera.data.lens_unit = "FOV"
            camera.data.angle = fov_deg * math.pi / 180
    elif projection_type == "ORTHO":
        assert ortho_scale is not None
        camera.data.ortho_scale = ortho_scale
    
    camera.data.clip_start = near_clip
    camera.data.clip_end = far_clip


def compute_object_bbox_np(
    method: str,
    obj
) -> Tuple[np.ndarray, np.ndarray]:
    matrix_world = np.array(obj.matrix_world)
    if method == "bound_box":
        # TODO: may not be axis-aligned, as the bounding box is defined in the local coordinate system, for example 0004170416344b7797497cd34eed5940
        # potential consequence: computed bounding box not tight enough
        bbox_coords = np.array([np.array(pt) for pt in obj.bound_box]) # (8, 3)
        bbox_coords = transform_points(matrix_world, bbox_coords)
        bbox_min, bbox_max = bbox_coords.min(0), bbox_coords.max(0)
    elif method == "vertex":
        # TODO: apply modifiers, for example 0362307da6ad4e639348c5d5e5c1b420
        vertices = np.array([np.array(v.co) for v in obj.data.vertices])
        vertices = transform_points(matrix_world, vertices)
        bbox_min, bbox_max = vertices.min(0), vertices.max(0)
    return bbox_min, bbox_max


def compute_objects_bbox_np(
    method: str,
    objs
):
    bbox_min, bbox_max = [], []
    for obj in objs:
        bbox_min_, bbox_max_ = compute_object_bbox_np(method, obj)
        bbox_min.append(bbox_min_)
        bbox_max.append(bbox_max_)
    bbox_min = np.array(bbox_min) # (N, 3)
    bbox_max = np.array(bbox_max) # (N, 3)
    bbox_min, bbox_max = bbox_min.min(0), bbox_max.max(0)
    return bbox_min, bbox_max


def scene_root_objects(override_context: bool = False):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_mesh_objects(override_context: bool = False):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH' and obj.visible_get() is True and obj.hide_get() is False:
            yield obj


def object_context(obj):
    override = bpy.context.copy()
    override["selected_objects"] = [obj]
    override["selected_editable_objects"] = [obj]
    override["active_object"] = obj
    return bpy.context.temp_override(**override)


def normalize(
    bbox_compute_method: str,
    normalizing_scale: float,
    rotation_euler: Tuple[float, float, float]
) -> NormalizationSpec:    
    bbox_min, bbox_max = compute_objects_bbox_np(bbox_compute_method, scene_mesh_objects())
    center = (bbox_min + bbox_max) / 2.
    scaling_factor = normalizing_scale * 2. / (bbox_max - bbox_min).max() # normalizing_scale is half object dimension
    translation = -center
    
    bpy.ops.object.empty_add(type='PLAIN_AXES')
    root_object = bpy.context.object
    for obj in scene_root_objects():
        if obj != root_object:
            _matrix_world = obj.matrix_world.copy()
            obj.parent = root_object
            obj.matrix_world = _matrix_world
    
    # order: translate -> scale
    root_object.location = Vector(translation)
    
    with object_context(root_object):
        bpy.ops.object.transform_apply(location=True)
    root_object.rotation_euler = rotation_euler
    root_object.scale = (scaling_factor, scaling_factor, scaling_factor)

    bbox_min, bbox_max = compute_objects_bbox_np(bbox_compute_method, scene_mesh_objects())

    return NormalizationSpec(
        scaling_factor=scaling_factor,
        rotation_euler=rotation_euler,
        translation=translation.tolist(),
        bbox_min=bbox_min.tolist(),
        bbox_max=bbox_max.tolist()
    )


def set_environment_light(
    use_environment_map: bool = False,
    environment_map_path: Optional[str] = None,
    light_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    light_strength: float = 1.0,
    environment_map_rotation_deg: float = 0.
):
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new('World')
        bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes.clear()

    bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = light_strength
    output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')

    link = make_link_func(world.node_tree)

    if use_environment_map:
        assert environment_map_path is not None and os.path.exists(environment_map_path)
        env_texture_node = world.node_tree.nodes.new(type='ShaderNodeTexEnvironment')
        bpy.ops.image.open(filepath=environment_map_path)
        env_texture_node.image = bpy.data.images.get(os.path.basename(environment_map_path))
        link(env_texture_node.outputs['Color'], bg_node.inputs['Color'])

        tex_coord_node = world.node_tree.nodes.new(type='ShaderNodeTexCoord')
        mapping_node = world.node_tree.nodes.new(type='ShaderNodeMapping')
        link(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
        link(mapping_node.outputs['Vector'], env_texture_node.inputs['Vector'])
        mapping_node.inputs["Rotation"].default_value[2] = math.radians(environment_map_rotation_deg)
    else:
        bg_node.inputs["Color"].default_value = (light_color[0], light_color[1], light_color[2], 1.)

    link(bg_node.outputs['Background'], output_node.inputs['Surface'])



def set_current_frame(frame_idx: int = 1):
    bpy.context.scene.frame_set(frame_idx)


def execute_render_and_save(
    render_types: set[str],
    save_dir: str,
    name_format: str,
    overwrite: bool,
    **kwargs
) -> Dict[str, str]:
    
    cache_dir = os.path.join(save_dir, ".cache")
    dump_dir = "/tmp" # if a render type is not needed, save it here

    scene = bpy.context.scene

    cache_output_files = {}

    scene.render.filepath = os.path.join(cache_dir if "Color" in render_types else dump_dir, RENDER_TYPE_TO_SAVE_NAME['Color'])
    if "Color" in render_types:
        cache_output_files["Color"] = scene.render.filepath + f".{RENDER_TYPE_TO_FILE_EXT['Color']}"
    
    for render_type in AUXILIARY_RENDER_TYPES:
        # print(render_type)
        if f"{render_type} Output" not in scene.node_tree.nodes:
            continue
        node = bpy.context.scene.node_tree.nodes[f"{render_type} Output"]
        render_type_name = RENDER_TYPE_TO_SAVE_NAME[render_type]
        file_ext = RENDER_TYPE_TO_FILE_EXT[render_type]
        node.file_slots.values()[0].path = render_type_name
        if render_type in render_types:
            # if we need this render type
            node.base_path = cache_dir
            cache_output_files[render_type] = os.path.join(cache_dir, f"{render_type_name}{bpy.context.scene.frame_current:04d}.{file_ext}")
        else:
            node.base_path = dump_dir

    bpy.ops.render.render(animation=False, write_still=True)

    render_outputs = []
    for render_type, cache_file_path in cache_output_files.items():
        file_name = name_format.format(render_type=RENDER_TYPE_TO_SAVE_NAME[render_type], file_ext=RENDER_TYPE_TO_FILE_EXT[render_type], **kwargs)
        dest_path = os.path.join(save_dir, file_name)
        render_outputs.append(RenderOutput(
            render_type=render_type,
            file_path=dest_path
        ))
        if not os.path.exists(dest_path) or overwrite:
            shutil.move(cache_file_path, dest_path)
        else:
            # TODO: should be a warning
            print(f"File {dest_path} already exists, skip.")

    shutil.rmtree(cache_dir)

    return render_outputs


def assign_object_indices():
    """Assign unique pass indices to each object in the scene.
    
    Returns:
        dict: A mapping from object index to object name
    """
    index = 1  # Start from 1, as 0 is typically background
    index_to_name = {}
    for obj in scene_mesh_objects():
        obj.pass_index = index
        index_to_name[index] = obj.name
        index += 1
    return index_to_name
