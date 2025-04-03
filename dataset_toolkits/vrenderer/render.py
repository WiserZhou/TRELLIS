from typing import List, Optional

from .spec import (
    AUXILIARY_RENDER_TYPES,
    InitializationSettings, RuntimeSettings,
    CameraSpec, NormalizationSpec,
    RenderOutput, InitializationOutput
)
from .blender_utils import (
    clear_scene, load_model,
    configure_render_output,
    set_current_frame,
    configure_renderer,
    configure_scene,
    configure_camera,
    set_environment_light,
    execute_render_and_save,
    use_opaque_material,
    use_single_sample_renderer,
    use_flat_shading
)



def initialize(settings: InitializationSettings, part_names: Optional[List[str]] = None) -> NormalizationSpec:
    """
    Initialize the rendering environment and load the 3D model.

    Args:
        settings: InitializationSettings object containing model loading and initialization parameters

    Returns:
        InitializationOutput containing normalization specs and valid render types
    """
    # Clear any existing objects from the Blender scene
    clear_scene()

    # Load and normalize the 3D model according to specified settings
    normalization_spec = load_model(
        file_path=settings.file_path,
        file_format=settings.file_format,
        forward_axis=settings.forward_axis,
        up_axis=settings.up_axis,
        merge_vertices=settings.merge_vertices,
        bbox_compute_method=settings.bbox_compute_method,
        normalizing_scale=settings.normalizing_scale,
        rotation_euler=settings.rotation_euler,
        default_frame=settings.default_frame,
        part_names=part_names
    )

    # Configure render passes and output nodes in the compositor
    # Returns the set of valid render types that can be generated
    valid_render_types = configure_render_output(clear_normal_map=settings.clear_normal_map)

    # Return initialization results including:
    # - normalization_spec: Contains model scaling and transformation info
    # - valid_render_types: Set of available render passes
    # - index_to_name: Optional mapping of indices to object names (None in this case)
    return InitializationOutput(
        normalization_spec=normalization_spec,
        valid_render_types=valid_render_types,
        index_to_name=None
    )


def render_and_save(
    settings: RuntimeSettings,
    cameras: List[CameraSpec],
    initialization_output: InitializationOutput,
    save_dir: str,
    name_format: str,
    render_types: set[str],
    camera_index_offset: int = 0,
    overwrite: bool = False
) -> List[List[RenderOutput]]:
    """
    Render scenes from multiple camera views and save the results.

    Args:
        settings: Runtime settings for rendering
        cameras: List of camera specifications 
        initialization_output: Output from initialization containing valid render types
        save_dir: Directory to save rendered images
        name_format: Format string for output filenames
        render_types: Set of render types to generate (e.g. Color, Normal, Depth etc.)
        camera_index_offset: Offset added to camera indices in output filenames
        overwrite: Whether to overwrite existing files

    Returns:
        List of lists containing RenderOutput objects for each camera view
    """
    # Set the current animation frame
    set_current_frame(settings.frame_index)

    # Configure environment lighting
    set_environment_light(
        use_environment_map=settings.use_environment_map,
        environment_map_path=settings.environment_map_path,
        light_color=settings.light_color,
        light_strength=settings.light_strength,
        environment_map_rotation_deg=settings.environment_map_rotation_deg
    )

    # Configure render engine settings (Cycles/EEVEE parameters)
    configure_renderer(
        engine=settings.engine,
        taa_render_samples=settings.taa_render_samples,
        use_gtao=settings.use_gtao,
        gtao_distance=settings.gtao_distance,
        gtao_factor=settings.gtao_factor,
        gtao_quality=settings.gtao_quality,
        use_gtao_bent_normals=settings.use_gtao_bent_normals,
        use_gtao_bounce=settings.use_gtao_bounce,
        use_ssr=settings.use_ssr,
        use_ssr_refraction=settings.use_ssr_refraction,
        cycles_device=settings.cycles_device,
        cycles_compute_backend=settings.cycles_compute_backend,
        preview_samples=settings.preview_samples,
        render_samples=settings.render_samples,
        use_denoising=settings.use_denoising,
        max_bounces=settings.max_bounces,
        diffuse_bounces=settings.diffuse_bounces,
        glossy_bounces=settings.glossy_bounces,
        transmission_bounces=settings.transmission_bounces,
        volume_bounces=settings.volume_bounces,
        transparent_max_bounces=settings.transparent_max_bounces,
        caustics_reflective=settings.caustics_reflective,
        caustics_refractive=settings.caustics_refractive,
        use_high_quality_normals=settings.use_high_quality_normals,
        film_transparent=settings.film_transparent,
        resolution_x=settings.resolution_x,
        resolution_y=settings.resolution_y
    )

    # Configure scene-wide settings
    configure_scene(
        use_auto_smooth=settings.use_auto_smooth,
        auto_smooth_angle_deg=settings.auto_smooth_angle_deg,
        blend_mode=settings.blend_mode,
        shadow_mode=settings.shadow_mode,
        show_transparent_back=settings.show_transparent_back
    )

    group_render_outputs = []
    # Iterate through each camera
    for camera_index, camera_spec in enumerate(cameras):
        # Configure camera parameters
        configure_camera(
            projection_type=camera_spec.projection_type,
            transform_matrix=camera_spec.transform_matrix,
            focal_length_mm=camera_spec.focal_length_mm,
            fov_deg=camera_spec.fov_deg,
            ortho_scale=camera_spec.ortho_scale,
            near_clip=camera_spec.near_clip,
            far_clip=camera_spec.far_clip
        )

        render_outputs = []
        # Handle color rendering separately since it needs different material settings
        if "Color" in render_types:
            render_outputs_ = execute_render_and_save(
                render_types={"Color"},
                save_dir=save_dir,
                name_format=name_format,
                overwrite=overwrite,
                camera_index=camera_index + camera_index_offset
            )
            render_outputs += render_outputs_
        
        # Handle auxiliary render types (normals, depth etc.) with special material settings
        auxiliary_render_types = render_types & initialization_output.valid_render_types & AUXILIARY_RENDER_TYPES
        if auxiliary_render_types:
            # Use context managers to temporarily modify material/render settings
            with use_opaque_material(), use_single_sample_renderer(), use_flat_shading(settings.use_auto_smooth, settings.auto_smooth_angle_deg):
                render_outputs_ = execute_render_and_save(
                    render_types=auxiliary_render_types,
                    save_dir=save_dir,
                    name_format=name_format,
                    overwrite=overwrite,
                    camera_index=camera_index + camera_index_offset
                )
                render_outputs += render_outputs_

        group_render_outputs.append(render_outputs)
    
    return group_render_outputs
