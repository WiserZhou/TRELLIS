"""
voxelize.py - A script for converting a 3D GLB model to voxel representation
This script converts a GLB file into a voxel grid within a normalized coordinate space.
The voxelized model is saved as a PLY file containing point coordinates.
"""

import os
import argparse
import numpy as np
import open3d as o3d
import trimesh


def voxelize_glb(input_file, output_file, resolution=64):
    """
    Voxelize a single 3D GLB model into a grid representation.
    
    Args:
        input_file: Path to the input GLB file
        output_file: Path to save the resulting voxel PLY file
        resolution: Voxel grid resolution (default: 64)
    
    Returns:
        dict: Information about the voxelization process
    """
    print(f"Processing {input_file}...")
    
    # Load the GLB file using trimesh
    mesh_trimesh = trimesh.load(input_file)
    
    # Convert to Open3D mesh
    vertices = np.array(mesh_trimesh.vertices)
    faces = np.array(mesh_trimesh.faces)
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Normalize mesh to centered [-0.5, 0.5] cube
    mesh.scale(1 / max(mesh.get_max_bound() - mesh.get_min_bound()), center=True)
    center = (mesh.get_max_bound() + mesh.get_min_bound()) / 2
    mesh.translate(-center)
    
    # Clamp vertices to ensure they're within bounds
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # Create a voxel grid from the mesh
    voxel_size = 1 / resolution
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=voxel_size, 
        min_bound=(-0.5, -0.5, -0.5), 
        max_bound=(0.5, 0.5, 0.5))
    
    # Extract voxel coordinates
    voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    
    # Verify voxels are within bounds
    assert np.all(voxel_indices >= 0) and np.all(voxel_indices < resolution), "Some vertices are out of bounds"
    
    # Convert grid indices back to normalized coordinate space
    voxel_points = (voxel_indices + 0.5) / resolution - 0.5
    
    # Save the voxel point cloud to a PLY file
    voxel_pc = o3d.geometry.PointCloud()
    voxel_pc.points = o3d.utility.Vector3dVector(voxel_points)
    o3d.io.write_point_cloud(output_file, voxel_pc)
    
    print(f"Voxelization complete: {len(voxel_points)} voxels created")
    print(f"Saved to {output_file}")
    
    return {
        'input_file': input_file,
        'output_file': output_file,
        'num_voxels': len(voxel_points),
        'resolution': resolution
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voxelize a GLB file into a PLY voxel representation')
    parser.add_argument('--input', type=str, required=True, help='Path to input GLB file')
    parser.add_argument('--output', type=str, required=True, help='Path to output PLY file')
    parser.add_argument('--resolution', type=int, default=64, help='Voxel grid resolution (default: 64)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Process the GLB file
    result = voxelize_glb(args.input, args.output, args.resolution)
