"""
render.py: A utility script for rendering a 3D GLB model using Blender.

This script automates the process of rendering multiple views of a 3D model. It:
1. Installs Blender if not already present
2. Takes a single GLB file as input
3. Renders it from various camera angles using Hammersley sequence for uniform view distribution
4. Saves the rendered images and camera parameters
"""

import os
import json
import sys
import argparse
import numpy as np
from subprocess import DEVNULL, call
from utils import sphere_hammersley_sequence

# Constants for Blender installation
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def _install_blender():
    """
    Check if Blender is installed. If not, download and install it.
    Installs necessary dependencies for Blender to run headless.
    """
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def render_model(file_path, output_dir, num_views=150, resolution=512):
    """
    Render a 3D model from multiple viewpoints.
    
    Args:
        file_path: Path to the GLB file
        output_dir: Directory to save rendered images
        num_views: Number of views to render
        resolution: Image resolution in pixels
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate camera positions using Hammersley sequence for uniform distribution on sphere
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views  # Distance from camera to object
    fov = [40 / 180 * np.pi] * num_views  # Field of view in radians
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    # Prepare arguments for Blender command
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', str(resolution),
        '--output_folder', output_dir,
        '--engine', 'CYCLES',  # Use Cycles rendering engine for better quality
        '--save_mesh',
    ]
    
    # Execute Blender in headless mode with the rendering script
    print(f"Rendering {file_path} to {output_dir}...")
    call(args, stdout=DEVNULL, stderr=DEVNULL)
    
    # Check if rendering was successful
    if os.path.exists(os.path.join(output_dir, 'transforms.json')):
        print("Rendering completed successfully!")
        return True
    else:
        print("Rendering failed!")
        return False


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Render a GLB model from multiple viewpoints')
    parser.add_argument('--input', type=str, required=True, help='Path to the GLB file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the rendered images')
    parser.add_argument('--num_views', type=int, default=150, help='Number of views to render')
    parser.add_argument('--resolution', type=int, default=512, help='Image resolution in pixels')
    opt = parser.parse_args()
    
    # Install Blender if needed
    print('Checking blender...', flush=True)
    _install_blender()

    # Render the model
    render_model(opt.input, opt.output_dir, opt.num_views, opt.resolution)
