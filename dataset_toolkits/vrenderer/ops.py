import math
import numpy as np

from mathutils import Vector


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    assert points.shape[-1] == 3
    batched = True
    if points.ndim == 1:
        batched = False
        points = points[None, :]
    points = (matrix[:3, :3] @ points.T + matrix[:3, 3:4]).T
    if not batched:
        points = points[0]
    return points


def polar_to_transform_matrix(
    elevation_deg: float,
    azimuth_deg: float,
    distance: float
):
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_position = [
        distance * math.cos(elevation) * math.cos(azimuth),
        distance * math.cos(elevation) * math.sin(azimuth),
        distance * math.sin(elevation),
    ]
    # looking at world center
    camera_rotation = (-Vector(camera_position)).to_track_quat('-Z', 'Y').to_matrix()
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = camera_rotation
    c2w[:3, 3] = camera_position
    return c2w

def position_to_transform_matrix(camera_position: list[float]) -> np.ndarray:
    camera_rotation = (-Vector(camera_position)).to_track_quat('-Z', 'Y').to_matrix()
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = camera_rotation
    c2w[:3, 3] = camera_position
    return c2w