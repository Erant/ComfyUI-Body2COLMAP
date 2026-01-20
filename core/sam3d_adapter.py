"""Adapter for converting SAM3D-Body output to body2colmap Scene."""

import numpy as np
import torch
from body2colmap.scene import Scene
from body2colmap.coordinates import sam3d_to_world


def sam3d_output_to_scene(mesh_data: dict, include_skeleton: bool = True) -> Scene:
    """
    Convert SAM3D_OUTPUT dict to body2colmap Scene.

    SAM3D uses MHR coordinates (Y-down, Z-forward).
    We convert to world coordinates (Y-up, Z-backward) for internal processing.

    Args:
        mesh_data: SAM3D_OUTPUT dict from ComfyUI-SAM3DBody containing:
            - vertices: (N, 3) mesh vertices
            - faces: (F, 3) face indices
            - camera: (3,) camera translation vector
            - joints: (70, 3) MHR70 joint positions (optional)
        include_skeleton: Whether to include skeleton from joints

    Returns:
        Scene object in world coordinates
    """
    # Extract and convert tensors to numpy
    vertices = _to_numpy(mesh_data["vertices"])
    faces = _to_numpy(mesh_data["faces"])
    cam_t = _to_numpy(mesh_data["camera"])

    # Convert from SAM3D/MHR coordinates to world coordinates
    # MHR: Y-down, Z-forward → World: Y-up, Z-backward
    vertices_world = sam3d_to_world(vertices, cam_t)

    # Extract skeleton if requested
    skeleton_joints = None
    skeleton_format = None
    if include_skeleton and "joints" in mesh_data and mesh_data["joints"] is not None:
        joints = _to_numpy(mesh_data["joints"])

        # SAM3D outputs MHR70 format (70 joints) directly
        # Just convert coordinates from SAM3D to world
        skeleton_joints = sam3d_to_world(joints, cam_t)
        skeleton_format = "mhr70"

    # Create Scene with vertices and faces directly
    return Scene(
        vertices=vertices_world,
        faces=faces,
        skeleton_joints=skeleton_joints,
        skeleton_format=skeleton_format
    )


def _to_numpy(tensor_or_array) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy.

    Args:
        tensor_or_array: torch.Tensor or np.ndarray

    Returns:
        numpy array
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().numpy().astype(np.float32)
    return np.asarray(tensor_or_array, dtype=np.float32)
