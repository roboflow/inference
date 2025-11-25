import plotly.graph_objects as go
from plotly.graph_objects import Figure
import numpy as np
import os
import imageio
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from PIL import Image
import io
import numpy as np
from tqdm import tqdm


def img_bytes_to_np(img_bytes):
    return np.array(Image.open(io.BytesIO(img_bytes)))


def make_video(
    scene: Figure,
    output_path: str = "scene_video.mp4",
    fps: int = 15,
    duration: int = 1,
    camera_trajectory: Optional[List[Dict[str, Any]]] = None,
    temp_dir: Optional[str] = None,
    trajectory_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Creates a video by updating the camera view location and saving snapshots.

    Args:
        scene: A Plotly Figure object, typically created with plot_tdfy_scene
        output_path: Path to save the output video file
        fps: Frames per second for the output video
        duration: Duration of the video in seconds
        camera_trajectory: List of camera positions. If None, creates a default circular trajectory.
            Each item should be a dict with eye, center, and up keys as expected by Plotly's scene.camera.
        temp_dir: Directory to store temporary frame images. If None, uses ./tmp_frames

    Returns:
        Path to the saved video file
    """
    if not scene._has_subplots():
        raise ValueError("Scene must have subplots to create a video")

    num_frames = fps * duration

    if camera_trajectory is None:
        if trajectory_kwargs is None:
            trajectory_kwargs = {}
        camera_trajectory = _create_default_camera_trajectory(
            num_frames, **trajectory_kwargs
        )

    frames = []
    for i, camera_pos in tqdm(enumerate(camera_trajectory), total=num_frames):
        # update the camera position
        scene.update_scenes(camera=camera_pos)
        img_as_png = scene.to_image(engine="kaleido")
        frames.append(img_bytes_to_np(img_as_png))

    return frames


def _create_default_camera_trajectory(
    num_frames: int,
    axis: str = "y",
    elevation: float = 1.0,
    radius: float = 2.0,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Creates a default camera trajectory, rotating around the scene in a circle.

    Args:
        num_frames: Number of frames in the trajectory
        axis: Axis to rotate around ('x', 'y', or 'z')

    Returns:
        List of camera positions
    """
    trajectory = []

    # Create a circular path
    for i in range(num_frames):
        angle = (i / num_frames) * 2 * np.pi

        # Default position (all zeros)
        eye_x, eye_y, eye_z = 0.0, 0.0, 0.0

        # Calculate camera position based on selected axis
        if axis.lower() == "z":
            # Rotate in the xy-plane (around z-axis)
            eye_x = radius * np.sin(angle)
            eye_y = radius * np.cos(angle)
            eye_z = elevation  # Slightly above the scene
            up = {"x": 0, "y": 0, "z": 1}
        elif axis.lower() == "y":
            # Rotate in the xz-plane (around y-axis)
            eye_x = radius * np.sin(angle)
            eye_z = radius * np.cos(angle)
            eye_y = elevation  # Slightly offset from y-axis
            up = {"x": 0, "y": 1, "z": 0}
        elif axis.lower() == "x":
            # Rotate in the yz-plane (around x-axis)
            eye_y = radius * np.sin(angle)
            eye_z = radius * np.cos(angle)
            eye_x = elevation  # Slightly offset from x-axis
            up = {"x": 1, "y": 0, "z": 0}
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

        camera_pos = {
            "eye": {"x": eye_x, "y": eye_y, "z": eye_z},
            "center": {"x": 0, "y": 0, "z": 0},  # Look at center
            "up": up,  # Orientation based on rotation axis
        }

        trajectory.append(camera_pos)

    return trajectory
