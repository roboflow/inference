"""
Visualize triangulated 3D points and camera positions (matplotlib 3D plot).
Outputs the plot as an image for use in workflows or display.
"""

from typing import Any, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    NUMPY_ARRAY_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

# Optional matplotlib; block may run in headless or script context
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


LONG_DESCRIPTION = """
Create a 3D visualization of triangulated points and camera positions. Plots points in the first camera's coordinate frame and draws camera 1 at the origin and camera 2 at its position derived from R and t (camera 2 position in camera 1 frame is -R^T @ t). Outputs the plot as an image (e.g. for display in workflows or saving to file).

## How This Block Works

1. Receives points_3d (list of [x,y,z] or Nx3 array from TriangulationBlockV1), rotation and translation (from EssentialMatrixBlockV1).
2. Builds camera 2 position in camera 1 frame: cam2_pos = -R^T @ t.
3. Uses matplotlib to create a 3D scatter of points and markers for camera positions.
4. Renders the figure to an RGB image and returns it as WorkflowImageData.
"""

SHORT_DESCRIPTION = "Visualize triangulated 3D points and camera positions (matplotlib 3D → image)."


def _points_to_array(points_3d: Any) -> np.ndarray:
    """Convert points_3d (list of [x,y,z] or Nx3 array) to Nx3 numpy array."""
    if isinstance(points_3d, np.ndarray):
        arr = np.asarray(points_3d, dtype=np.float64)
        return arr.reshape(-1, 3) if arr.size else np.zeros((0, 3))
    if isinstance(points_3d, (list, tuple)):
        if not points_3d:
            return np.zeros((0, 3))
        first = points_3d[0]
        if isinstance(first, (list, tuple)) and len(first) >= 3:
            return np.array(
                [[float(p[0]), float(p[1]), float(p[2])] for p in points_3d]
            )
        if isinstance(first, (int, float)):
            return np.array(points_3d, dtype=np.float64).reshape(-1, 3)
    return np.zeros((0, 3))


def render_sfm_3d_plot(
    points_3d: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    figsize: tuple = (10, 8),
    dpi: int = 100,
    point_size: float = 2.0,
    camera_scale: float = 0.5,
) -> np.ndarray:
    """
    Render 3D points and two camera positions to an RGB image.

    - points_3d: Nx3 array in camera 1 frame.
    - rotation: 3x3 (camera 2 w.r.t. camera 1).
    - translation: 3-vector (up to scale).
    - Returns: HxWx3 uint8 RGB image.
    """
    if not _HAS_MATPLOTLIB:
        return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)

    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).ravel()[:3]
    cam2_pos = -R.T @ t
    # Scale for visibility if t is normalized
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-8:
        cam2_pos = cam2_pos / t_norm * camera_scale

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    if points_3d.size > 0:
        ax.scatter(
            points_3d[:, 0],
            points_3d[:, 1],
            points_3d[:, 2],
            c="b",
            s=point_size,
            alpha=0.6,
        )

    ax.scatter(0, 0, 0, c="r", s=80, marker="^", label="Cam 1")
    ax.scatter(
        cam2_pos[0], cam2_pos[1], cam2_pos[2],
        c="g", s=80, marker="^", label="Cam 2",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Equal aspect and auto limits
    if points_3d.size > 0:
        all_pts = np.vstack([points_3d, np.zeros((1, 3)), cam2_pos.reshape(1, 3)])
    else:
        all_pts = np.array([[0, 0, 0], cam2_pos])
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    margin = max(np.ptp(all_pts) * 0.2, 0.1)
    ax.set_xlim(mn[0] - margin, mx[0] + margin)
    ax.set_ylim(mn[1] - margin, mx[1] + margin)
    ax.set_zlim(mn[2] - margin, mx[2] + margin)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape((h, w, 4))[:, :, :3].copy()
    plt.close(fig)
    return buf


class SfMVisualization3DBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "SfM 3D Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "ui_manifest": {
                "section": "visualizations",
                "icon": "far fa-cube",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/sfm_3d_visualization@v1"]

    points_3d: Selector(kind=[LIST_OF_VALUES_KIND, NUMPY_ARRAY_KIND]) = Field(
        description="Triangulated 3D points from TriangulationBlockV1 (list of [x,y,z] or Nx3 array, camera 1 frame).",
        examples=["$steps.triangulation.points_3d"],
    )
    rotation: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="3x3 rotation from EssentialMatrixBlockV1.",
        examples=["$steps.essential_matrix.rotation"],
    )
    translation: Selector(kind=[NUMPY_ARRAY_KIND]) = Field(
        description="3-vector translation from EssentialMatrixBlockV1.",
        examples=["$steps.essential_matrix.translation"],
    )
    figsize_width: Union[int, float, Selector] = Field(
        default=10,
        description="Figure width in inches.",
    )
    figsize_height: Union[int, float, Selector] = Field(
        default=8,
        description="Figure height in inches.",
    )
    dpi: Union[int, Selector] = Field(
        default=100,
        description="Figure DPI for output image resolution.",
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
        ]


class SfMVisualization3DBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SfMVisualization3DBlockManifest

    def run(
        self,
        points_3d: Any,
        rotation: np.ndarray,
        translation: np.ndarray,
        figsize_width: Union[int, float] = 10,
        figsize_height: Union[int, float] = 8,
        dpi: int = 100,
    ) -> BlockResult:
        pts = _points_to_array(points_3d)
        figsize = (float(figsize_width), float(figsize_height))
        img = render_sfm_3d_plot(pts, rotation, translation, figsize=figsize, dpi=dpi)
        parent_metadata = ImageParentMetadata(parent_id="sfm_3d_visualization")
        output_image = WorkflowImageData(
            parent_metadata=parent_metadata,
            numpy_image=img,
        )
        return {"image": output_image}
