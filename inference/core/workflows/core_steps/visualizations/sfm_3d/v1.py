"""
Visualize triangulated 3D points and camera positions (Plotly 3D plot).
Outputs the plot as an image for use in workflows or display.
Requires: plotly, kaleido (for static image export).
"""

import io
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

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


LONG_DESCRIPTION = """
Create a 3D visualization of triangulated points and camera positions using Plotly. Plots points in the first camera's coordinate frame and draws camera 1 at the origin and camera 2 at its position derived from R and t (camera 2 position in camera 1 frame is -R^T @ t). Outputs the plot as an image (e.g. for display in workflows or saving to file). Requires plotly and kaleido for static image export.
"""

SHORT_DESCRIPTION = "Visualize triangulated 3D points and camera positions (Plotly 3D → image)."


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


def create_sfm_3d_figure(
    points_3d: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    point_size: float = 2.0,
    camera_scale: float = 0.5,
):
    """
    Create a Plotly 3D figure of triangulated points and camera positions.

    - points_3d: Nx3 array in camera 1 frame.
    - rotation: 3x3 (camera 2 w.r.t. camera 1).
    - translation: 3-vector (up to scale).
    - Returns: plotly.go.Figure, or None if plotly is not available.
    """
    if not _HAS_PLOTLY:
        return None

    R = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).ravel()[:3]
    cam2_pos = -R.T @ t
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-8:
        cam2_pos = cam2_pos / t_norm * camera_scale

    traces = []

    if points_3d.size > 0:
        traces.append(
            go.Scatter3d(
                x=points_3d[:, 0].tolist(),
                y=points_3d[:, 1].tolist(),
                z=points_3d[:, 2].tolist(),
                mode="markers",
                marker=dict(size=point_size, color="royalblue", opacity=0.6),
                name="Points",
            )
        )

    traces.append(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers+text",
            marker=dict(size=10, color="red", symbol="diamond"),
            text=["Cam 1"],
            textposition="top center",
            name="Cam 1",
        )
    )
    traces.append(
        go.Scatter3d(
            x=[cam2_pos[0]],
            y=[cam2_pos[1]],
            z=[cam2_pos[2]],
            mode="markers+text",
            marker=dict(size=10, color="green", symbol="diamond"),
            text=["Cam 2"],
            textposition="top center",
            name="Cam 2",
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        showlegend=True,
    )
    return fig


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
    Render 3D points and two camera positions to an RGB image using Plotly.

    - points_3d: Nx3 array in camera 1 frame.
    - rotation: 3x3 (camera 2 w.r.t. camera 1).
    - translation: 3-vector (up to scale).
    - Returns: HxWx3 uint8 RGB image.
    """
    width_px = int(figsize[0] * dpi)
    height_px = int(figsize[1] * dpi)
    placeholder = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    fig = create_sfm_3d_figure(
        points_3d, rotation, translation,
        point_size=point_size, camera_scale=camera_scale,
    )
    if fig is None:
        return placeholder

    try:
        img_bytes = fig.to_image(format="png", width=width_px, height=height_px)
    except Exception:
        return placeholder

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        arr = np.array(img)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr
    except Exception:
        return placeholder


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
