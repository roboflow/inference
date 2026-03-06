import math
import os
import numpy as np
from loguru import logger
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots
import torch
from typing import Tuple, Union, Optional, Dict, Any, List


# plot prediction
def plot_logits(
    output: torch.Tensor,
    cutoff: float = 0,
    log_num_points: bool = False,
    **layout_kwargs: Dict[str, Any],
) -> Figure:
    n_voxels = output.shape[-1]
    # if output.ndim == 4:
    #     output = output.unsqueeze(0)
    if output.ndim != 5:
        raise ValueError(f"Output must be 5D, got {output.shape}D")
    coords = torch.argwhere(output > cutoff)[:, [0, 2, 3, 4]].int().cpu().data.numpy()
    if log_num_points:
        logger.info(f"num of points: {len(coords)} ({len(coords)/output.numel():.2%})")
    return plot_points(coords[:, 1:], axis_range=(0, n_voxels - 1), **layout_kwargs)


def plot_points(
    coords: np.ndarray,
    axis_range: Optional[Tuple[int, int]] = (0, 63),
    **layout_kwargs: Dict[str, Any],
) -> Figure:
    import plotly.graph_objects as go

    fig = go.Figure()
    point_size = 1
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers",
            marker=dict(size=point_size, opacity=0.8),
            name=f"pred",
        )
    )
    fig.update_scenes(
        dict(
            xaxis=dict(range=axis_range),
            yaxis=dict(range=axis_range),
            zaxis=dict(range=axis_range),
            aspectmode="cube",
        ),
    )
    if layout_kwargs:
        fig.update_layout(**layout_kwargs)
    return fig


def plot_logits_grid(
    outputs: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    cutoff: float = 0,
    n_cols: int = 2,
    cell_size: int = 350,
    **layout_kwargs: Dict[str, Any],
) -> Figure:
    """
    Create a grid of 3D plots for multiple tensor outputs.

    Args:
        outputs: List of tensor outputs to visualize
        titles: List of titles for each subplot
        cutoff: Threshold value for logits
        n_cols: Number of columns in the grid
        layout_kwargs: Additional layout parameters for the figure

    Returns:
        A plotly Figure with a grid of 3D plots
    """
    if titles is None:
        titles = [f"Output {i}" for i in range(len(outputs))]
    assert len(outputs) == len(titles), "Number of outputs must match number of titles"
    n_plots = len(outputs)
    n_rows = math.ceil(n_plots / n_cols)

    # Create subplot specs for 3D plots
    specs = [[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
    )

    # Get the approximate height and width for the layout
    # height = max(cell_size, min(cell_size * n_rows, 700))
    # width = max(cell_size, min(cell_size * n_cols, 1000))
    height = cell_size * n_rows + 100
    width = cell_size * n_cols + 100

    # Default layout settings
    default_layout = {
        "height": height,
        "width": width,
        "showlegend": False,
        "margin": dict(l=10, r=10, t=30, b=10),
    }

    # Update with any user-provided layout settings
    default_layout.update(layout_kwargs)
    fig.update_layout(**default_layout)

    # Plot each output in its respective subplot
    for i, output in enumerate(outputs):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Create individual figure
        individual_fig = plot_logits(output.unsqueeze(0), cutoff)

        # Add traces to the grid
        for trace in individual_fig.data:
            fig.add_trace(trace, row=row, col=col)

        # Update the scene for this subplot
        n_voxels = output.shape[-1]
        fig.update_scenes(
            dict(
                xaxis=dict(range=[0, n_voxels - 1], showticklabels=False),
                yaxis=dict(range=[0, n_voxels - 1], showticklabels=False),
                zaxis=dict(range=[0, n_voxels - 1], showticklabels=False),
                aspectmode="cube",
                # camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            row=row,
            col=col,
        )

    return fig


def plot_side_by_side(
    pred_output: torch.Tensor,
    gt_output: Union[torch.Tensor, np.ndarray],
    cutoff: float = 0,
    gt_points: bool = False,
    titles: Tuple[str, str] = ("Predicted", "Ground Truth"),
) -> None:
    from plotly.subplots import make_subplots

    pred_fig = plot_logits(pred_output, cutoff, title="Predicted")
    if gt_points:
        gt_fig = plot_points(gt_output, title="Ground Truth", axis_range=None)
    else:
        gt_fig = plot_logits(gt_output, cutoff, title="Ground Truth")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=titles,
    )

    # Copy traces from individual figures
    for trace in pred_fig.data:
        fig.add_trace(trace, row=1, col=1)
    for trace in gt_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # Update layout for both subplots
    fig.update_layout(height=600, width=1200, showlegend=False)

    for i in [1, 2]:
        fig.update_scenes(
            dict(
                xaxis=dict(range=[0, 63]),
                yaxis=dict(range=[0, 63]),
                zaxis=dict(range=[0, 63]),
                aspectmode="cube",
            ),
            row=1,
            col=i,
        )

    if gt_points:
        axis_range = (gt_output.min().item(), gt_output.max().item())
        fig.update_scenes(
            dict(
                xaxis=dict(range=axis_range),
                yaxis=dict(range=axis_range),
                zaxis=dict(range=axis_range),
                aspectmode="cube",
            ),
            row=1,
            col=2,
        )
    return fig
    # fig.show()


# PLY file saving utility function
def save_points_to_ply(points, filename):
    """Save points to a PLY file.

    Args:
        points: Tensor of shape (N, 3) containing point coordinates
        filename: Output filename
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # Write point data
        for i in range(len(points)):
            f.write(
                f"{points[i, 0].item()} {points[i, 1].item()} {points[i, 2].item()}\n"
            )
