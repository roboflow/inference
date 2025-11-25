# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from pytorch3d.viz.plotly_vis which has license:
# BSD License

# For PyTorch3D software

# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Meta nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

import math

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from pytorch3d.renderer import (
    HeterogeneousRayBundle,
    RayBundle,
    TexturesAtlas,
    TexturesVertex,
    ray_bundle_to_ray_points,
)
from pytorch3d.renderer.camera_utils import camera_to_eye_at_up
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import (
    Meshes,
    Pointclouds,
    join_meshes_as_scene,
)
from pytorch3d.vis.plotly_vis import (
    AxisArgs,
    Lighting,
    _add_camera_trace,
    _add_pointcloud_trace,
    _add_ray_bundle_trace,
    _is_ray_bundle,
    _scale_camera_to_bounds,
    _update_axes_bounds,
)


Struct = Union[CamerasBase, Meshes, Pointclouds, RayBundle, HeterogeneousRayBundle]


default_axisargs = dict(
    xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
    yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
    zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
    axis_args=AxisArgs(showgrid=True),
)

NO_BACKGROUND = dict(
    xaxis={"backgroundcolor": "rgb(255, 255, 255)", "visible": False},
    yaxis={"backgroundcolor": "rgb(255, 255, 255)", "visible": False},
    zaxis={"backgroundcolor": "rgb(255, 255, 255)", "visible": False},
)


@torch.no_grad()
def plot_tdfy_scene(
    plots: Dict[str, Dict[str, Struct]],
    *,
    viewpoint_cameras: Optional[CamerasBase] = None,
    ncols: int = 1,
    camera_scale: float = 0.3,
    camera_wireframe_width: int = 3,
    pointcloud_max_points: int = 20000,
    pointcloud_marker_size: int = 1,
    raybundle_max_rays: int = 20000,
    raybundle_max_points_per_ray: int = 1000,
    raybundle_ray_point_marker_size: int = 1,
    raybundle_ray_line_width: int = 1,
    boxes_wireframe_width: int = 1,
    boxes_add_cross_face_bars: bool = False,
    boxes_name_int_to_display_name_dict: Optional[Dict[int, str]] = None,
    boxes_plot_together: bool = False,
    height: int = None,
    width: int = None,
    use_orthographic: bool = False,
    equalticks: bool = True,
    ticklen: float = 1.0,
    aspectmode: str = "cube",
    **kwargs,
):  # pragma: no cover
    """
    Main function to visualize Cameras, Meshes, Pointclouds, and RayBundle.
    Plots input Cameras, Meshes, Pointclouds, and RayBundle data into named subplots,
    with named traces based on the dictionary keys. Cameras are
    rendered at the camera center location using a wireframe.

    Args:
        plots: A dict containing subplot and trace names,
            as well as the Meshes, Cameras and Pointclouds objects to be rendered.
            See below for examples of the format.
        viewpoint_cameras: an instance of a Cameras object providing a location
            to view the plotly plot from. If the batch size is equal
            to the number of subplots, it is a one to one mapping.
            If the batch size is 1, then that viewpoint will be used
            for all the subplots will be viewed from that point.
            Otherwise, the viewpoint_cameras will not be used.
        ncols: the number of subplots per row
        camera_scale: determines the size of the wireframe used to render cameras.
        pointcloud_max_points: the maximum number of points to plot from
            a pointcloud. If more are present, a random sample of size
            pointcloud_max_points is used.
        pointcloud_marker_size: the size of the points rendered by plotly
            when plotting a pointcloud.
        raybundle_max_rays: maximum number of rays of a RayBundle to visualize. Randomly
            subsamples without replacement in case the number of rays is bigger than max_rays.
        raybundle_max_points_per_ray: the maximum number of points per ray in RayBundle
            to visualize. If more are present, a random sample of size
            max_points_per_ray is used.
        raybundle_ray_point_marker_size: the size of the ray points of a plotted RayBundle
        raybundle_ray_line_width: the width of the plotted rays of a RayBundle
        **kwargs: Accepts lighting (a Lighting object) and any of the args xaxis,
            yaxis and zaxis which Plotly's scene accepts. Accepts axis_args,
            which is an AxisArgs object that is applied to all 3 axes.
            Example settings for axis_args and lighting are given at the
            top of this file.

    Example:

    ..code-block::python

        mesh = ...
        point_cloud = ...
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        })
        fig.show()

    The above example will render one subplot which has both a mesh and pointcloud.

    If the Meshes, Pointclouds, or Cameras objects are batched, then every object in that batch
    will be plotted in a single trace.

    ..code-block::python
        mesh = ... # batch size 2
        point_cloud = ... # batch size 2
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        })
        fig.show()

    The above example renders one subplot with 2 traces, each of which renders
    both objects from their respective batched data.

    Multiple subplots follow the same pattern:
    ..code-block::python
        mesh = ... # batch size 2
        point_cloud = ... # batch size 2
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh[0],
                "pointcloud_trace_title": point_cloud[0]
            },
            "subplot2_title": {
                "mesh_trace_title": mesh[1],
                "pointcloud_trace_title": point_cloud[1]
            }
        },
        ncols=2)  # specify the number of subplots per row
        fig.show()

    The above example will render two subplots, each containing a mesh
    and a pointcloud. The ncols argument will render two subplots in one row
    instead of having them vertically stacked because the default is one subplot
    per row.

    To view plotly plots from a PyTorch3D camera's point of view, we can use
    viewpoint_cameras:
    ..code-block::python
        mesh = ... # batch size 2
        R, T = look_at_view_transform(2.7, 0, [0, 180]) # 2 camera angles, front and back
        # Any instance of CamerasBase works, here we use FoVPerspectiveCameras
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh[0]
            },
            "subplot2_title": {
                "mesh_trace_title": mesh[1]
            }
        },
        viewpoint_cameras=cameras)
        fig.show()

    The above example will render the first subplot seen from the camera on the +z axis,
    and the second subplot from the viewpoint of the camera on the -z axis.

    We can visualize these cameras as well:
    ..code-block::python
        mesh = ...
        R, T = look_at_view_transform(2.7, 0, [0, 180]) # 2 camera angles, front and back
        # Any instance of CamerasBase works, here we use FoVPerspectiveCameras
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        fig = plot_scene({
            "subplot1_title": {
                "mesh_trace_title": mesh,
                "cameras_trace_title": cameras,
            },
        })
        fig.show()

    The above example will render one subplot with the mesh object
    and two cameras.

    RayBundle visualization is also supproted:
    ..code-block::python
        cameras = PerspectiveCameras(...)
        ray_bundle = RayBundle(origins=..., lengths=..., directions=..., xys=...)
        fig = plot_scene({
            "subplot1_title": {
                "ray_bundle_trace_title": ray_bundle,
                "cameras_trace_title": cameras,
            },
        })
        fig.show()

    For an example of using kwargs, see below:
    ..code-block::python
        mesh = ...
        point_cloud = ...
        fig = plot_scene({
            "subplot_title": {
                "mesh_trace_title": mesh,
                "pointcloud_trace_title": point_cloud
            }
        },
        axis_args=AxisArgs(backgroundcolor="rgb(200,230,200)")) # kwarg axis_args
        fig.show()

    The above example will render each axis with the input background color.

    See the tutorials in pytorch3d/docs/tutorials for more examples
    (namely rendered_color_points.ipynb and rendered_textured_meshes.ipynb).
    """

    subplots = list(plots.keys())
    fig = _gen_fig_with_subplots(len(subplots), ncols, subplots)
    lighting = kwargs.get("lighting", Lighting())._asdict()
    axis_args_dict = kwargs.get("axis_args", AxisArgs(showgrid=True))._asdict()

    # Set axis arguments to defaults defined at the top of this file
    x_settings = {**axis_args_dict}
    y_settings = {**axis_args_dict}
    z_settings = {**axis_args_dict}

    # Update the axes with any axis settings passed in as kwargs.
    x_settings.update(**kwargs.get("xaxis", {"backgroundcolor": "rgb(200, 200, 230)"}))
    y_settings.update(**kwargs.get("yaxis", {"backgroundcolor": "rgb(230, 200, 200)"}))
    z_settings.update(**kwargs.get("zaxis", {"backgroundcolor": "rgb(200, 230, 200)"}))
    camera = {
        "up": {
            "x": 0.0,
            "y": 0.0,
            "z": 1.0,
        }  # set the up vector to match PyTorch3D world coordinates conventions
    }
    viewpoints_eye_at_up_world = None
    if viewpoint_cameras:
        n_viewpoint_cameras = len(viewpoint_cameras)
        if n_viewpoint_cameras == len(subplots) or n_viewpoint_cameras == 1:
            # Calculate the vectors eye, at, up in world space
            # to initialize the position of the camera in
            # the plotly figure
            viewpoints_eye_at_up_world = camera_to_eye_at_up(
                viewpoint_cameras.get_world_to_view_transform().cpu()
            )
        else:
            msg = "Invalid number {} of viewpoint cameras were provided. Either 1 \
            or {} cameras are required".format(
                len(viewpoint_cameras), len(subplots)
            )
            warnings.warn(msg)

    for subplot_idx in range(len(subplots)):
        subplot_name = subplots[subplot_idx]
        traces = plots[subplot_name]
        for trace_name, struct in traces.items():
            if isinstance(struct, Meshes):
                _add_mesh_trace(fig, struct, trace_name, subplot_idx, ncols, lighting)
            elif isinstance(struct, Pointclouds):
                _add_pointcloud_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    pointcloud_max_points,
                    pointcloud_marker_size,
                )
            elif isinstance(struct, CamerasBase):
                _add_camera_trace(
                    fig, struct, trace_name, subplot_idx, ncols, camera_scale
                )
            elif isinstance(struct, CamTrace):
                struct._add_camera_trace(
                    fig=fig,
                    trace_name=trace_name,
                    subplot_idx=subplot_idx,
                    ncols=ncols,
                    camera_wireframe_width=camera_wireframe_width,
                )
            elif _is_ray_bundle(struct):
                _add_ray_bundle_trace(
                    fig,
                    struct,
                    trace_name,
                    subplot_idx,
                    ncols,
                    raybundle_max_rays,
                    raybundle_max_points_per_ray,
                    raybundle_ray_point_marker_size,
                    raybundle_ray_line_width,
                )
            else:
                raise ValueError(
                    "struct {} is not a Cameras, Meshes, BBoxes3D, Pointclouds,".format(
                        struct
                    )
                    + "RayBundle or HeterogeneousRayBundle object."
                )

        # Ensure update for every subplot.
        plot_scene = "scene" + str(subplot_idx + 1)
        current_layout = fig["layout"][plot_scene]
        xaxis = current_layout["xaxis"]
        yaxis = current_layout["yaxis"]
        zaxis = current_layout["zaxis"]

        # mins = min([axis['range'][0] for axis in (xaxis, yaxis, zaxis)])
        # maxes = max([axis['range'][1] for axis in (xaxis, yaxis, zaxis)])
        # xaxis['range'] = [mins, maxes]
        # yaxis['range'] = [mins, maxes]
        # zaxis['range'] = [mins, maxes]
        maxlen = max(
            [abs(axis["range"][1] - axis["range"][0]) for axis in (xaxis, yaxis, zaxis)]
        )
        halflen = maxlen / 2.0
        nticks = math.ceil(maxlen / ticklen)
        xaxis["range"] = [
            sum(xaxis["range"]) / 2.0 + delta for delta in [-halflen, halflen]
        ]
        yaxis["range"] = [
            sum(yaxis["range"]) / 2.0 + delta for delta in [-halflen, halflen]
        ]
        zaxis["range"] = [
            sum(zaxis["range"]) / 2.0 + delta for delta in [-halflen, halflen]
        ]

        xaxis["nticks"] = nticks
        yaxis["nticks"] = nticks
        zaxis["nticks"] = nticks

        # Update the axes with our above default and provided settings.
        xaxis.update(**x_settings)
        yaxis.update(**y_settings)
        zaxis.update(**z_settings)

        # update camera viewpoint if provided
        if viewpoints_eye_at_up_world is not None:
            # Use camera params for batch index or the first camera if only one provided.
            viewpoint_idx = min(n_viewpoint_cameras - 1, subplot_idx)

            eye, at, up = (i[viewpoint_idx] for i in viewpoints_eye_at_up_world)
            eye_x, eye_y, eye_z = eye.tolist()
            at_x, at_y, at_z = at.tolist()
            up_x, up_y, up_z = up.tolist()

            # scale camera eye to plotly [-1, 1] ranges
            x_range = xaxis["range"]
            y_range = yaxis["range"]
            z_range = zaxis["range"]

            eye_x = _scale_camera_to_bounds(eye_x, x_range, True)
            eye_y = _scale_camera_to_bounds(eye_y, y_range, True)
            eye_z = _scale_camera_to_bounds(eye_z, z_range, True)

            at_x = _scale_camera_to_bounds(at_x, x_range, True)
            at_y = _scale_camera_to_bounds(at_y, y_range, True)
            at_z = _scale_camera_to_bounds(at_z, z_range, True)

            up_x = _scale_camera_to_bounds(up_x, x_range, False)
            up_y = _scale_camera_to_bounds(up_y, y_range, False)
            up_z = _scale_camera_to_bounds(up_z, z_range, False)

            camera["eye"] = {"x": eye_x, "y": eye_y, "z": eye_z}
            camera["center"] = {"x": at_x, "y": at_y, "z": at_z}
            camera["up"] = {"x": up_x, "y": up_y, "z": up_z}
            camera["projection"] = {"type": "orthographic"}

        current_layout.update(
            {
                "xaxis": xaxis,
                "yaxis": yaxis,
                "zaxis": zaxis,
                # "aspectmode": "data",
                "aspectmode": aspectmode,
                # "aspectratio": {
                #     'x': 1.0,
                #     'y': 1.0,
                #     'z': 1.0,
                # },
                "camera": camera,
            }
        )
    if width is not None or height is not None:
        fig.update_layout(
            width=width,
            height=height,
            # aspectmode="data"
        )

    if use_orthographic:
        # fig.update_scenes(aspectmode='data')
        fig.layout.scene.camera.projection.type = "orthographic"
    return fig


def _gen_fig_with_subplots(
    batch_size: int,
    ncols: int,
    subplot_titles: List[str],
    row_heights: Optional[List[int]] = None,
    column_widths: Optional[List[int]] = None,
):  # pragma: no cover
    """
    Takes in the number of objects to be plotted and generate a plotly figure
    with the appropriate number and orientation of titled subplots.
    Args:
        batch_size: the number of elements in the batch of objects to be visualized.
        ncols: number of subplots in the same row.
        subplot_titles: titles for the subplot(s). list of strings of length batch_size.

    Returns:
        Plotly figure with ncols subplots per row, and batch_size subplots.
    """
    fig_rows = batch_size // ncols
    if batch_size % ncols != 0:
        fig_rows += 1  # allow for non-uniform rows
    fig_cols = ncols
    fig_type = [{"type": "scene"}]
    specs = [fig_type * fig_cols] * fig_rows
    # subplot_titles must have one title per subplot
    fig = make_subplots(
        rows=fig_rows,
        cols=fig_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[1.0] * fig_cols,
    )
    return fig


# From https://github.com/facebookresearch/pytorch3d/blob/0a59450f0ebbe12d9a8db3de937814932517633b/pytorch3d/vis/plotly_vis.py#L634
def _add_mesh_trace(
    fig: go.Figure,
    meshes: Meshes,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    lighting: Lighting,
) -> None:  # pragma: no cover
    """
    Adds a trace rendering a Meshes object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        meshes: Meshes object to render. It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        lighting: a Lighting object that specifies the Mesh3D lighting.
    """

    mesh = join_meshes_as_scene(meshes)
    mesh = mesh.detach().cpu()
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    # If mesh has vertex colors or face colors, use them
    # for figure, otherwise use plotly's default colors.
    verts_rgb = None
    faces_rgb = None
    if isinstance(mesh.textures, TexturesVertex):
        verts_rgb = mesh.textures.verts_features_packed()
        verts_rgb.clamp_(min=0.0, max=1.0)
        verts_rgb = (torch.tensor(255.0) * verts_rgb).to(torch.uint8)
    if isinstance(mesh.textures, TexturesAtlas):
        atlas = mesh.textures.atlas_packed()
        # If K==1
        if atlas.shape[1] == 1 and atlas.shape[3] == 3:
            faces_rgb = atlas[:, 0, 0]

    # Reposition the unused vertices to be "inside" the object
    # (i.e. they won't be visible in the plot).
    verts_used = torch.zeros((verts.shape[0],), dtype=torch.bool)
    verts_used[torch.unique(faces)] = True
    verts_center = verts[verts_used].mean(0)
    verts[~verts_used] = verts_center

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            vertexcolor=verts_rgb,
            facecolor=faces_rgb,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            lighting=lighting,
            name=trace_name,
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    # Access the current subplot's scene configuration
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]

    # update the bounds of the axes for the current trace
    max_expand = (verts.max(0)[0] - verts.min(0)[0]).max()
    _update_axes_bounds(verts_center, max_expand, current_layout)
