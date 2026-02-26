import json
from pathlib import Path

import click
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from pillow_heif import open_heif
from examples.indoor_design.plane_detection.utils import get_camera_intrinsics_from_exif_in_heic_image


def cluster_normals(planes: dict[int, tuple[np.ndarray, float, int]], alpha: float = 50):
    # collect normals + weights
    normals = []

    for v_id, v in planes.items():
        n = np.array(v["n"])
        normals.append(n)

    normals = np.array(normals)

    # normalize normals
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # weighted kmeans (hack: repeat large planes)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(normals)
    centers = kmeans.cluster_centers_
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    return centers


def identify_floor(centers: np.ndarray):
    ys = np.abs(centers[:,1])
    floor_idx = np.argmax(ys)
    y_axis = centers[floor_idx]
    return floor_idx, y_axis


def identify_forward_wall(centers: np.ndarray, floor_idx: int):
    candidates = [i for i in range(centers.shape[0]) if i != floor_idx]
    forward_idx = max(candidates, key=lambda i: centers[i][2])
    z_axis = centers[forward_idx]
    return forward_idx, z_axis


def compute_lateral_axis(y_axis: np.ndarray, z_axis: np.ndarray):
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    if x_axis[0] > 0:
        x_axis = -x_axis

    return x_axis


def reorthogonalize_z_axis(x_axis: np.ndarray, y_axis: np.ndarray):
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    return z_axis


def dominant_plane_for_axis(axis, planes):
    best = None
    best_score = 0
    for k,v in planes.items():
        n = np.array(v["n"])
        d = v["d"]

        dot = np.dot(n, axis)

        if dot < 0:
            n = -n
            d = -d

        score = abs(dot)

        if score > best_score:
            best = (k,n,d)
            best_score = score

    return best


def solve_corner_intersection(n1: np.ndarray, d1: float, n2: np.ndarray, d2: float, n3: np.ndarray, d3: float):
    A = np.stack([n1,n2,n3])
    b = -np.array([d1,d2,d3])
    corner = np.linalg.solve(A,b)
    return corner


def project_point_in_camera_coordinates(point, fx, fy, cx, cy):
    x, y, z = point
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v


def visualize_image_with_circle(image_path: Path, u: float, v: float, radius: float = 20):
    """Display the image with a circle drawn at the projected point (u, v)."""
    img = np.array(open_heif(image_path).to_pillow())
    theta = np.linspace(0, 2 * np.pi, 50)
    circle_x = u + radius * np.cos(theta)
    circle_y = v + radius * np.sin(theta)
    fig = go.Figure()
    fig.add_trace(go.Image(z=img))
    fig.add_trace(
        go.Scatter(
            x=circle_x,
            y=circle_y,
            mode="lines",
            line=dict(color="red", width=3),
            fill="toself",
            fillcolor="rgba(255,0,0,0.1)",
        )
    )
    fig.update_layout(
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=20, b=0),
    )
    fig.show()


@click.command()
@click.option("--planes-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", required=True)
@click.option("--image-path", type=click.Path(exists=True), required=True)
def main(planes_path: Path, output_path: Path, image_path: Path):
    with open(planes_path, "r") as f:
        planes = json.load(f)

    centers = cluster_normals(planes)

    floor_idx, y_axis = identify_floor(centers)
    _, z_axis = identify_forward_wall(centers, floor_idx)
    x_axis = compute_lateral_axis(y_axis, z_axis)
    z_axis = reorthogonalize_z_axis(x_axis, y_axis)

    floor_plane = dominant_plane_for_axis(y_axis, planes)
    forward_plane = dominant_plane_for_axis(z_axis, planes)
    lateral_plane = dominant_plane_for_axis(x_axis, planes)

    print(f"Plane ids for axes: floor={floor_plane[0]}, forward={forward_plane[0]}, lateral={lateral_plane[0]}")

    corner = solve_corner_intersection(floor_plane[1], floor_plane[2], forward_plane[1], forward_plane[2], lateral_plane[1], lateral_plane[2])

    if np.linalg.det(np.stack([x_axis, y_axis, z_axis], axis=1)) < 0:
        z_axis = -z_axis

    R_room = np.stack([x_axis, y_axis, z_axis], axis=1)
    t_corner = corner

    fx, fy, cx, cy = get_camera_intrinsics_from_exif_in_heic_image(str(image_path))
    u, v = project_point_in_camera_coordinates(corner, fx, fy, cx, cy)

    visualize_image_with_circle(image_path, u, v)

    with open(output_path, "w") as f:
        json.dump({
            "R_room": R_room.tolist(),
            "t_corner": t_corner.tolist()
        }, f)

if __name__ == "__main__":
    main()
