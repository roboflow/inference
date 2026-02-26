import json
from pathlib import Path

import click
import numpy as np
from sklearn.cluster import KMeans


def cluster_normals(planes: dict[int, tuple[np.ndarray, float, int]], alpha: float = 50):
    # collect normals + weights
    normals = []
    weights = []

    for v_id, v in planes.items():
        n = np.array(v["n"])
        normals.append(n)
        weights.append(v["n_points"])

    normals = np.array(normals)
    weights = np.array(weights)

    # normalize normals
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # weighted kmeans (hack: repeat large planes)
    repeat = (weights / weights.max() * alpha).astype(int) + 1
    expanded = np.repeat(normals, repeat, axis=0)

    kmeans = KMeans(n_clusters=3, random_state=0).fit(expanded)
    centers = kmeans.cluster_centers_
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    return centers


def identify_floor(centers: np.ndarray):
    ys = np.abs(centers[:,1])
    floor_idx = np.argmax(ys)
    y_axis = centers[floor_idx]

    if y_axis[1] < 0:
        y_axis = -y_axis

    return floor_idx, y_axis


def identify_forward_wall(centers: np.ndarray, floor_idx: int):
    candidates = [i for i in range(centers.shape[0]) if i != floor_idx]
    forward_idx = max(candidates, key=lambda i: abs(centers[i][2]))
    z_axis = centers[forward_idx]
    return forward_idx, z_axis


def compute_lateral_axis(y_axis: np.ndarray, z_axis: np.ndarray):
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    return x_axis


def reorthogonalize_z_axis(x_axis: np.ndarray, y_axis: np.ndarray):
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    return z_axis


def dominant_plane_for_axis(axis: np.ndarray, planes: dict[int, tuple[np.ndarray, float, int]]):
    best = None
    best_score = 0
    for k,v in planes.items():
        n = np.array(v["n"])
        score = abs(np.dot(n, axis)) * v["n_points"]
        if score > best_score:
            best = (k,n,v["d"])
            best_score = score
    return best


def solve_corner_intersection(n1: np.ndarray, d1: float, n2: np.ndarray, d2: float, n3: np.ndarray, d3: float):
    A = np.stack([n1,n2,n3])
    b = -np.array([d1,d2,d3])
    corner = np.linalg.solve(A,b)
    return corner


@click.command()
@click.option("--planes-path", type=click.Path(exists=True), required=True)
@click.option("--output-path", required=True)
def main(planes_path: Path, output_path: Path):
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
    
    corner = solve_corner_intersection(floor_plane[1], floor_plane[2], forward_plane[1], forward_plane[2], lateral_plane[1], lateral_plane[2])

    R_room = np.stack([x_axis, y_axis, z_axis], axis=1)
    t_corner = corner

    print(x_axis @ y_axis)
    print(y_axis @ z_axis)
    print(x_axis @ z_axis)
    print(np.linalg.det(R_room))

    with open(output_path, "w") as f:
        json.dump({
            "R_room": R_room.tolist(),
            "t_corner": t_corner.tolist()
        }, f)

if __name__ == "__main__":
    main()
