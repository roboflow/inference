from re import I
import numpy as np
from plyfile import PlyData
import plotly.graph_objects as go
import click
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image


# -------------------------------------------------
# Load gaussian splats from PLY
# -------------------------------------------------
def load_gaussians_from_ply(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 3D Gaussian splat parameters from a PLY file.

    Args:
        path: Path to the PLY file containing Gaussian splat data.

    Returns:
        A tuple of (means, scales, rots, opacity, colors):
            - means: (N, 3) array of 3D positions (x, y, z).
            - scales: (N, 3) array of scale factors per axis.
            - rots: (N, 4) array of quaternion rotations (w, x, y, z).
            - opacity: (N,) array of opacity values.
            - colors: (N, 3) array of RGB colors (normalized 0–1).
    """
    ply = PlyData.read(path)
    v = ply["vertex"].data

    means = np.vstack([v["x"], v["y"], v["z"]]).T

    # opacity
    opacity = v["opacity"] if "opacity" in v.dtype.names else np.ones(len(means))

    # scales
    scales = np.vstack([v["scale_0"], v["scale_1"], v["scale_2"]]).T

    # quaternion rotation
    rots = np.vstack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]]).T

    # DC SH color
    colors = np.vstack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]]).T
    colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-8)

    return means, scales, rots, opacity, colors


# -------------------------------------------------
# Quaternion → rotation matrix
# -------------------------------------------------
def quat_to_rot(q: np.ndarray | tuple[float, float, float, float]) -> np.ndarray:
    """Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Args:
        q: Quaternion as (w, x, y, z) array or tuple.

    Returns:
        A 3x3 rotation matrix.
    """
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# -------------------------------------------------
# Build covariance from scale + rotation
# -------------------------------------------------
def build_covariances(scales: np.ndarray, rots: np.ndarray) -> np.ndarray:
    """Build 3D covariance matrices from scale and quaternion rotation.

    Each covariance is computed as R @ diag(s²) @ R.T where R is the rotation
    matrix from the quaternion and s is the scale vector.

    Args:
        scales: (N, 3) array of scale factors per Gaussian.
        rots: (N, 4) array of quaternion rotations per Gaussian.

    Returns:
        (N, 3, 3) array of 3D covariance matrices.
    """
    covs = []
    for s, q in zip(scales, rots):
        R = quat_to_rot(q)
        S = np.diag(np.exp(s)**2)
        covs.append(R @ S @ R.T)
    return np.array(covs)

import torch


def render_gaussians(
    means, covs, colors, opacity, K,
    R_obj, t_obj, scale,
    R_cam, t_cam,
    H, W,
    device="cpu"
):
    means = torch.tensor(means, device=device, dtype=torch.float32)
    covs = torch.tensor(covs, device=device, dtype=torch.float32)
    colors = torch.tensor(colors, device=device, dtype=torch.float32)

    opacity = torch.tensor(opacity, device=device, dtype=torch.float32)
    opacity = torch.sigmoid(opacity)

    K = torch.tensor(K, device=device, dtype=torch.float32)
    R_obj = torch.tensor(R_obj, device=device, dtype=torch.float32)
    t_obj = torch.tensor(t_obj, device=device, dtype=torch.float32)
    R_cam = torch.tensor(R_cam, device=device, dtype=torch.float32)
    t_cam = torch.tensor(t_cam, device=device, dtype=torch.float32)

    # -----------------------------
    # Object → world transform
    # -----------------------------
    means_w = (R_obj @ means.T).T * scale + t_obj
    covs_w = scale**2 * torch.einsum("ij,njk,kl->nil", R_obj, covs, R_obj.T)

    # -----------------------------
    # World → camera transform
    # -----------------------------
    means_cam = (R_cam @ means_w.T).T + t_cam
    covs_cam = torch.einsum("ij,njk,kl->nil", R_cam, covs_w, R_cam.T)

    # -----------------------------
    # Projection
    # -----------------------------
    z = means_cam[:,2:3]
    pts_norm = means_cam[:,:2]/z
    pts_img = (K[:2,:2] @ pts_norm.T).T + K[:2,2]

    # grid
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    grid = torch.stack([xs, ys], dim=-1).float()

    img = torch.zeros(H, W, 3, device=device)
    T = torch.ones(H, W, device=device)

    order = torch.argsort(means_cam[:,2], descending=True)

    for idx in tqdm(order):
        mu = means_cam[idx]
        x,y,z = mu
        fx, fy = K[0,0], K[1,1]

        J = torch.tensor([[fx/z,0,-fx*x/(z*z)],
                          [0,fy/z,-fy*y/(z*z)]], device=device)

        Sigma2D = J @ covs_cam[idx] @ J.T
        invS = torch.linalg.inv(Sigma2D)

        center = pts_img[idx]

        d = grid - center
        d = d.reshape(-1,2)

        w = torch.exp(-0.5*(d @ invS * d).sum(dim=1))
        w = w.reshape(H,W)

        a = opacity[idx]*w
        img += (T.unsqueeze(-1)*a.unsqueeze(-1)*colors[idx])
        T *= (1-a)

    return img.clamp(0,1).cpu().numpy()


def show_plotly(img: np.ndarray) -> None:
    """Display an image in a Plotly figure (600x600).

    Args:
        img: (H, W, 3) image array with values in [0, 1].
    """
    fig = go.Figure(go.Image(z=(img * 255).astype(np.uint8)))
    fig.update_layout(width=600, height=600)
    fig.show()


@click.command()
@click.option(
    "--file-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the PLY file to render.",
)
@click.option(
    "--image-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the image to render into.",
)
@click.option(
    "--room-axes-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the room axes file.",
)
@click.option(
    "--room-length",
    type=float,
    required=True,
    help="Length of the room.",
)
@click.option(
    "--sofa-length",
    type=float,
    required=True,
    help="Length of the sofa.",
)
def main(
    file_path: str | Path,
    image_path: str | Path,
    room_axes_path: str | Path,
    room_length: float,
    sofa_length: float,
) -> None:
    """Load Gaussian splats from a PLY file and render them.

    Args:
        file_path: Path to the PLY file containing Gaussian splat data.
    """
    means, scales, rots, opacity, colors = load_gaussians_from_ply(str(file_path))
    covs = build_covariances(scales, rots)

    image = Image.open(image_path)

    H, W = 512, 512
    fx = fy = 250

    K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]])

    R_obj = np.eye(3)
    t_obj = np.zeros(3)
    scale = 1.0

    R_cam = np.eye(3)
    d = 2.5 * np.linalg.norm(means, axis=1).max()
    t_cam = np.array([0,0,d])

    img = render_gaussians(means, covs, colors, opacity, K, R_obj, t_obj, scale, R_cam, t_cam, H, W, device="cpu")
    show_plotly(img)


if __name__ == "__main__":
    main()
