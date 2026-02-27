import json
import numpy as np
from plyfile import PlyData
import plotly.graph_objects as go
import click
from pathlib import Path
from tqdm import tqdm
import torch
from pillow_heif import open_heif
from PIL import Image
from collections.abc import Callable

import matplotlib.pyplot as plt

from examples.indoor_design.plane_detection.utils import get_camera_intrinsics_from_exif_in_heic_image
from examples.indoor_design.plane_detection.visualizations import get_point_cloud_3d_fig

C0 = 0.28209479177387814

# -------------------------------------------------
# Load gaussian splats from PLY
# -------------------------------------------------
def load_gaussians_from_ply(
    path: str | Path,
    subsample: float | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load 3D Gaussian splat parameters from a PLY file.

    Args:
        path: Path to the PLY file containing Gaussian splat data.
        subsample: If set, randomly subsample points. Use a float in (0, 1] for
            fraction to keep (e.g. 0.5 = half), or an int >= 1 for max number of points.

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
    harmonics = np.vstack([v["f_dc_2"], v["f_dc_1"], v["f_dc_0"]]).T
    colors = 0.5 + harmonics * C0
    colors = np.clip(colors, 0, 1)

    if subsample is not None:
        n = len(means)
        if isinstance(subsample, float):
            keep = max(1, int(n * subsample))
        else:
            keep = min(n, subsample)
        rng = np.random.default_rng()
        idx = rng.choice(n, size=keep, replace=False)
        means = means[idx]
        scales = scales[idx]
        rots = rots[idx]
        opacity = opacity[idx]
        colors = colors[idx]

    return means, scales, rots, opacity, colors


def get_bbox_and_shift_to_corner(
    means: torch.Tensor,
    corner: str = "min",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 3D axis-aligned bounding box and shift points so origin is at a corner.

    Args:
        means: (N, 3) array of 3D positions.
        corner: Which corner to use as origin. "min" = (min_x, min_y, min_z),
            "max" = (max_x, max_y, max_z). Default "min" (bottom-left-front).

    Returns:
        Tuple of (means_shifted, bbox_min, bbox_max):
            - means_shifted: Points translated so the chosen corner is at origin.
            - bbox_min: (3,) min coordinates (x, y, z) of the original bbox.
            - bbox_max: (3,) max coordinates (x, y, z) of the original bbox.
    """
    bbox_min = torch.tensor([means[:, 0].max(), means[:, 1].min(), means[:, 2].min()], device=means.device, dtype=means.dtype)
    bbox_max = torch.tensor([means[:, 0].max(), means[:, 1].max(), means[:, 2].max()], device=means.device, dtype=means.dtype)

    origin = bbox_min if corner == "min" else bbox_max
    means_shifted = means - origin

    return means_shifted, bbox_min, bbox_max


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
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def reorthonormalize_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Reorthogonalize a rotation matrix.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3, 3) reorthogonalized rotation matrix.
    """
    U, _, Vt = np.linalg.svd(R)
    R_orthonormal = U @ Vt

    if np.linalg.det(R_orthonormal) < 0:
        U[:, -1] *= -1
        R_orthonormal = U @ Vt

    return R_orthonormal


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


def _gaussian_blur_2d(
    x: torch.Tensor, sigma: float, kernel_size: int | None = None
) -> torch.Tensor:
    """Apply 2D Gaussian blur to a tensor of shape (H, W) or (H, W, C)."""
    if kernel_size is None:
        kernel_size = max(3, int(2 * sigma * 4 + 1) | 1)
    pad = kernel_size // 2

    # Build 2D Gaussian kernel
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - pad
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = g / g.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)  # (k, k)

    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
        out = torch.nn.functional.conv2d(x, kernel_2d, padding=pad)
        return out.squeeze(0).squeeze(0)
    else:
        # (H, W, C) -> (1, C, H, W)
        x = x.permute(2, 0, 1).unsqueeze(0)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).expand(x.shape[1], 1, -1, -1)
        out = torch.nn.functional.conv2d(x, kernel_2d, padding=pad, groups=x.shape[1])
        return out.squeeze(0).permute(1, 2, 0)


def render_gaussians(
    means: np.ndarray,
    covs: np.ndarray,
    colors: np.ndarray,
    opacity: np.ndarray,
    K: np.ndarray,
    R_obj: np.ndarray,
    scale: float,
    R_room: np.ndarray,
    t_corner: np.ndarray,
    offset: np.ndarray,
    img: np.ndarray,
    device: str,
    on_progress: Callable[[np.ndarray, int, int], None] | None = None,
    progress_interval_pct: float = 5.0,
    blend_sigma: float = 2.0,
):
    """Render gaussians on an empty canvas and composite over the input image.

    The object is rendered on a zeros plane, then composited so it occludes the
    input image. A small Gaussian blur is applied to the alpha mask at the edges
    for smooth blending.
    """
    means = torch.tensor(means, device=device, dtype=torch.float32)
    covs = torch.tensor(covs, device=device, dtype=torch.float32)
    colors = torch.tensor(colors, device=device, dtype=torch.float32)
    opacity = torch.tensor(opacity, device=device, dtype=torch.float32)
    opacity = torch.sigmoid(opacity)
    offset = torch.tensor(offset, device=device, dtype=torch.float32)
    K = torch.tensor(K, device=device, dtype=torch.float32)
    R_obj = torch.tensor(R_obj, device=device, dtype=torch.float32)
    R_room = torch.tensor(R_room, device=device, dtype=torch.float32)
    t_corner = torch.tensor(t_corner, device=device, dtype=torch.float32)
    scale = torch.tensor(scale, device=device, dtype=torch.float32)

    height, width, _ = img.shape
    img_orig = torch.tensor(img, device=device, dtype=torch.float32) / 255.0

    # Render on empty (zeros) canvas
    rendered = torch.zeros(height, width, 3, device=device, dtype=torch.float32)
    T = torch.ones(height, width, device=device)

    # -----------------------------
    # Object → room transform (canonical object pose + manual placement)
    # -----------------------------
    means_room = (R_obj @ means.T).T * scale
    covs_room = scale**2 * torch.einsum("ij,njk,kl->nil", R_obj, covs, R_obj.T)

    means_room, _, _ = get_bbox_and_shift_to_corner(means_room, corner="min")

    # -----------------------------
    # Room → camera transform
    # -----------------------------
    means_cam = (R_room @ means_room.T).T + t_corner + offset
    covs_cam = torch.einsum("ij,njk,kl->nil", R_room, covs_room, R_room.T)

    # -----------------------------
    # Projection
    # -----------------------------
    z = means_cam[:, 2:3]
    pts_norm = means_cam[:, :2] / z
    pts_img = (K[:2, :2] @ pts_norm.T).T + K[:2, 2]

    # grid
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=-1).float().to(device)

    order = torch.argsort(means_cam[:, 2], descending=True)
    n_total = len(order)
    last_reported_pct = -1.0

    for step, idx in enumerate(tqdm(order)):
        mu = means_cam[idx]
        x, y, z = mu
        fx, fy = K[0, 0], K[1, 1]

        J = torch.tensor(
            [[fx / z, 0, -fx * x / (z * z)], [0, fy / z, -fy * y / (z * z)]],
            device=device,
        )

        Sigma2D = J @ covs_cam[idx] @ J.T
        invS = torch.linalg.inv(Sigma2D)

        center = pts_img[idx]

        d = grid - center
        d = d.reshape(-1, 2)

        w = torch.exp(-0.5 * (d @ invS * d).sum(dim=1))
        w = w.reshape(height, width)

        a = opacity[idx] * w
        rendered += T.unsqueeze(-1) * a.unsqueeze(-1) * colors[idx]
        T *= 1 - a

        if on_progress is not None:
            pct = 100 * (step + 1) / n_total
            if pct - last_reported_pct >= progress_interval_pct or step == n_total - 1:
                last_reported_pct = pct
                alpha = 1 - T
                composite = img_orig * T.unsqueeze(-1) + rendered * alpha.unsqueeze(-1)
                snapshot = composite.clamp(0, 1).cpu().numpy()
                on_progress(snapshot, step + 1, n_total)

    # Alpha mask: where the object was rendered (1 = occluding, 0 = transparent)
    alpha = 1 - T

    # Smooth alpha edges for blending
    if blend_sigma > 0:
        alpha = _gaussian_blur_2d(alpha, sigma=blend_sigma).clamp(0, 1)

    # Composite: rendered occludes input image
    final = img_orig * (1 - alpha).unsqueeze(-1) + rendered.clamp(0, 1) * alpha.unsqueeze(-1)

    return final.clamp(0, 1).cpu().numpy()


def show_plotly(img: np.ndarray) -> None:
    """Display an image in a Plotly figure (600x600).

    Args:
        img: (H, W, 3) image array with values in [0, 1].
    """
    fig = go.Figure(go.Image(z=(img * 255).astype(np.uint8)))
    fig.update_layout(width=600, height=600)
    fig.show()


def make_live_progress_callback() -> Callable[[np.ndarray, int, int], None]:
    """Create a callback that updates a live matplotlib window on each invocation."""

    fig = None
    im = None
    ax = None

    def on_progress(snapshot: np.ndarray, step: int, total: int) -> None:
        nonlocal fig, im, ax
        if fig is None:
            plt.ion()
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(snapshot)
            ax.axis("off")
            fig.canvas.manager.set_window_title("Render progress")
            plt.show(block=False)
        im.set_data(snapshot)
        ax.set_title(f"Rendering: {step}/{total} ({100 * step / total:.0f}%)")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    return on_progress


@click.command()
@click.option(
    "--object-model-file-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the object PLY file containing Gaussian splat data.",
)
@click.option(
    "--image-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the image file to render the object into.",
)
@click.option(
    "--room-axes-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the room axes JSON file.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Device to use for rendering.",
)
@click.option(
    "--output-image-path",
    type=click.Path(),
    required=True,
    help="Path to the output image file.",
)
@click.option(
    "--visualize-progress",
    is_flag=True,
    default=False,
    help="Show live visualization of how the image builds.",
)
@click.option(
    "--progress-interval",
    type=float,
    default=1.0,
    help="Progress interval (in %%) for updating the visualization. Default: 5%%.",
)
@click.option(
    "--blend-sigma",
    type=float,
    default=2.0,
    help="Gaussian sigma for smoothing alpha edges when blending rendered object with image. Default: 2.0.",
)
def main(
    object_model_file_path: str | Path,
    image_path: str | Path,
    room_axes_path: str | Path,
    device: str,
    output_image_path: str | Path,
    visualize_progress: bool,
    progress_interval: float,
    blend_sigma: float,
) -> None:
    """Load Gaussian splats from a PLY file and render them.

    Args:
        object_model_file_path: Path to the PLY file containing Gaussian splat data.
        image_path: Path to the image to render into.
        room_axes_path: Path to the room axes file.
    """
    R_obj = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    offset = np.array([0, 0, 0])
    scale = 0.05

    means, scales, rots, opacity, colors = load_gaussians_from_ply(str(object_model_file_path), subsample=0.4)

    fx, fy, cx, cy = get_camera_intrinsics_from_exif_in_heic_image(str(image_path))
    img = open_heif(image_path).to_pillow()

    # -------------------------------------------------
    # Optional downscale for faster debugging
    # -------------------------------------------------
    resize_factor = 0.5  # TODO: change or expose as CLI arg

    if resize_factor != 1.0:
        new_h = int(img.height * resize_factor)
        new_w = int(img.width * resize_factor)

        img = np.array(img.resize((new_w, new_h)))

        # scale intrinsics accordingly
        fx *= resize_factor
        fy *= resize_factor
        cx *= resize_factor
        cy *= resize_factor

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    with open(room_axes_path, "r") as f:
        room_axes = json.load(f)

    R_room = np.array(room_axes["R_room"])
    t_corner = np.array(room_axes["t_corner"])

    covs = build_covariances(scales, rots)

    on_progress_cb = make_live_progress_callback() if visualize_progress else None
    img = render_gaussians(
        means, covs, colors, opacity, K, R_obj, scale, R_room, t_corner, offset,
        img, device, on_progress=on_progress_cb, progress_interval_pct=progress_interval,
        blend_sigma=blend_sigma,
    )

    Image.fromarray((img * 255).astype(np.uint8)).save(output_image_path)

    if visualize_progress:
        plt.ioff()
        plt.show() 


if __name__ == "__main__":
    main()
