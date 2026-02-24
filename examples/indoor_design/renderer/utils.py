import numpy as np
from plyfile import PlyData
import plotly.graph_objects as go
import click
from pathlib import Path


# -------------------------------------------------
# Load gaussian splats from PLY
# -------------------------------------------------
def load_gaussians_from_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    colors = (colors - colors.min())/(colors.max()-colors.min()+1e-8)

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
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


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
        S = np.diag(s**2)
        covs.append(R @ S @ R.T)
    return np.array(covs)


# -------------------------------------------------
# Projection helpers
# -------------------------------------------------
def project_points(K: np.ndarray, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points in camera space to 2D image coordinates.

    Args:
        K: 3x3 camera intrinsic matrix.
        pts_cam: (N, 3) array of 3D points in camera coordinates (x, y, z).

    Returns:
        A tuple of (pts_img, depth):
            - pts_img: (N, 2) array of 2D image coordinates (u, v).
            - depth: (N,) array of z/depth values.
    """
    z = pts_cam[:,2:3]
    pts_norm = pts_cam[:,:2]/z
    pts_img = (K[:2,:2] @ pts_norm.T).T + K[:2,2]
    return pts_img, z.squeeze()


def project_covariance(K: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Project a 3D covariance matrix to 2D using the Jacobian of the projection.

    Args:
        K: 3x3 camera intrinsic matrix.
        mu: 3D point (x, y, z) in camera space.
        Sigma: 3x3 covariance matrix in 3D.

    Returns:
        2x2 covariance matrix in image space.
    """
    x,y,z = mu
    fx, fy = K[0,0], K[1,1]
    J = np.array([[fx/z, 0, -fx*x/(z*z)],
                  [0, fy/z, -fy*y/(z*z)]])
    return J @ Sigma @ J.T


# -------------------------------------------------
# Minimal CPU renderer
# -------------------------------------------------
def render_gaussians(
    means: np.ndarray,
    covs: np.ndarray,
    colors: np.ndarray,
    opacity: np.ndarray,
    K: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    """Render 3D Gaussians to a 2D image using splatting.

    Projects each Gaussian to screen space, computes 2D covariance, and
    splats with alpha blending in back-to-front order.

    Args:
        means: (N, 3) array of 3D positions.
        covs: (N, 3, 3) array of 3D covariance matrices.
        colors: (N, 3) array of RGB colors (0–1).
        opacity: (N,) array of opacity values.
        K: 3x3 camera intrinsic matrix.
        H: Image height in pixels.
        W: Image width in pixels.

    Returns:
        (H, W, 3) RGB image array, values clipped to [0, 1].
    """
    pts_img, depth = project_points(K, means)
    order = np.argsort(depth)[::-1]

    img = np.zeros((H,W,3))
    T = np.ones((H,W))

    for idx in order:
        mu = means[idx]
        if mu[2] <= 0:
            continue

        center = pts_img[idx]
        Sigma2D = project_covariance(K, mu, covs[idx])

        try:
            invS = np.linalg.inv(Sigma2D)
        except:
            continue

        eigvals = np.linalg.eigvals(Sigma2D)
        radius = 3*np.sqrt(np.max(np.real(eigvals)))

        xmin = int(max(0, center[0]-radius))
        xmax = int(min(W-1, center[0]+radius))
        ymin = int(max(0, center[1]-radius))
        ymax = int(min(H-1, center[1]+radius))

        for y in range(ymin, ymax):
            for x in range(xmin, xmax):
                d = np.array([x-center[0], y-center[1]])
                w = np.exp(-0.5 * d @ invS @ d)

                a = opacity[idx]*w
                img[y,x] += T[y,x]*a*colors[idx]
                T[y,x] *= (1-a)

    return np.clip(img,0,1)


# -------------------------------------------------
# Plotly image display
# -------------------------------------------------
def show_plotly(img: np.ndarray) -> None:
    """Display an image in a Plotly figure (600x600).

    Args:
        img: (H, W, 3) image array with values in [0, 1].
    """
    fig = go.Figure(go.Image(z=(img*255).astype(np.uint8)))
    fig.update_layout(width=600, height=600)
    fig.show()


# -------------------------------------------------
# Example usage
# -------------------------------------------------
# means, scales, rots, opacity, colors = load_gaussians_from_ply("gaussians.ply")
# covs = build_covariances(scales, rots)

# H, W = 512, 512
# fx = fy = 500
# K = np.array([[fx,0,W/2],
#               [0,fy,H/2],
#               [0,0,1]])

# img = render_gaussians(means, covs, colors, opacity, K, H, W)
# show_plotly(img)


@click.command()
@click.option(
    "--file-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the PLY file to render.",
)
def main(file_path: str | Path) -> None:
    """Load Gaussian splats from a PLY file and render them.

    Args:
        file_path: Path to the PLY file containing Gaussian splat data.
    """
    means, scales, rots, opacity, colors = load_gaussians_from_ply(str(file_path))


if __name__ == "__main__":
    main()
