import numpy as np
from plyfile import PlyData
import plotly.graph_objects as go
import click
from pathlib import Path


# -------------------------------------------------
# Load gaussian splats from PLY
# -------------------------------------------------
def load_gaussians_from_ply(path):
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
def quat_to_rot(q):
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


# -------------------------------------------------
# Build covariance from scale + rotation
# -------------------------------------------------
def build_covariances(scales, rots):
    covs = []
    for s, q in zip(scales, rots):
        R = quat_to_rot(q)
        S = np.diag(s**2)
        covs.append(R @ S @ R.T)
    return np.array(covs)


# -------------------------------------------------
# Projection helpers
# -------------------------------------------------
def project_points(K, pts_cam):
    z = pts_cam[:,2:3]
    pts_norm = pts_cam[:,:2]/z
    pts_img = (K[:2,:2] @ pts_norm.T).T + K[:2,2]
    return pts_img, z.squeeze()


def project_covariance(K, mu, Sigma):
    x,y,z = mu
    fx, fy = K[0,0], K[1,1]
    J = np.array([[fx/z, 0, -fx*x/(z*z)],
                  [0, fy/z, -fy*y/(z*z)]])
    return J @ Sigma @ J.T


# -------------------------------------------------
# Minimal CPU renderer
# -------------------------------------------------
def render_gaussians(means, covs, colors, opacity, K, H, W):
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
def show_plotly(img):
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
def main(file_path: Path) -> None:
    means, scales, rots, opacity, colors = load_gaussians_from_ply(str(file_path))
    pass


if __name__ == "__main__":
    main()
