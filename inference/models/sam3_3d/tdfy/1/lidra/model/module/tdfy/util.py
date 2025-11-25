import torch
import numpy as np
import random
from pytorch3d.transforms import (
    RotateAxisAngle,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from pytorch3d.renderer.cameras import PerspectiveCameras
import warnings


def binary_labels_from_point_distances(predicted_xyz, gt_xyz, dist_thres):
    CHUNK_SIZE = 50_000
    is_not_multiple = (predicted_xyz.shape[0] % CHUNK_SIZE) > 0
    predicted_xyz = torch.chunk(
        predicted_xyz,
        predicted_xyz.shape[0] // CHUNK_SIZE + int(is_not_multiple),
        dim=0,
    )

    min_dist = []
    for pred_xyz in predicted_xyz:
        dist = torch.cdist(pred_xyz, gt_xyz)
        min_dist.append(dist.min(dim=-1).values)

    min_dist = torch.cat(min_dist, dim=0)

    return min_dist < dist_thres


def torch_or_numpy(object):
    if isinstance(object, torch.Tensor):
        return torch
    elif isinstance(object, np.ndarray):
        return np
    raise RuntimeError(f"cannot handle object of type : {type(object)}")


def sample_uniform_box(
    B: int, N: int, size: float, device: torch.device
) -> torch.Tensor:
    """Sample points uniformly from a 3D box centered at origin.

    Args:
        B: Batch size
        N: Number of points per batch
        size: Half-length of box side
        device: Device to create tensor on

    Returns:
        Points tensor of shape (B*N, 3)
    """
    points = torch.empty((B, N, 3), device=device).uniform_(-1.0, 1.0)
    points *= size
    return points


def grid_uniform_box(
    B: int, granularity: float, size: float, device: torch.device
) -> torch.Tensor:
    """Create uniform grid of points in a 3D box centered at origin.

    Args:
        B: Batch size
        granularity: Grid spacing
        size: Half-length of box side
        device: Device to create tensor on

    Returns:
        Grid points tensor of shape (B, N, 3) where N = ((2*size/granularity + 1)**3)
    """
    n_grid_pts = int(size / granularity) * 2 + 1
    grid = torch.linspace(-size, size, n_grid_pts, device=device)
    x, y, z = torch.meshgrid(grid, grid, grid, indexing="ij")
    grid_xyz = torch.stack([x, y, z], dim=-1)
    grid_xyz = grid_xyz.reshape(-1, 3)
    return grid_xyz[None].repeat(B, 1, 1)


def construct_uniform_cube(
    gt_xyz,
    gt_rgb,
    world_size,
    n_queries,
    dist_threshold,
    is_train,
    granularity,
):
    B = gt_xyz.shape[0]
    device = gt_xyz.device
    if is_train:
        unseen_xyz = sample_uniform_box(B, n_queries, world_size, device)
    else:
        unseen_xyz = grid_uniform_box(B, granularity, world_size, device)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels] = torch.gather(
        gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3)
    )[labels]
    return unseen_xyz, unseen_rgb, labels.float()


def construct_uniform_semisphere(
    gt_xyz,
    gt_rgb,
    semisphere_size,
    n_queries,
    dist_threshold,
    is_train,
    granularity,
):
    B = gt_xyz.shape[0]
    device = gt_xyz.device
    if is_train:
        unseen_xyz = sample_uniform_semisphere(B, n_queries, semisphere_size, device)
    else:
        unseen_xyz = get_grid_semisphere(B, granularity, semisphere_size, device)
    dist, idx_to_gt = get_min_dist(unseen_xyz[:, :, None], gt_xyz[:, None])
    labels = dist < dist_threshold
    unseen_rgb = torch.zeros_like(unseen_xyz)
    unseen_rgb[labels] = torch.gather(
        gt_rgb, 1, idx_to_gt.unsqueeze(-1).repeat(1, 1, 3)
    )[labels]
    return unseen_xyz, unseen_rgb, labels.float()


def sample_uniform_semisphere(B, N, semisphere_size, device):
    for _ in range(100):
        points = torch.empty(B * N * 3, 3, device=device).uniform_(
            -semisphere_size, semisphere_size
        )
        points[..., 2] = points[..., 2].abs()
        dist = (points**2.0).sum(axis=-1) ** 0.5
        if (dist < semisphere_size).sum() >= B * N:
            return points[dist < semisphere_size][: B * N].reshape((B, N, 3))
        else:
            print("resampling sphere")


def get_grid_semisphere(B, granularity, semisphere_size, device):
    n_grid_pts = int(semisphere_size / granularity) * 2 + 1
    grid_unseen_xyz = torch.zeros(
        (n_grid_pts, n_grid_pts, n_grid_pts // 2 + 1, 3), device=device
    )
    for i in range(n_grid_pts):
        grid_unseen_xyz[i, :, :, 0] = i
        grid_unseen_xyz[:, i, :, 1] = i
    for i in range(n_grid_pts // 2 + 1):
        grid_unseen_xyz[:, :, i, 2] = i
    grid_unseen_xyz[..., :2] -= n_grid_pts // 2.0
    grid_unseen_xyz *= granularity
    dist = (grid_unseen_xyz**2.0).sum(axis=-1) ** 0.5
    grid_unseen_xyz = grid_unseen_xyz[dist <= semisphere_size]
    return grid_unseen_xyz[None].repeat(B, 1, 1)


def get_min_dist(a, b, slice_size=1000):
    all_min, all_idx = [], []
    for i in range(int(np.ceil(a.shape[1] / slice_size))):
        start = slice_size * i
        end = slice_size * (i + 1)
        # B, n_queries, n_gt
        dist = ((a[:, start:end] - b) ** 2.0).sum(axis=-1) ** 0.5
        # B, n_queries
        cur_min, cur_idx = dist.min(axis=2)
        all_min.append(cur_min)
        all_idx.append(cur_idx)
    return torch.cat(all_min, dim=1), torch.cat(all_idx, dim=1)


def aug_xyz(
    seen_xyz,
    unseen_xyz,
    random_scale_delta,
    origin_at_cam,
    random_rotate_degree,
    random_shift,
    is_train,
):
    degree_x = 0
    degree_y = 0
    degree_z = 0
    if is_train:
        r_delta = random_scale_delta
        scale = torch.tensor(
            [
                random.uniform(1.0 - r_delta, 1.0 + r_delta),
                random.uniform(1.0 - r_delta, 1.0 + r_delta),
                random.uniform(1.0 - r_delta, 1.0 + r_delta),
            ],
            device=seen_xyz.device,
        )

        # TODO: make this not dependent on the use_hypersim flag (scene-based instead)
        if origin_at_cam:
            shift = 0
        else:
            degree_x = random.randrange(-random_rotate_degree, random_rotate_degree + 1)
            degree_y = random.randrange(-random_rotate_degree, random_rotate_degree + 1)
            degree_z = random.randrange(-random_rotate_degree, random_rotate_degree + 1)

            r_shift = random_shift
            shift = torch.tensor(
                [
                    [
                        [
                            random.uniform(-r_shift, r_shift),
                            random.uniform(-r_shift, r_shift),
                            random.uniform(-r_shift, r_shift),
                        ]
                    ]
                ],
                device=seen_xyz.device,
            )
        seen_xyz = seen_xyz * scale + shift
        unseen_xyz = unseen_xyz * scale + shift

    B, H, W, _ = seen_xyz.shape
    return [
        rotate(seen_xyz.reshape((B, -1, 3)), degree_x, degree_y, degree_z).reshape(
            (B, H, W, 3)
        ),
        rotate(unseen_xyz, degree_x, degree_y, degree_z),
    ]


def rotate(sample, degree_x, degree_y, degree_z):
    for degree, axis in [(degree_x, "X"), (degree_y, "Y"), (degree_z, "Z")]:
        if degree != 0:
            sample = (
                RotateAxisAngle(degree, axis=axis)
                .to(sample.device)
                .transform_points(sample)
            )
    return sample


def shrink_points_beyond_threshold(xyz, threshold):
    xyz = xyz.clone().detach()
    dist = (xyz**2.0).sum(axis=-1) ** 0.5
    affected = (dist > threshold) * torch.isfinite(dist)
    xyz[affected] = (
        xyz[affected]
        * (threshold * (2.0 - threshold / dist[affected]) / dist[affected])[..., None]
    )
    return xyz


# adapted from PoseDiffusion https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/util/camera_transform.py#L108
def camera_encoding(
    camera,
    pose_encoding_type="absT_quaR",
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=20,
):
    """ """

    if pose_encoding_type == "absT_quaR_logFL":
        # Convert rotation matrix to quaternion
        quaternion_R = matrix_to_quaternion(camera.R)

        # Calculate log_focal_length
        log_focal_length = (
            torch.log(
                torch.clamp(
                    camera.focal_length, min=min_focal_length, max=max_focal_length
                )
            )
            - log_focal_length_bias
        )

        # Concatenate to form pose_encoding
        pose_encoding = torch.cat([camera.T, quaternion_R, log_focal_length], dim=-1)
    elif pose_encoding_type == "absT_quaR":
        # Convert rotation matrix to quaternion
        quaternion_R = matrix_to_quaternion(camera.R)
        pose_encoding = torch.cat([camera.T, quaternion_R], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding


def pose_encoding_to_camera(
    pose_encoding,
    pose_encoding_type="absT_quaR",
    log_focal_length_bias=1.8,
    min_focal_length=0.1,
    max_focal_length=20,
    return_dict=False,
):
    """
    Args:
        pose_encoding: A tensor of shape `BxNxC`, containing a batch of
                        `BxN` `C`-dimensional pose encodings.
        pose_encoding_type: The type of pose encoding,
                        only "absT_quaR_logFL" is supported.
    """

    pose_encoding_reshaped = pose_encoding.reshape(
        -1, pose_encoding.shape[-1]
    )  # Reshape to BNxC

    if pose_encoding_type == "absT_quaR_logFL":
        # forced that 3 for absT, 4 for quaR, 2 logFL
        # TODO: converted to 1 dim for logFL, consistent with our paper
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)

        log_focal_length = pose_encoding_reshaped[:, 7:9]

        # log_focal_length_bias was the hyperparameter
        # to ensure the mean of logFL close to 0 during training
        # Now converted back
        focal_length = (log_focal_length + log_focal_length_bias).exp()

        # clamp to avoid weird fl values
        focal_length = torch.clamp(
            focal_length, min=min_focal_length, max=max_focal_length
        )
        pred_cameras = PerspectiveCameras(
            focal_length=focal_length, R=R, T=abs_T, device=R.device
        )
    elif pose_encoding_type == "absT_quaR":
        abs_T = pose_encoding_reshaped[:, :3]
        quaternion_R = pose_encoding_reshaped[:, 3:7]
        R = quaternion_to_matrix(quaternion_R)
        focal_length = None
        pred_cameras = PerspectiveCameras(R=R, T=abs_T, device=R.device)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    if return_dict:
        return {"focal_length": focal_length, "R": R, "T": abs_T}

    return pred_cameras
