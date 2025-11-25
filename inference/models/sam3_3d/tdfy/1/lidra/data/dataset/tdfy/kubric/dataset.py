from collections import namedtuple
import json
from loguru import logger
import numpy as np
import os
from PIL import Image
import roma
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import trimesh
from typing import Dict, List, Optional, Tuple, Union
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.transforms import Translate, Rotate, quaternion_invert
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
from loguru import logger


from .mesh_loading import load_gso_kubric_mesh as load_gso_mesh

KubricDatasetSampleID = namedtuple("KubricDatasetSampleID", ["video_id", "frame_ids"])


class KubricDataset(Dataset):
    """Dataset for loading processed Kubric data."""

    def __init__(
        self,
        data_dir: str,
        dataset_name: str = "movi_d",
        split: str = "train",
        frame_skip: int = 1,
        sequence_length: int = None,
        skip_first_n_frames: int = 0,
        transform=None,
        reload_cache: bool = False,
        mesh_dir: str = None,
        load_meshes: bool = False,
        random_frame_selection: bool = True,
    ):
        self.base_dir = data_dir
        self.data_dir = os.path.join(data_dir, dataset_name, split)
        self.dataset_name = dataset_name
        self.frame_skip = frame_skip
        self.reload_cache = reload_cache
        self.sequence_length = sequence_length
        self.skip_first_n_frames = skip_first_n_frames
        self.split = split
        self.transform = transform
        self.random_frame_selection = random_frame_selection

        self.load_meshes = load_meshes
        if mesh_dir is None:
            data_dir_parent = os.path.dirname(data_dir)
            mesh_dir = os.path.join(data_dir_parent, "assets", "GSO")
        self.mesh_dir = mesh_dir

        self._cache_n_frames_per_video()

        logger.info(f"kubric video directory : {self.data_dir}")

        # Verify the first video to get number of frames
        first_video = os.path.join(self.data_dir, self.video_dirs[0])
        rgb_dir = os.path.join(first_video, "rgb")

        self.total_sequences = len(self.video_dirs)

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: Union[int, KubricDatasetSampleID]) -> dict:

        # Get sample by idx or by looking up a KubricDatasetSampleID
        if isinstance(idx, tuple):
            video_uuid = idx.video_id
            frame_indices = idx.frame_ids
        else:  # Choose some frames from the video
            video_id = self.video_dirs[idx]
            video_uuid = f"{self.dataset_name}/{self.split}/{video_id}"
            frame_indices = list(
                range(
                    self.skip_first_n_frames,
                    self.n_frames_per_video[video_id],
                    self.frame_skip,
                )
            )
            # Choose frames
            n_frames = self.n_frames_per_video[video_id]
            if (
                self.sequence_length is not None
                and len(frame_indices) > self.sequence_length
            ):
                if self.random_frame_selection:
                    start_idx = np.random.randint(
                        0, len(frame_indices) - self.sequence_length + 1
                    )
                else:
                    start_idx = 0
                frame_indices = frame_indices[
                    start_idx : start_idx + self.sequence_length
                ]

        # Make sample uuid
        video_dir = os.path.join(self.base_dir, video_uuid)
        sample_uuid = KubricDatasetSampleID(video_uuid, frame_indices)

        # Load metadata
        with open(os.path.join(video_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        height, width = metadata["height"], metadata["width"]

        # Load instance info
        instance_info_path = os.path.join(video_dir, "instance_info.npz")
        instance_info = dict(np.load(instance_info_path, allow_pickle=True))
        instance_info = {
            "asset_id": [x.decode("utf-8") for x in instance_info["asset_id"]],
            "positions": torch.from_numpy(instance_info["positions"]),
            "quaternions_l2w": quaternion_invert(
                torch.from_numpy(instance_info["quaternions"])
            ),
            "scale": torch.from_numpy(instance_info["scale"]),
            "visibility": torch.from_numpy(
                instance_info["visibility"].astype(np.int32)
            ),
        }

        # Load frames
        rgbs, instances, depths = [], [], []
        # If we are loading many frames, we can use a py wrapper of libav
        # https://github.com/PyAV-Org/PyAV
        for frame_idx in frame_indices:
            rgb, inst, depth = self.load_frame(video_dir, frame_idx)
            rgbs.append(rgb)
            instances.append(inst)
            depths.append(depth)

        # Load camera parameters
        camera_info_path = os.path.join(video_dir, "camera.npz")
        camera_info = dict(np.load(camera_info_path, allow_pickle=True))
        for k, v in camera_info.items():
            camera_info[k] = torch.from_numpy(v)
        cameras: PerspectiveCameras = self._prepare_cameras(
            camera_info, (height, width)
        )

        # Convert distance to depth in world units
        depths = torch.stack(depths)
        depths = self._convert_to_depth(
            depths,
            metadata["depth_range"][0],
            metadata["depth_range"][1],
            camera_info["field_of_view"],
        )

        # Load meshes
        meshes = {}
        mesh_data = {}
        object_bounds = None
        for asset_id in instance_info["asset_id"]:
            asset_id = (
                asset_id.decode("utf-8") if isinstance(asset_id, bytes) else asset_id
            )
            if self.load_meshes:
                meshes[asset_id], mesh_data[asset_id] = self.load_mesh(asset_id)
            else:
                mesh_data[asset_id] = self.load_mesh_data(asset_id)

        object_bounds = torch.tensor(
            [
                mesh_data[asset_id]["kwargs"]["bounds"]
                for asset_id in instance_info["asset_id"]
            ]
        )

        return sample_uuid, {
            "metadata": metadata,
            "video": torch.stack(rgbs),
            "instances": torch.stack(instances),
            "depth": depths,
            "meshes": meshes,
            "object_bounds": object_bounds,
            "instance_info": instance_info,
            "cameras": cameras,
            # "video_idx": idx,
            "video_uuid": video_uuid,
            "frame_indices": torch.tensor(frame_indices),
            # 'camera': camera_info,
            # 'mesh_data': mesh_data,
        }

    def load_frame(
        self, video_dir: str, frame_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load RGB, instance segmentation, and depth data for a single frame."""
        # Load RGB frame
        rgb_path = os.path.join(video_dir, "rgb", f"frame_{frame_idx:04d}.png")
        rgb = Image.open(rgb_path).convert("RGB")
        if self.transform:
            rgb = self.transform(rgb)
        else:
            rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0

        # Load instance segmentation
        inst_path = os.path.join(video_dir, "instances", f"frame_{frame_idx:04d}.png")
        inst = torch.from_numpy(np.array(Image.open(inst_path))).long()

        # Load depth map
        depth_path = os.path.join(video_dir, "depth", f"frame_{frame_idx:04d}.png")
        depth = torch.from_numpy(np.float32(np.array(Image.open(depth_path))))

        return rgb, inst, depth

    def load_mesh(self, asset_id: str) -> trimesh.Trimesh:
        assert self.mesh_dir is not None, "Mesh directory is not set"
        return load_gso_mesh(self.mesh_dir, asset_id)

    def load_mesh_data(self, asset_id: str) -> Dict:
        with open(os.path.join(self.mesh_dir, asset_id, "data.json"), "r") as f:
            data_json = json.load(f)
        return data_json

    def _prepare_cameras(
        self, camera_info: Dict[str, torch.Tensor], video_shape: Tuple[int, int]
    ) -> PerspectiveCameras:
        # Intrinsics
        K = get_cam_K(
            camera_info["field_of_view"],
            video_shape[-1],
            video_shape[-2],
            screen_space=True,
        )

        # Extrinsics
        R_c2w = quaternion_to_matrix(camera_info["quaternions"])
        R_c2w = Rotate(R=R_c2w.permute(0, 2, 1))  # Row-major
        T_c2w = Translate(camera_info["positions"])
        cam_to_world = R_c2w.compose(T_c2w)
        world_to_cam = cam_to_world.inverse()

        convention_transform = blender_to_pt3d_cam_convention_transform()
        world_to_cam = world_to_cam.compose(convention_transform)
        R = world_to_cam.get_matrix()[:, :3, :3]
        T = world_to_cam.get_matrix()[:, 3, :3]

        # PerspectiveCameras
        cam = PerspectiveCameras(
            focal_length=((K[0, 0], K[1, 1]),),
            principal_point=((K[0, 2], K[1, 2]),),
            R=R,
            T=T,
            in_ndc=False,
            image_size=((video_shape[-2], video_shape[-1]),),
        )
        return cam

    def _convert_to_depth(
        self, depth: torch.Tensor, depth_min: float, depth_max: float, fov_rad: float
    ) -> torch.Tensor:
        assert depth_max > depth_min, "Depth max must be greater than depth min"
        assert depth.min() >= 0, "Depth must be non-negative"
        depth = depth / 65535 * (depth_max - depth_min) + depth_min
        h, w = depth.shape[-2:]
        depth = distance_to_pts3d(depth, get_cam_K(fov_rad, w, h))[..., 2]
        return depth

    def _cache_n_frames_per_video(self) -> Dict[str, int]:
        if self.reload_cache or not os.path.exists(
            os.path.join(self.data_dir, "n_frames_per_video.json")
        ):
            self.n_frames_per_video = {}

            # Get list of all video directories
            self.video_dirs = sorted(
                [
                    d
                    for d in os.listdir(self.data_dir)
                    if os.path.isdir(os.path.join(self.data_dir, d))
                ]
            )

            for video_dir in tqdm(self.video_dirs, desc="Caching frames per video"):
                self.n_frames_per_video[video_dir] = len(
                    [
                        f
                        for f in os.listdir(
                            os.path.join(self.data_dir, video_dir, "rgb")
                        )
                        if f.endswith(".png")
                    ]
                )
            with open(os.path.join(self.data_dir, "n_frames_per_video.json"), "w") as f:
                json.dump(self.n_frames_per_video, f)

        with open(os.path.join(self.data_dir, "n_frames_per_video.json"), "r") as f:
            self.n_frames_per_video = json.load(f)
        self.video_dirs = sorted(self.n_frames_per_video.keys())


def get_cam_K(
    fov: float, width: int, height: int, screen_space: bool = False
) -> torch.Tensor:
    # Compute focal length directly from field of view
    if not screen_space:
        width, height = 1.0, 1.0  # self.active_scene.resolution

    f_x = width / (2 * np.tan(fov / 2))
    f_y = height / (2 * np.tan(fov / 2))
    p_x = width / 2.0
    p_y = height / 2.0

    return torch.tensor(
        [
            [f_x, 0, p_x],
            [0, f_y, p_y],
            [0, 0, 1],
        ]
    )


def distance_to_pts3d(distance_image: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Kubric depth image is distance of each pixel from the center of the camera.
    (Note this is different from the z-value sometimes used, which measures the
    distance to the camera *plane*.)
    The values are stored as uint16 and span the range specified in
    sample["metadata"]["depth_range"]. To convert them back to world-units
    use:
        minv, maxv = sample["metadata"]["depth_range"]
        depth = sample["depth"] / 65535 * (maxv - minv) + minv
    https://github.com/google-research/kubric/blob/0ee21e2a723b2131123d67e55d1f65b6d0e6cf0f/challenges/movi/movi_d.py#L60
    """
    # Get camera rays
    height, width = distance_image.shape[-2:]
    i, j = torch.meshgrid(
        torch.arange(width, dtype=torch.float32),
        torch.arange(height, dtype=torch.float32),
        indexing="xy",
    )

    # Convert pixel coordinates to normalized device coordinates
    x_ndc = (i + 0.5) / width * 2 - 1
    y_ndc = (j + 0.5) / height * 2 - 1

    # Get ray directions in camera space
    x_cam = x_ndc * K[0, 2] / K[0, 0]
    y_cam = y_ndc * K[1, 2] / K[1, 1]
    z_cam = -torch.ones_like(x_cam)

    # Stack to get ray directions
    rays_d = torch.stack([x_cam, y_cam, z_cam], dim=-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    return -rays_d * distance_image[..., None]
    # Ray origins are at camera center
    rays_o = torch.zeros_like(rays_d)

    points_3d = rays_o + rays_d * distance_image[..., None]
    return points_3d


def blender_to_pt3d_cam_convention_transform():
    R = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # T = torch.tensor([0, 0, 0])
    return Rotate(R=R)
