from lidra.data.dataset.tdfy.kubric.dataset import (
    KubricDataset,
    get_cam_K,
    KubricDatasetSampleID,
)
import roma
import torch
from typing import List, Dict, Optional, Union
import trimesh
from collections import namedtuple

from lidra.data.dataset.tdfy.mesh import convert_trimesh_to_pt3d
from pytorch3d.ops import sample_points_from_meshes


from loguru import logger


KubricMultiObjInFrameDatasetSampleID = namedtuple(
    "KubricMultiObjInFrameDatasetSampleID", ["video_id", "frame_id", "instance_ids"]
)
PosQuat = namedtuple("PosQuat", ["pos", "quat"])


class KubricMultiObjInFrameDataset(torch.utils.data.Dataset):
    """
    This dataset is used for training the model with multiple objects in the scene.
    """

    def __init__(
        self,
        dataset: KubricDataset,
        preload_gt_pts: bool = False,
        n_points_per_instance: int = 10000,
        latent_loader_dataset: Optional["TrellisPerSubsetDataset"] = None,
        keep_k_instances: Optional[int] = None,
        min_prop_visible_pixels: float = 0.02,
    ):
        super().__init__()
        self.dataset = dataset
        self.latent_loader_dataset = latent_loader_dataset
        self.keep_k_instances = keep_k_instances
        self.min_prop_visible_pixels = min_prop_visible_pixels

        logger.info(f"Min prop visible pixels: {self.min_prop_visible_pixels}")
        assert self.dataset.split in ["train", "validation"]
        is_train = dataset.split == "train"
        self.n_points_per_instance = n_points_per_instance
        self.preload_gt_pts = preload_gt_pts
        self.is_train = is_train
        self.name = "KubricMultiObjInFrameDataset"

    def __len__(self):
        return len(self.dataset)

    def _validate_dataset(self):
        if not isinstance(self.dataset, KubricDataset):
            raise ValueError("dataset must be an instance of KubricDataset")
        assert (
            self.dataset.sequence_length == 1
        ), f"KubricDataset in KubricMultiObjInFrameDataset must have sequence length of 1 (got {self.dataset.sequence_length})"

    def _prepare_instance_extrinsics(self, positions, quaternion, frame_idx):
        positions = positions[:, frame_idx]
        quaternion = quaternion[:, frame_idx]
        return PosQuat(positions, quaternion)

    def _get_instance_visibility(
        self, instance_info: Dict, metadata: Dict, frame_idx: int
    ) -> torch.Tensor:
        prop_visible_pixels = instance_info["visibility"][:, frame_idx] / (
            metadata["height"] * metadata["width"]
        )
        return prop_visible_pixels

    def _prepare_instance_points(
        self,
        asset_ids: List[str],
        meshes: Dict[str, trimesh.Trimesh],
        scale: torch.Tensor,  # (N, 1, 1)
        positions: torch.Tensor,
        quaternions: torch.Tensor,
        object_bounds: torch.Tensor,
        transform_to_world: bool = False,
    ) -> torch.Tensor:
        if len(asset_ids) == 0:
            raise ValueError("len(asset_ids) == 0")

        points_list = []
        for asset_id in asset_ids:
            mesh_pt3d = convert_trimesh_to_pt3d(meshes[asset_id])
            points = sample_points_from_meshes(mesh_pt3d, self.n_points_per_instance)
            points_list.append(points)
        points_tensor = torch.cat(points_list, dim=0)

        if transform_to_world:
            raise NotImplementedError(
                "Transform to world not implemented. Need to port over implementation of AnythingDataset"
            )

        return points_tensor

    def __getitem__(self, idx: Union[int, KubricMultiObjInFrameDatasetSampleID]):
        if isinstance(idx, tuple):
            video_id = idx.video_id
            frame_idx = idx.frame_id
            keep_instance_ids = idx.instance_ids
            raw_uuid, raw_sample = self.dataset[
                KubricDatasetSampleID(video_id, (frame_idx,))
            ]
        else:
            keep_instance_ids = None
            raw_uuid, raw_sample = self.dataset[idx]

        frame_idx = raw_sample["frame_indices"].squeeze(0).item()
        # # video_id = raw_sample["metadata"]["video_name"]
        # video_id = raw_sample["video_name"]
        # video_id = f"{self.dataset.dataset_name}/{video_id}"

        # Frames
        rgb_image = raw_sample["video"].squeeze(0)
        depth_image = raw_sample["depth"].squeeze(0)
        instance_image = raw_sample["instances"].squeeze(0)

        # Instance points in world coordinates
        asset_ids = raw_sample["instance_info"]["asset_id"]
        instance_points = None
        if len(raw_sample["meshes"]) > 0:
            instance_points = self._prepare_instance_points(
                asset_ids=asset_ids,
                meshes=raw_sample["meshes"],
                scale=raw_sample["instance_info"]["scale"],
                positions=raw_sample["instance_info"]["positions"][:, frame_idx],
                quaternions=raw_sample["instance_info"]["quaternions_l2w"][
                    :, frame_idx
                ],
                object_bounds=raw_sample["object_bounds"],
                transform_to_world=False,
            )

        instance_positions, instance_quaternions = self._prepare_instance_extrinsics(
            raw_sample["instance_info"]["positions"],
            raw_sample["instance_info"]["quaternions_l2w"],
            frame_idx,
        )

        # Cameras
        cameras = raw_sample["cameras"][frame_idx]

        # Create instance masks
        instance_masks = self._create_instance_masks(instance_image, len(asset_ids))

        sample = {
            "rgb_image": rgb_image,
            "depth_image": depth_image,
            "instance_image": instance_image,
            "instance_idx_kept": None,
            "instance_masks": instance_masks,
            "instance_points": instance_points,
            "cameras": cameras,
            "asset_ids": asset_ids,
            "frame_id": frame_idx,
            "frame_uuid": raw_uuid,
            # "video_id": video_id,
            # Additional info that we may want to require
            "instance_positions": instance_positions,
            "instance_quaternions_l2w": instance_quaternions,
            "instance_scales": raw_sample["instance_info"]["scale"],
            "instance_bounds": raw_sample["object_bounds"],
        }
        if self.keep_k_instances is not None:
            prop_visible_pixels = self._get_instance_visibility(
                raw_sample["instance_info"],
                raw_sample["metadata"],
                frame_idx,
            )
            try:
                sample = self._subsample_instances(
                    sample,
                    self.keep_k_instances,
                    prop_visible_pixels=prop_visible_pixels,
                    instance_ids=keep_instance_ids,
                )
            except ValueError as e:
                logger.warning(f"Caught exception in KubricMultiObjInFrameDataset {e}")
                # from lidra.data.collator import auto_uncollate
                if self.dataset.random_frame_selection:
                    new_idx = torch.randint(0, len(self.dataset), (1,))
                else:
                    new_idx = 1
                return self.__getitem__(new_idx)

        # Load latents
        instance_latents = self._load_instance_latents(sample["asset_ids"])
        sample.update(
            {
                "instance_latents": instance_latents,
            }
        )

        return (
            KubricMultiObjInFrameDatasetSampleID(
                video_id=raw_uuid.video_id,
                frame_id=raw_uuid.frame_ids[0],
                instance_ids=sample["instance_idx_kept"],
            ),
            sample,
        )

    def _subsample_instances(
        self,
        sample: Dict,
        n_instances: int,
        prop_visible_pixels: torch.Tensor,
        instance_ids: Optional[List[int]] = None,
    ) -> Dict:
        """
        Keep only n_instances instances in the sample.
        """
        new_sample = {k: v for k, v in sample.items()}
        n_orig_instances = new_sample["instance_masks"].shape[0]
        if n_orig_instances < n_instances:
            raise ValueError(
                f"n_instances ({n_instances}) is greater than the number of instances in the sample ({n_orig_instances})"
            )

        # Only keep instances with prop_visible_pixels > min_prop_visible_pixels
        idxs = torch.arange(n_orig_instances, device=prop_visible_pixels.device)

        if instance_ids is not None:
            idxs = torch.tensor(instance_ids, device=prop_visible_pixels.device)
        else:
            if self.min_prop_visible_pixels > 0:
                idxs = idxs[prop_visible_pixels > self.min_prop_visible_pixels]
                if len(idxs) < n_instances:
                    raise ValueError(
                        f"Not enough visible instances ({len(idxs)}) < {n_instances} instances"
                    )

            if self.dataset.random_frame_selection:
                # random sample n_instances from idxs
                n_viable_instances = len(idxs)
                idxs = idxs[torch.randperm(n_viable_instances)][:n_instances]
            else:
                idxs = idxs[:n_instances]

        idxs_list = idxs.tolist()
        new_sample["asset_ids"] = [new_sample["asset_ids"][i] for i in idxs_list]
        new_sample["instance_masks"] = new_sample["instance_masks"][idxs]

        if new_sample.get("shape_latents", None) is not None:
            for k in new_sample["shape_latents"].keys():
                new_sample["shape_latents"][k] = new_sample["shape_latents"][k][idxs]

        if new_sample.get("instance_points", None) is not None:
            new_sample["instance_points"] = new_sample["instance_points"][idxs]

        new_sample["instance_positions"] = new_sample["instance_positions"][idxs]
        new_sample["instance_quaternions_l2w"] = new_sample["instance_quaternions_l2w"][
            idxs
        ]
        new_sample["instance_scales"] = new_sample["instance_scales"][idxs]
        if new_sample.get("instance_bounds", None) is not None:
            new_sample["instance_bounds"] = new_sample["instance_bounds"][idxs]
        new_sample["instance_idx_kept"] = idxs
        return new_sample

    def _create_instance_masks(
        self, instance_image: torch.Tensor, n_instances: int
    ) -> torch.Tensor:
        h, w = instance_image.shape
        one_hot = torch.zeros(n_instances + 1, h, w)
        one_hot.scatter_(0, instance_image.unsqueeze(0), 1)
        return one_hot[1:]  # Dont include background

    def _load_instance_latents(
        self,
        asset_ids: List[str],
    ):
        if self.latent_loader_dataset is None:
            return None

        shape_latents = []
        for asset_id in asset_ids:
            sha256 = self.latent_loader_dataset.get_sha256(
                self.latent_loader_dataset.metadata, f"raw/{asset_id}"
            )
            latent = self.latent_loader_dataset._load_latent(sha256)
            poses = self.latent_loader_dataset._load_available_poses(sha256)
            assert (
                len(poses) > 0
            ), f"Empty scale and offset information for latents: {asset_id}"
            latent["scale"] = torch.tensor(poses["scale"])
            latent["offset"] = torch.tensor(poses["offset"])
            latent["aabb"] = torch.tensor(poses["aabb"])
            shape_latents.append(latent)

        # Stack based on keys [mean, logvar, coords]
        keys = list(shape_latents[0].keys())
        shape_latents = {
            k: torch.stack([latent[k] for latent in shape_latents]) for k in keys
        }
        return shape_latents
