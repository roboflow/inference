from collections import namedtuple
from pytorch3d.renderer import PerspectiveCameras
from loguru import logger
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_to_matrix,
    quaternion_multiply,
    quaternion_apply,
    Rotate,
    Translate,
    Scale,
    Transform3d,
)

from typing import List, Dict, Optional, Union
from torch.utils.data import Dataset
import torch

from lidra.data.dataset.tdfy.kubric.multi_obj_frame_dataset import (
    KubricMultiObjInFrameDataset,
    KubricMultiObjInFrameDatasetSampleID,
)
from lidra.data.dataset.tdfy.point_cloud import get_rgbd_points
from lidra.data.dataset.tdfy.transforms_3d import (
    compose_transform,
    decompose_transform,
    # normalize_points_to_unit_cube,
)
from lidra.data.dataset.tdfy.trellis.pose_loader import (
    convert_to_decoupled_instance_pose,
)
from .utils import normalize_points_to_unit_cube

KubricAnythingSampleID = namedtuple(
    "KubricAnythingSampleID", ["video_id", "frame_id", "instance_id"]
)


class AnythingDataset(Dataset):
    COORDINATE_FRAME_CHOICES = ("world", "camera")

    def __init__(
        self,
        dataset: KubricMultiObjInFrameDataset,
        object_pose_coordinate_frame: str = "camera",
        return_camera_args: bool = True,
        return_pointmap: bool = False,
        return_mesh: bool = False,
    ):
        super().__init__()
        assert dataset.keep_k_instances == 1
        assert dataset.dataset.sequence_length == 1
        self.dataset = dataset

        assert dataset.latent_loader_dataset is not None
        self.latent_loader_dataset = dataset.latent_loader_dataset

        object_pose_coordinate_frame = object_pose_coordinate_frame.lower()
        assert object_pose_coordinate_frame in self.COORDINATE_FRAME_CHOICES
        self.object_pose_coordinate_frame = object_pose_coordinate_frame

        self.return_camera_args = return_camera_args

        self.return_pointmap = return_pointmap
        self.return_mesh = return_mesh

    def __len__(self):
        return len(self.dataset)

    def _get_sample_uuid(self, idx: Union[int, KubricAnythingSampleID]):
        if isinstance(idx, tuple):
            sample_uuid = KubricAnythingSampleID(*idx)
            raw_uuid, raw_sample = self.dataset[
                KubricMultiObjInFrameDatasetSampleID(
                    video_id=sample_uuid.video_id,
                    frame_id=sample_uuid.frame_id,
                    instance_ids=(sample_uuid.instance_id,),
                )
            ]
        else:
            raw_uuid, raw_sample = self.dataset[idx]
            sample_uuid = KubricAnythingSampleID(
                video_id=raw_uuid.video_id,
                frame_id=raw_uuid.frame_id,
                instance_id=raw_uuid.instance_ids[0].item(),
            )

        return sample_uuid, raw_sample

    def _get_camera_args(self, cameras: PerspectiveCameras) -> Dict[str, torch.Tensor]:
        if not self.return_camera_args:
            return {}
        return {
            "camera_quaternion_c2w": matrix_to_quaternion(cameras.R),
            "camera_position_c2w": cameras.T,
            "camera_focal_length": cameras.focal_length,
            "camera_principal_point": cameras.principal_point,
            "camera_image_size": cameras.image_size,
        }

    def _get_mesh_vertices(
        self, instance_points, instance_bounds, normalize: bool = True
    ):
        if instance_points is None or not self.return_mesh:
            return {}
        # Pointclouds for objects (for vis)
        instance_points_local = instance_points
        if normalize:
            instance_points_local = normalize_points_to_unit_cube(
                instance_points_local,
                instance_bounds,
            )
        return {
            "mesh_vertices": instance_points_local.squeeze(0),
        }

    def _get_pointmap(self, rgb_image, depth_image, cameras):
        rgb_image_shape = rgb_image.shape
        depth_im_shape = depth_image.shape
        pointmap, pointmap_mask = get_rgbd_points(
            imh=rgb_image_shape[-2],
            imw=rgb_image_shape[-1],
            depth_map=depth_image.unsqueeze(0).unsqueeze(0),
            camera=cameras,
        )

        # This is so hacky, but it helps keep the code in one place.
        return_dict = self.latent_loader_dataset._prepare_pointmap(
            pointmap, self.return_pointmap
        )
        return return_dict

    def _process_image_mask_mess(self, rgb_image, mask_image):
        # Load images
        processed_image, processed_mask = (
            self.latent_loader_dataset._preprocess_image_and_mask_inference(
                rgb_image, mask_image
            )
        )

        processed_image = self.latent_loader_dataset.img_transform(processed_image)
        processed_mask = self.latent_loader_dataset.mask_transform(processed_mask)
        rgb_image = self.latent_loader_dataset.img_transform(rgb_image)
        mask_image = self.latent_loader_dataset.mask_transform(mask_image)
        return {
            "image": processed_image,
            "mask": processed_mask,
            "rgb_image": rgb_image,
            "rgb_image_mask": mask_image,
        }

    def _transform_mesh_to_latent_frame(
        self,
        mesh_to_world,
        mesh_dict,
        latent_scale,
        latent_offset,
        latent_aabb,
        instance_bounds,
    ):
        # Shape latents are rotated by 90 degrees around x-axis, because of how they are preprocessed.
        #  So here we rotate the pose and shape by 90 degrees
        mesh_to_latent_quat = self.latent_loader_dataset.mesh_to_latent_quat
        mesh_to_latent = Rotate(R=quaternion_to_matrix(mesh_to_latent_quat))

        # We also need to adjust for offset (stored in trellis?)
        orig_mesh_scale = instance_bounds[:, 1] - instance_bounds[:, 0]
        orig_mesh_scale = orig_mesh_scale.max(dim=-1).values
        scale = Scale(*orig_mesh_scale * latent_scale)
        translate = Translate(latent_offset)
        mesh_to_latent = mesh_to_latent.compose(scale).compose(translate)

        # Now adjust the mesh and update the mesh_to_world accordingly
        latent_to_world = mesh_to_latent.inverse().compose(mesh_to_world)

        if "mesh_vertices" in mesh_dict:
            mesh_dict = {k: v for k, v in mesh_dict.items()}
            mesh_dict["mesh_vertices"] = mesh_to_latent.transform_points(
                mesh_dict["mesh_vertices"]
            )

        return latent_to_world, mesh_dict

    def _transform_kubric_to_trellis(
        self,
        scale,
        rotation,
        translation,
        cameras,
        mesh_dict,
        latent_scale,
        latent_offset,
        latent_aabb,
        instance_bounds,
    ):

        obj_to_world = compose_transform(
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        # Convert to cam frame
        if self.object_pose_coordinate_frame == "camera":
            obj_to_world, cameras = self._transform_world_frame_to_cam_frame(
                obj_to_world, cameras
            )

        obj_to_world, mesh_dict = self._transform_mesh_to_latent_frame(
            mesh_to_world=obj_to_world,
            mesh_dict=mesh_dict,
            latent_scale=latent_scale,
            latent_offset=latent_offset,
            latent_aabb=latent_aabb,
            instance_bounds=instance_bounds,
        )

        instance_pose_dict = convert_to_decoupled_instance_pose(obj_to_world)
        return instance_pose_dict, cameras, mesh_dict

    @staticmethod
    def _transform_world_frame_to_cam_frame(
        obj_to_world: Transform3d,
        original_cam: PerspectiveCameras,
    ):
        # Convert to cam frame
        cameras = original_cam.clone()
        world_to_cam_transform = original_cam.get_world_to_view_transform()
        obj_to_cam = obj_to_world.compose(world_to_cam_transform)
        obj_to_world = obj_to_cam
        cameras.R = torch.eye(3)[None]
        cameras.T = torch.zeros(3)[None]
        return obj_to_world, cameras

    def __getitem__(self, idx: Union[int, KubricAnythingSampleID]):
        sample_uuid, raw_sample = self._get_sample_uuid(idx)

        # Load latents
        instance_latents = {
            "mean": raw_sample["instance_latents"]["mean"].squeeze(0),
            "logvar": raw_sample["instance_latents"]["logvar"].squeeze(0),
        }

        # Get instance points: Scale is normalized to be in [-0.5, 0.5]
        mesh_dict = self._get_mesh_vertices(
            instance_points=raw_sample.get("instance_points", None),
            instance_bounds=raw_sample.get("instance_bounds", None),
            normalize=True,
        )

        # Transform instance and camera geometry to trellis frame
        instance_pose_dict, cameras, mesh_dict = self._transform_kubric_to_trellis(
            scale=raw_sample["instance_scales"],
            rotation=quaternion_to_matrix(raw_sample["instance_quaternions_l2w"]),
            translation=raw_sample["instance_positions"],
            cameras=raw_sample["cameras"],
            mesh_dict=mesh_dict,
            latent_scale=raw_sample["instance_latents"]["scale"],
            latent_offset=raw_sample["instance_latents"]["offset"],
            latent_aabb=raw_sample["instance_latents"]["aabb"],
            instance_bounds=raw_sample["instance_bounds"],
        )

        # Resize image and masks. Create other crops, etc
        image_mask_dict = self._process_image_mask_mess(
            raw_sample["rgb_image"], raw_sample["instance_masks"]
        )

        # Create pointmap
        pointmap_dict = self._get_pointmap(
            image_mask_dict["rgb_image"], raw_sample["depth_image"], cameras
        )

        camera_args = self._get_camera_args(cameras)

        return sample_uuid, {
            **image_mask_dict,
            "depth_image": raw_sample["depth_image"],  # Just z-channel of pointmap
            **instance_pose_dict,
            **camera_args,
            **instance_latents,
            **pointmap_dict,
            **mesh_dict,
        }
