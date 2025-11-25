from collections import defaultdict
import json
import os
import torch
import numpy as np
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    FoVPerspectiveCameras,
)
from pytorch3d.transforms import Transform3d, matrix_to_quaternion
from pytorch3d.io import load_ply
from loguru import logger
from typing import Dict, Tuple, Optional

from lidra.data.dataset.tdfy.transforms_3d import (
    compose_transform,
    decompose_transform,
    DecomposedTransform,
)
from lidra.data.dataset.tdfy.objaverse.utils import (
    blender2pytorch3d,
)
from .mesh_loader import load_trellis_mesh


def convert_to_decoupled_instance_pose(
    full_transform: Transform3d,
) -> Dict[str, torch.Tensor]:
    # Decomposed format expected by dataloader
    decomposed_transform = decompose_transform(full_transform)
    quat_l2c = matrix_to_quaternion(decomposed_transform.rotation)
    trans_l2c = decomposed_transform.translation
    scale_l2c = decomposed_transform.scale
    return {
        "instance_quaternion_l2c": quat_l2c,
        "instance_position_l2c": trans_l2c,
        "instance_scale_l2c": scale_l2c,
    }


def empty_pose(transforms_path: str) -> dict:
    return {}


def identity_pose(transforms_path: str, z_at_origin=False) -> dict:
    R_pytorch3d = torch.eye(3)
    T_pytorch3d = torch.zeros((1, 3))

    if not z_at_origin:
        T_pytorch3d[:, -1] = -1
    camera_pose = {
        "instance_quaternion_l2c": matrix_to_quaternion(R_pytorch3d).unsqueeze(0),
        "instance_position_l2c": T_pytorch3d,
        "instance_scale_l2c": torch.ones_like(T_pytorch3d),
        "camera_K": torch.eye(3).tolist(),
    }

    return camera_pose


def dummy_VLMranked_identity_pose(transforms_path: str, view_id, z_at_origin=False):
    return {
        "000.png": identity_pose(
            transforms_path=transforms_path, z_at_origin=z_at_origin
        )
    }


def dummy_arctic_hand_identity_pose(transforms_path: str, view_id, z_at_origin=False):
    return {
        f"rgba_{i:03d}.png": identity_pose(
            transforms_path=transforms_path, z_at_origin=z_at_origin
        )
        for i in range(10)
    }


def load_transforms_json(transforms_path: str) -> dict:
    with open(transforms_path, "r") as file:
        transforms_data = json.load(file)
    return transforms_data


class R3:
    @staticmethod
    def load_pose(transforms_path: str, file_name: str = "rgba_001.png") -> dict:
        if not os.path.exists(transforms_path):
            logger.opt(exception=False).warning(f"{transforms_path} does not exist.")
            return {}

        with open(transforms_path, "r") as file:
            transforms_data = json.load(file)

        frames_data = transforms_data["frames"][0]

        RR = torch.tensor(frames_data["RR"], dtype=torch.float32).unsqueeze(0)
        TT = torch.tensor(frames_data["TT"], dtype=torch.float32).unsqueeze(0)
        scene_to_obj_scale = frames_data["scale"]

        # Transform pose to PyTorch3D camera space
        full_transform = R3.r3_transform_json_to_trellis(RR, TT, scene_to_obj_scale)
        instance_pose = convert_to_decoupled_instance_pose(full_transform)

        # TODO: Renme "scales" from transforms.json. This is used for scaling the axes
        # of the R3 mesh to better match the annotation.
        # scales -> annotator_scaling_object_xyz?
        if not torch.allclose(
            instance_pose["instance_scale_l2c"],
            instance_pose["instance_scale_l2c"][..., -1].unsqueeze(-1),
        ):
            logger.opt(exception=False).warning(
                f"Scale is not isotropic. This is unexpected!! Are you using 'scales' instead of 'scale' in the transforms.json file? {instance_pose['instance_scale_l2c']}."
            )

        # Camera intrinsic matrix is in NDC space.
        image_size_hw = torch.tensor(frames_data["image_size"])
        camera_K = frames_data["intrinsic_matrix"]
        camera_K_ndc = R3.convert_camera_K_to_ndc(
            torch.tensor(camera_K)[:3, :3], image_size_hw
        )
        return {
            file_name: {
                **instance_pose,
                "camera_K": camera_K,
            }
        }

    @staticmethod
    def load_pose_general(transforms_path: str, view_id: str = None) -> dict:
        if not os.path.exists(transforms_path):
            logger.opt(exception=False).warning(f"{transforms_path} does not exist.")
            return {}

        with open(transforms_path, "r") as file:
            transforms_data = json.load(file)

        pose_dict = {}

        for frame_data in transforms_data["frames"]:
            file_name = os.path.basename(frame_data["file_path"])  # e.g. "rgba_003.png"
            if view_id is not None and file_name != view_id:
                continue

            RR = torch.tensor(frame_data["RR"], dtype=torch.float32).unsqueeze(0)
            TT = torch.tensor(frame_data["TT"], dtype=torch.float32).unsqueeze(0)
            scene_to_obj_scale = frame_data["scale"]

            full_transform = R3.r3_transform_json_to_trellis(RR, TT, scene_to_obj_scale)
            instance_pose = convert_to_decoupled_instance_pose(full_transform)

            scale_l2c = instance_pose["instance_scale_l2c"]
            if not torch.allclose(scale_l2c, scale_l2c[..., -1].unsqueeze(-1)):
                logger.opt(exception=False).warning(
                    f"Scale is not isotropic in '{file_name}'. "
                    f"This is unexpected!! Are you using 'scales' instead of 'scale' in the transforms.json file? {scale_l2c}."
                )

            image_size_hw = torch.tensor(frame_data["image_size"])
            camera_K = frame_data["intrinsic_matrix"]
            camera_K_ndc = R3.convert_camera_K_to_ndc(
                torch.tensor(camera_K), image_size_hw
            )

            pose_dict[file_name] = {
                **instance_pose,
                "camera_K": camera_K,
            }

        return pose_dict

    # TODO(Pierre) this should be removed when dataset pose if corrected (in data) by Fu-Jen.
    def _temporary_R3_load_pose_fix(transforms_path):
        import json
        import torch
        import numpy as np
        from loguru import logger
        from pytorch3d.transforms import matrix_to_quaternion
        from pytorch3d.renderer import look_at_view_transform

        with open(transforms_path, "r") as file:
            transforms_data = json.load(file)

        assert len(transforms_data["frames"]) == 1, "Only one frame is supported"
        assert transforms_data["frames"][0]["file_path"] == "rgba_001.png"

        frames_data = transforms_data["frames"][0]

        # load object pose from file
        object_R = torch.tensor(frames_data["RR"], dtype=torch.float32)
        object_T = torch.tensor(frames_data["TT"], dtype=torch.float32)
        object_S = torch.tensor([frames_data["scale"]], dtype=torch.float32)

        # initial R3 camera is shifted along the z axis (this is unconventional)
        starting_camera_R, starting_camera_T = look_at_view_transform(
            eye=np.array([[0, 0, -1]]),
            at=np.array([[0, 0, 0]]),
            up=np.array([[0, -1, 0]]),
            device="cpu",
        )

        # remove batch dimension
        starting_camera_R = starting_camera_R.squeeze(0)
        # make column vector
        object_T = object_T.unsqueeze(-1)

        # instead of applying transform to object vertices, apply to camera pose instead
        camera_T = object_T.T @ starting_camera_R + starting_camera_T * object_S
        camera_R = object_R.T @ starting_camera_R

        # convert camera instrinsic to NDC convention
        # https://pytorch3d.org/docs/cameras#perspectivecameras-orthographiccameras
        # <
        image_size_hw = torch.tensor(frames_data["image_size"])
        camera_K = torch.tensor(frames_data["intrinsic_matrix"], dtype=torch.float32)
        s = min(image_size_hw[0], image_size_hw[1])
        fx_ndc = camera_K[0][0] * 2.0 / s
        fy_ndc = camera_K[1][1] * 2.0 / s
        px_ndc = -(camera_K[0][2] - image_size_hw[1] / 2.0) * 2.0 / s
        py_ndc = -(camera_K[1][2] - image_size_hw[0] / 2.0) * 2.0 / s
        camera_K = torch.tensor(
            [
                [fx_ndc, 0, px_ndc],
                [0, fy_ndc, py_ndc],
                [0, 0, 1],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        # >

        return {
            "rgba_001.png": {
                "instance_quaternion_l2c": matrix_to_quaternion(camera_R.unsqueeze(0)),
                "instance_position_l2c": camera_T,
                "instance_scale_l2c": object_S,
                "camera_K": camera_K,
            }
        }

    @staticmethod
    def load_mesh(mesh_path: str) -> torch.Tensor:
        trellis_mesh = load_trellis_mesh(mesh_path)
        # Latents are in [-1,1]. Trellis meshes are in [-0.5, 0.5]
        trellis_mesh["mesh_vertices"] = trellis_mesh["mesh_vertices"]  # * 2
        return trellis_mesh

    @staticmethod
    def load_pointmap(
        image_basename: str,
        pointcloud_directory: str,
        sha256: str = None,
        image_fname: str = None,
        file_identifier: str = None,
    ) -> torch.Tensor:
        # Try standard format first (for moge)
        standard_dir = os.path.join(pointcloud_directory, image_basename)

        if os.path.exists(standard_dir):
            # Use standard moge format
            dirname = image_basename
            base_path = standard_dir
        elif sha256 and file_identifier:
            file_identifier = file_identifier.split("/")[-1]
            dirname = f"{image_basename}_{sha256}_{file_identifier}"
            base_path = os.path.join(pointcloud_directory, dirname)
            if not os.path.exists(base_path):
                raise FileNotFoundError(
                    f"Pointmap directory not found for {image_basename} at {base_path}. "
                )
        elif sha256 and image_fname:
            # Extract frame number from image filename (e.g., "rgba_003.png" -> "003")
            frame = (
                image_fname.replace("rgba_", "").replace(".png", "")
                if "rgba_" in image_fname
                else None
            )
            if not frame:
                raise ValueError(
                    f"Could not extract frame number from image_fname: {image_fname}"
                )

            # Try moge_corrected format
            dirname = f"{image_basename}_{sha256}_{frame}"
            base_path = os.path.join(pointcloud_directory, dirname)
            if not os.path.exists(base_path):
                raise FileNotFoundError(
                    f"Pointmap directory not found for {image_basename}. "
                    f"Tried: {standard_dir} and {base_path}"
                )
        else:

            raise FileNotFoundError(
                f"Pointmap directory not found at {standard_dir} and sha256/image_fname not provided for corrected format"
            )

        return R3._load_pointmap(base_path, dirname)

    @staticmethod
    def _load_pointmap(base_path: str, base_name: str) -> torch.Tensor:
        # Load the pointmap files
        ply_path = os.path.join(base_path, f"{base_name}.ply")
        metadata_path = os.path.join(base_path, f"{base_name}_metadata.json")

        verts, faces = load_ply(ply_path)
        with open(metadata_path, "r") as f:
            pointmap_metadata = json.load(f)
        image_dims = pointmap_metadata["image_size"]
        points_tensor = verts.reshape(image_dims[1], image_dims[0], 3)

        camera_convention_transform = Transform3d().rotate(
            R3.r3_camera_to_pytorch3d_camera().rotation
        )
        points_tensor = camera_convention_transform.transform_points(points_tensor)

        # Replace non-finite values (inf, -inf, nan) with nan so they are handled properly
        # This is important for moge_corrected pointmaps which may contain inf values
        points_tensor = torch.where(
            torch.isfinite(points_tensor), points_tensor, torch.nan
        )

        return points_tensor

    @staticmethod
    def r3_transform_json_to_trellis(
        RR, TT, scale_scene_to_instance, device="cpu"
    ) -> Transform3d:
        # Object space --> R3 camera space
        object_to_scene_scale = 1 / scale_scene_to_instance
        object_to_r3_cam_transform = (
            Transform3d()
            .rotate(RR[0, :3, :3].T)  # Bizarrely, PyTorch3d is row-major
            .translate(TT)
            .scale(object_to_scene_scale)
            .to(device)
        )

        # R3 camera space --> PyTorch3D camera space
        r3_to_p3d_decomposed = R3.r3_camera_to_pytorch3d_camera(device)
        r3_to_p3d_cam_transform = compose_transform(*r3_to_p3d_decomposed)

        full_transform = object_to_r3_cam_transform.compose(r3_to_p3d_cam_transform)
        return full_transform

    @staticmethod
    def r3_camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
        """
        R3 camera space --> PyTorch3D camera space
        Also needed for pointmaps
        """
        r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
            eye=np.array([[0, 0, -1]]),
            at=np.array([[0, 0, 0]]),
            up=np.array([[0, -1, 0]]),
            device=device,
        )
        return DecomposedTransform(
            rotation=r3_to_p3d_R,
            translation=r3_to_p3d_T,
            scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
        )

    @staticmethod
    def convert_camera_K_to_ndc(
        camera_K: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        K = torch.eye(4).unsqueeze(0)
        K[0, :3, :3] = camera_K
        cam = PerspectiveCameras(K=K, in_ndc=False, image_size=image_size)
        ndc_transform = cam.get_projection_transform().compose(
            cam.get_ndc_camera_transform()
        )
        return ndc_transform.get_matrix()[0, :3, :3].T


def load_trellis_pose(transforms_path: str, view_id: Optional[str] = None) -> dict:
    with open(transforms_path, "r") as file:
        transforms_data = json.load(file)
    frames_data = transforms_data["frames"]
    Rs, Ts, fovs = [], [], []
    file_paths = []
    for i in range(len(frames_data)):
        transform = np.array(frames_data[i]["transform_matrix"])
        fov = frames_data[i]["camera_angle_x"]
        rotation = transform[:3, :3]
        location = transform[:3, -1]
        R_world2bcam = rotation.T
        T_world2bcam = -1 * R_world2bcam @ location
        Rs.append(torch.from_numpy(np.array(R_world2bcam)[:3, :3]))
        Ts.append(torch.from_numpy(np.array(T_world2bcam)))
        fovs.append(fov)
        file_paths.append(frames_data[i]["file_path"])

    Rs = torch.stack(Rs)
    Ts = torch.stack(Ts)
    cvt = [blender2pytorch3d(R, T) for R, T in zip(Rs, Ts)]
    Rs_pytorch3d = torch.stack([c[0] for c in cvt])
    Ts_pytorch3d = torch.cat([c[1] for c in cvt], dim=0)

    camera_poses = {}
    for cam_id in range(len(Rs_pytorch3d)):
        camera = FoVPerspectiveCameras(
            R=Rs_pytorch3d[cam_id][None],
            T=Ts_pytorch3d[cam_id][None],
            # fov=fovs[cam_id] * 180 / np.pi,
            # device=Rs.device,
        )
        camera_K = camera.compute_projection_matrix(
            znear=0.1,
            zfar=100000,
            fov=fovs[cam_id] * 180 / np.pi,
            aspect_ratio=1.0,
            degrees=True,
        )[0, :3, :3]

        camera_poses[file_paths[cam_id]] = {
            "instance_quaternion_l2c": matrix_to_quaternion(
                Rs_pytorch3d[cam_id]
            ).unsqueeze(0),
            "instance_position_l2c": Ts_pytorch3d[cam_id].unsqueeze(0),
            "instance_scale_l2c": torch.ones_like(
                Ts_pytorch3d[cam_id][..., -1]
            ).unsqueeze(0),
            "camera_K": camera_K,
        }

    return camera_poses


def load_trellis_pose_w_scale(
    transforms_path: str, view_id: Optional[str] = None
) -> dict:
    camera_poses = load_trellis_pose(transforms_path)
    with open(transforms_path, "r") as file:
        transforms_data = json.load(file)
    scale = transforms_data["scale"]
    for _, cam_info in camera_poses.items():
        cam_info["instance_scale_l2c"] = (
            torch.ones_like(cam_info["instance_position_l2c"]) * scale
        )
    return camera_poses


# R3.load_pose = R3._temporary_R3_load_pose_fix


def load_pose_objaversev1old(pose_path: str, view_id: Optional[str] = None) -> dict:
    camera_pos = np.load(pose_path)
    R, T = camera_pos[:3, :3], camera_pos[:3, -1]
    R_pytorch3d, T_pytorch3d = blender2pytorch3d(
        torch.from_numpy(R), torch.from_numpy(T)
    )

    # This is a magic transform that just works to transform to Trellis latent
    R_pytorch3d = (
        torch.Tensor(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        ).to(R_pytorch3d.device)
    ).T @ R_pytorch3d

    camera_pose = {
        "instance_quaternion_l2c": matrix_to_quaternion(R_pytorch3d).unsqueeze(0),
        "instance_position_l2c": T_pytorch3d,
        # "instance_scale_l2c": torch.ones_like(T_pytorch3d[..., -1]).unsqueeze(0),
        "instance_scale_l2c": torch.ones_like(T_pytorch3d),
    }

    return camera_pose


def load_objaversev1old_pose(pose_path: str, view_id: Optional[str] = None) -> dict:
    camera_pos = np.load(pose_path)
    R, T = camera_pos[:3, :3], camera_pos[:3, -1]
    R_pytorch3d, T_pytorch3d = blender2pytorch3d(
        torch.from_numpy(R), torch.from_numpy(T)
    )

    # This is a magic transform that just works to transform to Trellis latent
    R_pytorch3d = (
        torch.Tensor(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        ).to(R_pytorch3d.device)
    ).T @ R_pytorch3d

    camera_pose = {
        "instance_quaternion_l2c": matrix_to_quaternion(R_pytorch3d).unsqueeze(0),
        "instance_position_l2c": T_pytorch3d,
        "instance_scale_l2c": torch.ones_like(T_pytorch3d[..., -1]).unsqueeze(0),
    }

    return camera_pose
