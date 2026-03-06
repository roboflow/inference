import json
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch3d as pt3d
import torch
from PIL import Image
from tqdm import tqdm
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation as SciR

from lidra.data.dataset.tdfy.base_obj_in_scene_dataset import (
    ObjInSceneSingleSceneDataset,
)


class R3SingleSceneDataset(ObjInSceneSingleSceneDataset):
    def __init__(
        self,
        r3_estate_dir: str,
        ann_id: str,
        is_train: bool,
        masked_img: bool,
        obj_center_norm: bool,
        add_context_to_bbox: float,
        camera_data: dict,
        preload_gt_pts=True,
        n_gt_pts=30000,
        frustum_visible=True,
        dataset_prefix=None,
        normalize_img=False,
    ):
        self.main_data_path = os.path.join(r3_estate_dir, ann_id)
        self.init = True
        self.dataset_prefix = dataset_prefix
        try:
            self._prep_meta_data()
        except:
            print(f"{ann_id} init failed")
            self.init = False
            return
        self.scene_name = ann_id
        self.camera_data = camera_data
        super().__init__(
            masked_img=masked_img,
            obj_center_norm=obj_center_norm,
            add_context_to_bbox=add_context_to_bbox,
            is_train=is_train,
            preload_gt_pts=preload_gt_pts,
            n_gt_pts=n_gt_pts,
            frustum_visible=frustum_visible,
            padding_value=1,
            black_bg=False,
            normalize_img=normalize_img,
        )
        self.name = "R3"
        self.pt3d_io = IO()
        self.pt3d_io.register_meshes_format(MeshGlbFormat())

    def _prep_meta_data(self):
        required_files = ["fname.txt", "mask.png"]
        for required_file in required_files:
            assert os.path.exists(
                os.path.join(self.main_data_path, required_file)
            ), f"{(self.scene_name, required_file)} not exists"
        with open(os.path.join(self.main_data_path, "fname.txt"), "r") as fname_f:
            self.img_path = fname_f.readline().strip()
            if self.dataset_prefix is not None:
                self.img_path = (
                    self.dataset_prefix
                    + self.img_path[len("/datasets01/segment_anything") :]
                )
        with Image.open(self.img_path) as img:
            width, height = img.size
        self.image_shape = np.array([height, width])

    def _get_idx_list(self):
        return [0]

    def load_gt_points(self):
        mesh_path = os.path.join(self.main_data_path, "mesh.glb")
        if not os.path.exists(mesh_path):
            mesh_path = os.path.join(self.main_data_path, "mesh_color.glb")
            if not os.path.exists(mesh_path):
                return torch.ones([3, self.n_gt_pts]).float()
        pt3d_meshes = self.pt3d_io.load_mesh(mesh_path, device="cpu")
        points = sample_points_from_meshes(pt3d_meshes, num_samples=self.n_gt_pts)[0]
        points *= self.isotropic_scales
        return points

    def _load_obj_info(self, idx):
        points = self.load_gt_points()
        # no color
        color = torch.ones_like(points) / 2.0
        return [{"colors": color, "points": points}]

    def _prepare_cameras(self):
        self.color_cams = []
        d = self.camera_data
        euler = [d["rotations"][0], d["rotations"][1], d["rotations"][2]]
        trans = [d["translations"][0], d["translations"][1], d["translations"][2]]
        self.isotropic_scales = torch.tensor(
            [d["scale"][0], d["scale"][1], d["scale"][2]]
        ).float()

        TT = torch.Tensor(trans)[None, :]
        RR = torch.from_numpy(SciR.from_euler("XYZ", euler, degrees=False).as_matrix())[
            None, :
        ].float()
        RT = torch.cat((RR, TT[..., None]), 2).to("cpu")
        R, T = look_at_view_transform(
            eye=np.array([[0, 0, -1]]),
            at=np.array([[0, 0, 0]]),
            up=np.array([[0, -1, 0]]),
            device="cpu",
        )
        ext_cam_transform = PerspectiveCameras(
            R=RT[:, :3, :3],
            T=RT[:, :3, 3],
            # dummy focal
            focal_length=torch.tensor([0.5, 0.5])[None],
        ).get_world_to_view_transform()
        convention_transform = PerspectiveCameras(
            R=R,
            T=T,
            # dummy focal
            focal_length=torch.tensor([0.5, 0.5])[None],
        ).get_world_to_view_transform()
        real_RT = ext_cam_transform.compose(convention_transform).get_matrix()
        cameras = PerspectiveCameras(
            R=real_RT[:, :3, :3],
            T=real_RT[:, 3, :3],
            # dummy focal
            focal_length=torch.tensor([0.5, 0.5])[None],
        )
        self.color_cams.append(cameras)

    def load_rgb(self, idx):
        image = plt.imread(self.img_path)
        if image.dtype != "uint8":
            image = image * 255
        return image

    def load_depth(self, idx):
        return np.ones(self.image_shape)

    def _get_depth_cam(self, idx):
        return self.color_cams[0]

    def _get_color_cam(self, idx):
        return self.color_cams[0]

    def _get_obj_mask(self, idx, obj_info):
        mask = plt.imread(os.path.join(self.main_data_path, "mask.png")) > 0
        return mask, 0


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,  # Train, Val
        preload_gt_pts,
        add_context_to_bbox,  # float
        frustum_visible,  # bool
        masked_img,  # bool
        n_gt_pts=20000,
        r3_dir="/private/home/xingyuchen/xingyuchen/r3_mesh/",
        r3_split_json="/large_experiments/3dfy/datasets/r3_versions/20241028_splits.json",
        obj_center_norm=True,
        is_viz=False,
    ):
        # split = "train" if is_train else "val"

        self.masked_img = masked_img
        self.obj_center_norm = obj_center_norm
        self.add_context_to_bbox = add_context_to_bbox

        self.preload_gt_pts = preload_gt_pts
        self.n_gt_pts = n_gt_pts
        self.frustum_visible = (frustum_visible,)
        self.r3_split_json = r3_split_json

        self.dataset = self._get_all_scenes_in_split(r3_dir, split)

    def _get_all_scenes_in_split(self, r3_dir, split):
        is_train = True if split == "train" else False

        with open(self.r3_split_json, "r") as f:
            meta_data_path_split = json.load(f)
        annotation_path = os.path.join(r3_dir, "annotations.json")
        with open(annotation_path) as f:
            annotations = json.load(f)
        if is_train:
            meta_data_path_split = meta_data_path_split["train"]
        else:
            meta_data_path_split = meta_data_path_split["val"]
        single_scene_datasets = []

        for ann_id in tqdm(meta_data_path_split):
            single_scene_dataset = R3SingleSceneDataset(
                r3_estate_dir=r3_dir,
                ann_id=ann_id,
                is_train=is_train,
                masked_img=self.masked_img,
                obj_center_norm=self.obj_center_norm,
                add_context_to_bbox=self.add_context_to_bbox,
                camera_data=annotations[str(ann_id)],
                preload_gt_pts=False,
                n_gt_pts=self.n_gt_pts,
                frustum_visible=self.frustum_visible,
            )
            if single_scene_dataset.init:
                single_scene_datasets.append(single_scene_dataset)
        dataset = torch.utils.data.dataset.ConcatDataset(single_scene_datasets)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
