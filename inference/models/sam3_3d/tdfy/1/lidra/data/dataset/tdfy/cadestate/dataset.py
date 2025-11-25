import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch3d as pt3d
import torch
from tqdm import tqdm
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes

from lidra.data.dataset.tdfy.base_obj_in_scene_dataset import (
    ObjInSceneSingleSceneDataset,
)


class CADEstateSingleSceneDataset(ObjInSceneSingleSceneDataset):
    def __init__(
        self,
        cad_estate_rendered_bbox: bool,
        cad_estate_dir: str,
        video: str,
        is_train: bool,
        masked_img: bool,
        obj_center_norm: bool,
        add_context_to_bbox: float,
        preload_gt_pts=True,
        n_gt_pts=30000,
        frustum_visible=True,
        sample_freq=1,
    ):
        self.cad_estate_rendered_bbox = cad_estate_rendered_bbox
        processed_data_path = os.path.join(cad_estate_dir, video)
        self.sample_freq = sample_freq
        self.meta_data = torch.load(processed_data_path, weights_only=False)[
            :: self.sample_freq
        ]
        if len(self.meta_data) == 0:
            self.init = False
            return
        self.init = True
        self._clean_meta_data()
        self.scene_name = video.replace("/", "_")
        super().__init__(
            masked_img=masked_img,
            obj_center_norm=obj_center_norm,
            add_context_to_bbox=add_context_to_bbox,
            is_train=is_train,
            preload_gt_pts=preload_gt_pts,
            n_gt_pts=n_gt_pts,
            frustum_visible=frustum_visible,
        )
        self.name = "CADEstate"

    def _clean_meta_data(self):
        self.image_shape = None
        for frame_meta_data in self.meta_data:
            if "obj_segments" in frame_meta_data:
                del frame_meta_data["obj_segments"]
        if self.image_shape is None:
            self.image_shape = self.load_rgb((0, 0)).shape[:2]  # [H, W]

    def _get_scene_name(self):
        return self.scene_name

    def _get_idx_list(self):
        idx_list = []
        for i, frame_meta_data in enumerate(self.meta_data):
            for j, _ in enumerate(frame_meta_data["obj_mesh_file_paths"]):
                idx_list.append((i, j))
        return idx_list

    @staticmethod
    def convert_to_pytorch3d_mesh(triangle_vertices):
        N, _, _ = triangle_vertices.shape
        # Extract vertices for the i-th mesh

        # Reshape vertices to [M, 3] where M is the total number of vertices
        # and create a mapping from triangles to vertex indices
        # Note: this step assumes that all vertices in the triangle list are unique
        all_vertices = triangle_vertices.reshape(-1, 3)  # Shape [N*3, 3]
        unique_vertices, inverse_indices = torch.unique(
            all_vertices, dim=0, return_inverse=True
        )

        # Create faces based on the indices of unique vertices
        faces = inverse_indices.reshape(N, 3)  # Shape [N, 3]
        meshes = Meshes(verts=unique_vertices[None], faces=faces[None])
        return meshes

    def load_gt_points(self, mesh_path=None):
        # to handle compatibility with preloading; should not do this, will overwhelm memory
        if mesh_path is None:
            return None
        obj_triangle_mesh = np.load(mesh_path)["array"]
        pt3d_meshes = self.convert_to_pytorch3d_mesh(torch.tensor(obj_triangle_mesh))
        points = sample_points_from_meshes(pt3d_meshes, num_samples=self.n_gt_pts)[0]
        return points

    def _load_obj_info(self, idx):
        frame_idx, obj_idx = idx
        mesh_path = self.meta_data[frame_idx]["obj_mesh_file_paths"][obj_idx]
        points = self.load_gt_points(mesh_path=mesh_path)
        # no color
        color = torch.ones_like(points) / 2.0
        return [{"colors": color, "points": points}]

    def _prepare_cameras(self):
        self.color_cams = []
        for data in self.meta_data:
            e_mat = data["extrinsic"]
            R = (e_mat[:3, :3]).clone().float()
            T = (e_mat[:3, 3]).clone().float()
            T_world = -R.T @ T
            R = R.T @ pt3d.transforms.axis_angle_to_matrix(
                torch.FloatTensor([0, 0, torch.pi])
            ).transpose(-1, -2)
            T = -R.T @ T_world
            focal_length = torch.tensor([max(data["focal_length"])])
            cameras = PerspectiveCameras(
                R=R[None],
                T=T[None],
                focal_length=focal_length[None],
            )
            self.color_cams.append(cameras)

    def load_rgb(self, idx):
        image = plt.imread(self.meta_data[idx[0]]["image_file_path"]) * 255
        return image

    # TODO: check the depth map from RealEstate if they have it
    def load_depth(self, idx):
        return np.ones(self.image_shape)

    # TODO: check the depth map from RealEstate if they have it
    def _get_depth_cam(self, idx):
        return self.color_cams[idx[0]]

    def _get_color_cam(self, idx):
        return self.color_cams[idx[0]]

    # since SAM masks are not trustworthy, we return a box crop
    def _get_obj_mask(self, idx, obj_info):
        mask = np.zeros(self.image_shape)
        if self.cad_estate_rendered_bbox:
            if len(self.meta_data[idx[0]]["rendered_bbox"][idx[1]]) == 4:
                xmin, ymin, xmax, ymax = self.meta_data[idx[0]]["rendered_bbox"][idx[1]]
            else:
                xmin, ymin, xmax, ymax = self.meta_data[idx[0]]["obj_bboxes"][idx[1]]
        else:
            xmin, ymin, xmax, ymax = self.meta_data[idx[0]]["obj_bboxes"][idx[1]]
        mask[ymin:ymax, xmin:xmax] = 1.0
        return mask, 0


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,  # Train, Val
        preload_gt_pts,  # bool
        add_context_to_bbox,  # float
        frustum_visible,  # bool
        masked_img,  # bool
        n_gt_pts=20000,
        cad_estate_dir="/large_experiments/3dfy/datasets/cad_estate/obj_seg2cad/",
        cad_estate_split_json="/checkpoint/weiyaowang/data/cad-estate/data_split.json",
        cad_estate_rendered_bbox=True,
        obj_center_norm=True,
        is_viz=False,
    ):
        self.masked_img = masked_img
        self.obj_center_norm = obj_center_norm
        self.add_context_to_bbox = add_context_to_bbox

        self.preload_gt_pts = preload_gt_pts
        self.n_gt_pts = n_gt_pts
        self.frustum_visible = (frustum_visible,)
        self.dataset = self._get_all_scenes_in_split(
            cad_estate_dir, cad_estate_split_json, cad_estate_rendered_bbox, split
        )

    def _get_all_scenes_in_split(
        self,
        cad_estate_dir,
        cad_estate_split_json,
        cad_estate_rendered_bbox,
        split,
    ):
        # all_video_meta_data_path = os.listdir(cad_estate_dir)
        # hard code 1 out 10 as eval
        is_train = True if split == "train" else False
        with open(cad_estate_split_json, "r") as f:
            video_meta_data_path_split = json.load(f)
        if is_train:
            video_meta_data_path_split = video_meta_data_path_split["train"]
        else:
            video_meta_data_path_split = video_meta_data_path_split["val"]
        single_scene_datasets = []
        for video in tqdm(video_meta_data_path_split):
            single_scene_dataset = CADEstateSingleSceneDataset(
                cad_estate_dir=cad_estate_dir,
                cad_estate_rendered_bbox=cad_estate_rendered_bbox,
                video=video,
                is_train=is_train,
                masked_img=self.masked_img,
                obj_center_norm=self.obj_center_norm,
                add_context_to_bbox=self.add_context_to_bbox,
                preload_gt_pts=False,
                n_gt_pts=self.n_gt_pts,
                frustum_visible=self.frustum_visible,
                sample_freq=1 if is_train else 5,
            )
            if single_scene_dataset.init:
                single_scene_datasets.append(single_scene_dataset)
        dataset = torch.utils.data.dataset.ConcatDataset(single_scene_datasets)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
