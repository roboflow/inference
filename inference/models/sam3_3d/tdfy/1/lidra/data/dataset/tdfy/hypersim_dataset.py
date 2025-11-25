# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import glob

import torch
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.ops import sample_points_from_meshes

from lidra.data.dataset.tdfy.hypersim_utils import read_h5py, read_img


def hypersim_collate_fn(batch):
    assert len(batch[0]) == 4
    return (
        FrameData.collate([x[0] for x in batch]),
        FrameData.collate([x[1] for x in batch]),
        FrameData.collate([x[2] for x in batch]),
        [x[3] for x in batch],
    )


def is_good_xyz(xyz):
    assert len(xyz.shape) == 3
    return (torch.isfinite(xyz.sum(axis=2))).sum() > 2000


def get_camera_pos_file_name_from_frame_name(frame_name):
    tmp = frame_name.split("/")
    tmp[-3] = "_detail"
    tmp[-2] = "cam_" + tmp[-2].split("_")[2]
    tmp[-1] = "camera_keyframe_positions.hdf5"
    return "/".join(tmp)


def get_camera_look_at_file_name_from_frame_name(frame_name):
    tmp = frame_name.split("/")
    tmp[-3] = "_detail"
    tmp[-2] = "cam_" + tmp[-2].split("_")[2]
    tmp[-1] = "camera_keyframe_look_at_positions.hdf5"
    return "/".join(tmp)


def get_camera_orientation_file_name_from_frame_name(frame_name):
    tmp = frame_name.split("/")
    tmp[-3] = "_detail"
    tmp[-2] = "cam_" + tmp[-2].split("_")[2]
    tmp[-1] = "camera_keyframe_orientations.hdf5"
    return "/".join(tmp)


def read_scale_from_frame_name(frame_name):
    tmp = frame_name.split("/")
    with open("/".join(tmp[:-3] + ["_detail", "metadata_scene.csv"])) as f:
        for line in f:
            items = line.split(",")
    return float(items[1])


def random_crop(xyz, img, is_train=True):
    assert xyz.shape[0] == img.shape[0]
    assert xyz.shape[1] == img.shape[1]

    width, height = img.shape[0], img.shape[1]
    w = h = min(width, height)
    if is_train:
        i = torch.randint(0, width - w + 1, size=(1,)).item()
        j = torch.randint(0, height - h + 1, size=(1,)).item()
    else:
        i = (width - w) // 2
        j = (height - h) // 2
    xyz = xyz[i : i + w, j : j + h]
    img = img[i : i + w, j : j + h]
    xyz = torch.nn.functional.interpolate(
        xyz[None].permute(0, 3, 1, 2),
        (112, 112),
        mode="bilinear",
    ).permute(0, 2, 3, 1)[0]
    img = torch.nn.functional.interpolate(
        img[None].permute(0, 3, 1, 2),
        (224, 224),
        mode="bilinear",
    ).permute(0, 2, 3, 1)[0]
    return xyz, img


class HyperSimDataset(torch.utils.data.Dataset):
    def __init__(self, args, is_train, is_viz=False, **kwargs):

        self.args = args
        self.is_train = is_train
        self.is_viz = is_viz

        self.dataset_split = "train" if is_train else "val"
        self.scene_names = self.load_scene_names(is_train)

        if not is_train:
            self.meshes = self.load_meshes()

        self.hypersim_gt = self.load_hypersim_gt()

    def load_hypersim_gt(self):
        gt_filename = (
            "hypersim_gt_train.pt"
            if self.dataset_split == "train"
            else "hypersim_gt_val.pt"
        )
        print("loading GT file from", gt_filename)
        gt = torch.load(
            os.path.join(self.args.hypersim_path, "evermotion_dataset", gt_filename)
        )
        for scene_name in gt.keys():
            good = torch.isfinite(gt[scene_name][0].sum(axis=1)) & torch.isfinite(
                gt[scene_name][1].sum(axis=1)
            )

            # Subsample GT to reduce memory usage.
            if self.is_train:
                good = good & (torch.rand(good.shape) < 0.5)
            else:
                good = good & (torch.rand(good.shape) < 0.1)
            gt[scene_name] = [gt[scene_name][0][good], gt[scene_name][1][good]]
        return gt

    def load_meshes(self):
        return torch.load(
            os.path.join(
                self.args.hypersim_path,
                "evermotion_dataset/dataset_cache/all_hypersim_val_meshes.pt",
            )
        )

    def load_scene_names(self, is_train):
        split = "train" if is_train else "test"
        with open(
            os.path.join(
                self.args.hypersim_path, "evermotion_dataset/hypersim_missing_mesh.txt"
            ),
            "r",
        ) as f:
            lines = f.readlines()
        missing_scenes = [x.strip() for x in lines]

        scene_names = []
        with open(
            os.path.join(
                self.args.hypersim_path,
                "evermotion_dataset/analysis/metadata_images_split_scene_v1.csv",
            ),
            "r",
        ) as f:
            for line in f:
                items = line.split(",")
                if items[-1].strip() == split and items[0] not in missing_scenes:
                    scene_names.append(items[0])
        scene_names = sorted(list(set(scene_names)))
        print(len(scene_names), "scenes loaded:", scene_names)
        return scene_names

    def is_corrupted_frame(self, frame):
        return ("ai_003_001" in frame and "cam_00" in frame) or (
            "ai_004_009" in frame and "cam_01" in frame
        )

    def get_hypersim_data(self, index):
        for retry in range(1000):
            try:
                if retry < 10:
                    scene_name = self.scene_names[index % len(self.scene_names)]
                else:
                    scene_name = random.choice(self.scene_names)

                frames = glob.glob(
                    os.path.join(
                        self.args.hypersim_path,
                        "evermotion_dataset",
                        "scenes",
                        scene_name,
                        "images/scene_cam_*_final_preview/*tonemap*",
                    )
                )
                seen_frame = random.choice(frames)

                if self.is_corrupted_frame(seen_frame):
                    continue

                seen_data = self.load_frame_data(seen_frame)
                if not is_good_xyz(seen_data[0]):
                    continue

                cur_gt = self.hypersim_gt[scene_name]
                gt_data = [cur_gt[0], cur_gt[1]]

                if self.is_train:
                    mesh_points = torch.zeros((1,))
                else:
                    mesh_points = sample_points_from_meshes(
                        self.meshes[scene_name], 1000000
                    )

                # get camera positions
                camera_positions = read_h5py(
                    get_camera_pos_file_name_from_frame_name(seen_frame)
                )
                camera_position = camera_positions[int(seen_frame.split(".")[-3])]

                # get camera orientations
                cam_orientations = read_h5py(
                    get_camera_orientation_file_name_from_frame_name(seen_frame)
                )
                cam_orientation = cam_orientations[int(seen_frame.split(".")[-3])]
                cam_orientation = cam_orientation * (-1.0)

                # rotate to camera direction
                seen_data[0] = torch.matmul(seen_data[0], cam_orientation)
                gt_data[0] = torch.matmul(gt_data[0], cam_orientation)

                # shift to camera center
                camera_position = torch.matmul(camera_position, cam_orientation)
                seen_data[0] -= camera_position
                gt_data[0] -= camera_position
                # to meter
                asset_to_meter_scale = read_scale_from_frame_name(seen_frame)
                seen_data[0] = seen_data[0] * asset_to_meter_scale
                gt_data[0] = gt_data[0] * asset_to_meter_scale

                # get points GT
                n_gt = 30000
                in_front_of_cam = gt_data[0][..., 2] > 0
                if in_front_of_cam.sum() < 1000:
                    print("Warning! Not enough in front of cam.", in_front_of_cam.sum())
                    continue
                gt_data = [gt_data[0][in_front_of_cam], gt_data[1][in_front_of_cam]]

                if in_front_of_cam.sum() < n_gt:
                    selected = random.choices(range(gt_data[0].shape[0]), k=n_gt)
                else:
                    selected = random.sample(range(gt_data[0].shape[0]), n_gt)
                gt_data = [gt_data[0][selected], gt_data[1][selected]]

                if not self.is_train:
                    mesh_points = torch.matmul(mesh_points, cam_orientation)
                    mesh_points -= camera_position * asset_to_meter_scale
                    in_front_of_cam = mesh_points[..., 2] > 0
                    if in_front_of_cam.sum() < 1000:
                        print(
                            "Warning! Not enough mesh in front of cam.",
                            in_front_of_cam.sum(),
                        )
                        continue
                    mesh_points = mesh_points[in_front_of_cam]
                    if in_front_of_cam.sum() < n_gt:
                        selected = random.choices(range(mesh_points.shape[0]), k=n_gt)
                    else:
                        selected = random.sample(range(mesh_points.shape[0]), n_gt)
                    mesh_points = mesh_points[selected][None]
                    mesh_points[..., 0] *= -1

                seen_data[0][..., 0] *= -1
                gt_data[0][..., 0] *= -1

                seen_data[1] = seen_data[1].permute(2, 0, 1)

                return seen_data, gt_data, mesh_points, scene_name
            except Exception as e:
                print(scene_name, index, "loading failed", retry, e)

    def __getitem__(self, index):

        seen_data, gt_data, mesh_points, scene_name = self.get_hypersim_data(index)

        # normalize the data
        example_std = get_example_std(seen_data[0])
        seen_data[0] = seen_data[0] / example_std
        gt_data[0] = gt_data[0] / example_std
        mesh_points = mesh_points / example_std

        # TODO: fix this long term
        # Disable meshes for now
        mesh_points = torch.zeros((1,))

        return (
            seen_data,
            gt_data,
            mesh_points,
            f"hypersim_{scene_name}",
        )

    def load_frame_data(self, frame_path):
        frame_xyz_path = frame_path.replace("final_preview/", "geometry_hdf5/").replace(
            ".tonemap.jpg", ".position.hdf5"
        )
        xyz = read_h5py(frame_xyz_path)
        img = read_img(frame_path)

        xyz, img = random_crop(
            xyz,
            img,
            is_train=self.is_train,
        )
        return [xyz, img]

    def __len__(self) -> int:
        if self.is_train:
            return int(len(self.scene_names) * self.args.train_epoch_len_multiplier)
        elif self.is_viz:
            return len(self.scene_names)
        else:
            return int(len(self.scene_names) * self.args.eval_epoch_len_multiplier)


def get_example_std(x):
    x = x.reshape(-1, 3)
    x = x[torch.isfinite(x.sum(dim=1))]
    return x.std(dim=0).mean().detach()
