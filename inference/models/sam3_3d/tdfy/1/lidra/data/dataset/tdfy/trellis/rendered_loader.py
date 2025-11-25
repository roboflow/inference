import torch
import json
import numpy as np
from glob import glob
import os
from PIL import Image
import utils3d
import random


def load_rendered_features(data_path, frame_=None, n_frames=1):

    #########init json paths and data paths#####################
    #####example test path: /fsx-3dfy-v2/shared/datasets/trellis500k/ObjaverseXL_github/renders/000006a1bf62e68bc2029329a937e55348547c8194458175d3534c9b0592b60b
    metadata_path = os.path.join(data_path, "transforms.json")
    metadata = load_json_file(metadata_path)
    n_views = len(metadata["frames"])

    images, alphas, extrinsics_list, intrinsics_list = [], [], [], []

    frames = random.sample(
        range(n_views), n_frames
    )  ###sample n_frames from n_views, adjust n_frames based on your need for inference and training
    for itr in range(n_frames):
        frame = frames[itr]  ###randomly sampled frames
        if frame_ is not None:
            frame = frame_  ########fix the frame number for testing for quantitative result reproducibility
        metadata_cur = metadata["frames"][frame]

        fov = metadata_cur["camera_angle_x"]
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(
            torch.tensor(fov), torch.tensor(fov)
        )
        c2w = torch.tensor(metadata_cur["transform_matrix"])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(data_path, metadata_cur["file_path"])
        image = Image.open(image_path)
        alpha = image.getchannel(3)
        image = image.convert("RGB")

        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0

        images.append(image)
        alphas.append(alpha)
        extrinsics_list.append(extrinsics)
        intrinsics_list.append(intrinsics)

    images = torch.stack(images, dim=0)
    alphas = torch.stack(alphas, dim=0)
    extrinsics = torch.stack(extrinsics_list, dim=0)
    intrinsics = torch.stack(intrinsics_list, dim=0)

    return {
        "image_rendered": images,
        "alpha_rendered": alphas,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }


def load_json_file(json_path):
    json_dic = None
    with open(json_path, "r") as f:
        json_dic = json.load(f)

    return json_dic
