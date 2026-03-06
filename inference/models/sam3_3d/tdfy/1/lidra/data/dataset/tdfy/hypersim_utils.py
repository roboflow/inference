# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import h5py

import torch


def read_h5py(filename):
    with h5py.File(filename, "r") as f:
        data = torch.tensor(f["dataset"][:], dtype=torch.float32)
    return data


def read_img(frame_path):
    for retry in range(100):
        img = cv2.imread(frame_path)
        if img is not None:
            return torch.tensor(img / 255.0, dtype=torch.float32)[..., [2, 1, 0]]
        print("retry loading", retry, frame_path)
