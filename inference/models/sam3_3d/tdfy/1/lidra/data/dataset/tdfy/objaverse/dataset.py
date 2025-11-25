from collections import namedtuple
import torch
import os
import trimesh
from loguru import logger


from lidra.data.dataset.tdfy.objaverse.loaders.surface import load_surface_points
from lidra.utils.decorators.counter import garbage_collect


class Dataset(torch.utils.data.Dataset):
    VALID_SPLITS = {"train", "val"}

    # TODO(Pierre) : Move this to loader
    SURFACE_SAMPLES = 20_000

    class ModelTooLarge(RuntimeError):
        def __init__(self, current_size, limit_size):
            super().__init__(
                f"model file is too large (size: {current_size}, limit: {limit_size})"
            )

    def __init__(
        self,
        path,
        split,
        scale=6,
        preprocessed=True,
        size_limit_bytes=200 * 1024 * 1024,
        n_queries=550,
        rendering_folder_name="views_lookat_01_lens_35_70",
        loading_fn=load_surface_points,
    ):

        self.path = path
        self.scale = scale
        self.use_preprocessed = preprocessed
        self.size_limit_bytes = size_limit_bytes  # in bytes
        self.n_queries = n_queries
        self.rendering_folder_name = rendering_folder_name
        self.loading_fn = loading_fn

        assert (
            split in Dataset.VALID_SPLITS
        ), f"split should be in {Dataset.VALID_SPLITS}"
        self._load_uids(split)

    def _load_uids(self, split):
        valid_files = f"{self.path}/splits/valid_{split}.txt"
        with open(valid_files, "r") as f:
            self.uids = f.read().splitlines()
        logger.info(f"{len(self.uids)} uids found")

    def _check_size(self, path):
        size = os.path.getsize(path)
        if size > self.size_limit_bytes:
            raise Dataset.ModelTooLarge(size, self.size_limit_bytes)

    def __len__(self) -> int:
        return len(self.uids)

    def _load_mesh(self, uid):
        if self.use_preprocessed:
            preprocessed_path = f"{self.path}/preprocessed_models"
            model_path = os.path.join(preprocessed_path, uid, "model_concat_100k.glb")
        else:
            seq_dir = os.path.join(self.path, self.rendering_folder_name, uid)
            model_path = os.path.join(seq_dir, "model.glb")

        try:
            self._check_size(model_path)
            geometry = trimesh.load(model_path, force="mesh")
        except:
            logger.opt(exception=True).warning(f"error loading file {model_path}")
            return None

        return geometry

    @garbage_collect()
    def __getitem__(self, index):
        uid = self.uids[index]

        geometry = self._load_mesh(uid)

        if geometry is None:
            return None
        return self.loading_fn(self, uid, geometry)
