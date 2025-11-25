import os
from glob import glob
import torch
from torch.utils.data.dataset import Dataset
from loguru import logger

from lidra.utils.decorators.counter import garbage_collect


# shape2vec : /fsx-3dfy/shared/vae_features/3dShape2Vec/objaverse
# xcube : /fsx-3dfy/shared/vae_features/XCube/objaverse


class Pth(Dataset):
    def __init__(self, path, extension=".pth", device="cpu", **load_kwargs):
        super().__init__()
        self.path = path
        self.device = device
        self.load_kwargs = load_kwargs
        self.extension = extension if extension[0] == "." else f".{extension}"
        self.filepaths = sorted(
            glob(os.path.join(self.path, "**", f"*{self.extension}"), recursive=True)
        )

    def __len__(self):
        return len(self.filepaths)

    @garbage_collect()
    def __getitem__(self, index):
        filepath = self.filepaths[index]
        relative_path = os.path.relpath(filepath, self.path)
        try:
            item = torch.load(
                filepath,
                map_location=self.device,
                weights_only=False,
                **self.load_kwargs,
            )
        except:
            logger.opt(exception=True).error(
                f"error while loading pth file : {filepath}"
            )
            return None
        return relative_path, item
