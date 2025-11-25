from typing import List, Dict, Optional, Union, Tuple
from collections import namedtuple
from torch.utils.data import Dataset
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from lidra.data.dataset.tdfy.img_and_mask_transforms import (
    load_rgb,
    split_rgba,
    RGBAImageProcessor,
)


class AnythingInferenceFolderDataset(Dataset):
    def __init__(
        self,
        base_dir: Union[str, Path],
        metadata_csv_path: Optional[Union[str, Path]] = None,
        make_metadata_fname_glob_pattern: Optional[str] = None,
        image_mask_processor: Optional[RGBAImageProcessor] = None,
    ):
        # init super
        self.base_dir = base_dir
        self.metadata = self._load_metadata(
            base_dir, metadata_csv_path, make_metadata_fname_glob_pattern
        )
        if image_mask_processor is None:
            self.image_mask_processor = RGBAImageProcessor()
        else:
            self.image_mask_processor = image_mask_processor

    def _load_metadata(
        self,
        base_dir: Path | str,
        metadata_csv_path: Path | str,
        make_metadata_fname_glob_pattern: Optional[str] = None,
    ):
        if (
            not os.path.exists(metadata_csv_path)
            and make_metadata_fname_glob_pattern is not None
        ):
            metadata = self.make_metadata(
                base_dir, make_metadata_fname_glob_pattern, metadata_csv_path
            )
        else:
            metadata = pd.read_csv(metadata_csv_path)
        metadata.set_index("example_uuid", inplace=True)
        return metadata

    @staticmethod
    def make_metadata(
        base_dir: Path, fname_glob_pattern: str, save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        metadata = []
        base_dir = Path(base_dir)
        for fpath in tqdm(base_dir.glob(fname_glob_pattern)):
            metadata.append({"example_uuid": Path(fpath).stem, "fpath": fpath})
        df = pd.DataFrame(metadata)
        # df.set_index("example_uuid", inplace=True)
        if save_path is not None:
            df.to_csv(save_path)
        return df

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: Union[int, str]) -> Tuple[str, Path]:
        if isinstance(idx, int):
            example_uuid = self.metadata.index[idx]
        elif isinstance(idx, str):
            example_uuid = idx
        else:
            raise ValueError(f"Invalid index type: {type(idx)}")
        fpath = self.metadata.loc[example_uuid]["fpath"]

        image = load_rgb(fpath)
        assert image.shape[0] == 4, f"Image must have 4 channels, got {image.shape[0]=}"
        image, mask = split_rgba(image)
        image, mask = self.image_mask_processor(image, mask)

        return example_uuid, {
            "image": image,
            "mask": mask,
        }
