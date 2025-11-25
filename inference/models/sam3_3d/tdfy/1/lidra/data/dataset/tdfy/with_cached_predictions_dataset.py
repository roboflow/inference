from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import pickle
import torch
from typing import Any, Union
from tqdm import tqdm
from lidra.data.dataset.return_type import SampleUuidUtils
from loguru import logger


class WithCachedPredictionDataset(Dataset):
    FNAME_REPLACE_CHARS = {
        "/": "%2F",
    }
    FNAME_FIELDSEP = "___"
    TRIAL_NAME_PREFIX = "trial_"
    TRIAL_NAME_SEP = "____"

    def __init__(
        self,
        sample_dataset: Dataset,
        prediction_folder: Union[Path, str],
        device: torch.device = torch.device("cpu"),  # Only use cuda for debugging
    ):
        super().__init__()
        self.sample_dataset = sample_dataset
        self.prediction_folder = Path(prediction_folder)
        logger.info(f"Loading predictions from {self.prediction_folder}")
        self.device = device

        # Load or create metadata index
        metadata_path = self.prediction_folder / "metadata.csv"
        if not metadata_path.exists():
            self.prediction_metadata = self._make_metadata()
            if len(self.prediction_metadata) > 0:
                self.prediction_metadata.to_csv(metadata_path, index=False)
        else:
            self.prediction_metadata = pd.read_csv(metadata_path)

    def _make_metadata(self) -> pd.DataFrame:
        records = []
        # Look for .pt files in the main folder and all subdirectories
        for fpath in tqdm(
            list(self.prediction_folder.glob("**/*.pt")),
            desc="Making metadata for cached predictions",
        ):
            # Skip metadata.pt if it exists
            if fpath.name == "metadata.pt":
                continue

            example_uuid, _ = torch.load(fpath, map_location=self.device)
            # Convert namedtuple to tuple for pandas storage
            if isinstance(example_uuid, tuple) or isinstance(example_uuid, list):
                example_uuid = tuple(example_uuid)
            elif isinstance(example_uuid, str):
                pass
            else:
                raise ValueError(f"Unexpected example_uuid type: {type(example_uuid)}")

            # Store the relative path from the prediction folder
            rel_path = fpath.relative_to(self.prediction_folder)
            records.append({"example_uuid": example_uuid, "fname": str(rel_path)})
        return pd.DataFrame(records)

    def _load_prediction(self, fname: str) -> tuple[Any, dict]:
        data = torch.load(self.prediction_folder / fname, map_location=self.device)
        example_uuid, prediction = data

        # Convert tuple back to original namedtuple type if needed
        if (
            isinstance(example_uuid, tuple) and hasattr(example_uuid, "_fields")
        ) and self.sample_dataset is not None:
            uuid_type = type(self.sample_dataset[0][0])
            example_uuid = uuid_type(*example_uuid)

        return example_uuid, prediction

    @staticmethod
    def make_safe_filename(example_uuid: Any, trial_num: int = None) -> str:
        example_uuid = SampleUuidUtils.demote(example_uuid)

        # Convert example_uuid to string for filename
        # if hasattr(example_uuid, "_fields"):  # namedtuple
        if isinstance(example_uuid, tuple):  # namedtuple
            # Use the field values joined by underscores
            example_dir = WithCachedPredictionDataset.FNAME_FIELDSEP.join(
                str(x) for x in example_uuid
            )
        else:
            example_dir = str(example_uuid)

        # Make fname safe for filename by escaping slashes
        replace_chars = WithCachedPredictionDataset.FNAME_REPLACE_CHARS
        for char, replace_char in replace_chars.items():
            example_dir = example_dir.replace(char, replace_char)

        return f"{example_dir}/trial_{trial_num}.pt"

    @staticmethod
    def make_save_dict(prediction, sample_uuid, trial_num):
        sample_uuid = SampleUuidUtils.demote(sample_uuid)
        return sample_uuid, {
            "trial_idx": trial_num,
            "prediction": prediction,
        }

    @staticmethod
    def cache_prediction(
        base_path: Path,
        example_uuid: Any,
        trial_idx: int,
        prediction: dict,
        pickle_module=None,
    ) -> None:
        """
        Cache a prediction for a given example and trial index.

        Args:
            example_uuid: Unique identifier for the example
            trial_idx: Index of the trial
            prediction: Dictionary containing prediction data

        Returns:
            str: path where prediction was saved

        Example:
            example_uuid = (1, 2, 3)
            trial_idx = 1
            prediction = {"prediction": {"volume": 1.0}}

            Will save to:
            prediction_folder/
                1___2___3/
                    trial_1.pt
        """
        fname = WithCachedPredictionDataset.make_safe_filename(example_uuid, trial_idx)
        fpath = base_path / fname
        fpath.parent.mkdir(parents=True, exist_ok=True)
        save_dict = WithCachedPredictionDataset.make_save_dict(
            prediction, example_uuid, trial_idx
        )
        if pickle_module is None:
            pickle_module = pickle
        torch.save(save_dict, fpath, pickle_module=pickle_module)
        return fpath

    def __len__(self) -> int:
        return len(self.prediction_metadata)

    def __getitem__(self, idx: Union[int, tuple, str, dict]) -> tuple:
        if isinstance(idx, int):
            row = self.prediction_metadata.iloc[idx]
        elif isinstance(idx, tuple):
            # Look up by UUID
            row = self.prediction_metadata[
                self.prediction_metadata["example_uuid"].apply(
                    lambda x: tuple(x) == tuple(idx)
                )
            ].iloc[0]
        elif isinstance(idx, dict):
            example_uuid = idx["example_uuid"]
            trial_id = idx["trial"]
            rows = self.prediction_metadata[
                self.prediction_metadata["example_uuid"].apply(
                    lambda x: type(example_uuid)(x) == example_uuid
                )
            ]
            row = rows[
                rows["fname"].apply(
                    lambda x: x.split("/")[-1] == f"trial_{trial_id}.pt"
                )
            ].iloc[0]
        else:
            row = self.prediction_metadata[
                self.prediction_metadata["example_uuid"].apply(
                    lambda x: type(idx)(x) == type(idx)(idx)
                )
            ].iloc[0]
        # else:
        #     row = self.prediction_metadata.iloc[idx]

        example_uuid, prediction = self._load_prediction(row["fname"])

        # Get original example using the UUID
        if self.sample_dataset:
            _, example = self.sample_dataset[example_uuid]
        else:
            example = None

        return example_uuid, {"sample": example, **prediction}
