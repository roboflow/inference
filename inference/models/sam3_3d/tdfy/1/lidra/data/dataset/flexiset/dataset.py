from torch.utils.data import Dataset
import pandas as pd
import os
from typing import Union, Sequence
from copy import copy

from lidra.data.dataset.flexiset.flexi.loader import Loader as FlexiLoader
from lidra.data.dataset.flexiset.flexi.set import Set as FlexiSet

# loader = extract information from information
# transform = change existing information into new information


class FlexiDataset(Dataset):
    DEFAULT_METADATA_FILENAME = "metadata.csv"

    def __init__(
        self,
        path,
        loaders: Sequence[FlexiLoader],
        outputs: Union[str, Sequence[str]],
        metadata_filter=lambda x: x,
        metadata_filename=DEFAULT_METADATA_FILENAME,
        metadata_extension=None,
        output_mapping=None,
        transforms=None,
    ):
        super().__init__()

        self._flexi_set = FlexiSet(
            inputs={
                "uid",
                "metadata",
                "path",
            },
            loaders=loaders,
            outputs=outputs,
            transforms=transforms,
        )

        self.root_path = path
        self.output_mapping = output_mapping if output_mapping is not None else {}

        # load metadata
        self.metadata = pd.read_csv(os.path.join(self.root_path, metadata_filename))
        self.metadata = metadata_filter(self.metadata)
        self.metadata_extension = (
            {} if metadata_extension is None else metadata_extension
        )

    @property
    def flexi_set(self):
        return self._flexi_set

    def get_from_uid(self, uid):
        index = self.metadata[self.metadata["sha256"] == uid].index
        if len(index) == 0:
            raise ValueError(f"no metadata found for uid '{uid}'")
        index = index[0]
        return self[index]

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index, **kwargs):
        metadata = self.metadata.iloc[index]
        metadata = metadata.to_dict()
        metadata.update(self.metadata_extension)
        uid = metadata["sha256"]

        # get item
        item = self._flexi_set(
            path=self.root_path,
            uid=uid,
            metadata=metadata,
            **kwargs,
        )

        # map output keys
        for output_key, input_key in copy(self.output_mapping).items():
            if input_key in item:
                item[output_key] = item.pop(input_key)
        return item

    def __len__(self):
        return len(self.metadata)
