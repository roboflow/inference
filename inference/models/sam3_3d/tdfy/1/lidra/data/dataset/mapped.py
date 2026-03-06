from torch.utils.data import Dataset, IterableDataset
from datasets import (
    Dataset as HFDataset,
    IterableDataset as HFIterableDataset,
)


def mapped(dataset, function, **kwargs):
    if isinstance(dataset, Dataset) or isinstance(dataset, HFDataset):
        return MappedFixedSize(dataset, function, **kwargs)
    elif isinstance(dataset, IterableDataset) or isinstance(dataset, HFIterableDataset):
        return MappedIterable(dataset, function, **kwargs)
    raise RuntimeError(
        "dataset class should be an instace of `torch.utils.data.Dataset` or `torch.utils.data.IterableDataset`"
    )


class MappedFixedSize(Dataset):
    def __init__(self, dataset, function, **kwargs) -> None:
        self.dataset = dataset
        self.function = function
        self.kwargs = kwargs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.function(item, **self.kwargs)


class MappedIterable:
    def __init__(self, dataset, function, **kwargs) -> None:
        self.dataset = dataset
        self.function = function
        self.kwargs = kwargs

    def __iter__(self):
        for item in iter(self.dataset):
            yield self.function(item, **self.kwarg)
