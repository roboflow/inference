from torch.utils.data.dataset import IterableDataset, Dataset


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index):
        return self._transform(self._dataset[index])

    def __len__(self):
        return len(self._dataset)


class TranformedIterableDataset(IterableDataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self._dataset = dataset
        self._transform = transform

    def __iter__(self):
        for sample in self._dataset:
            yield self._transform(sample)


class Transformed:
    def __new__(cls, dataset, transform=None):
        if transform is None:
            return dataset
        if isinstance(dataset, IterableDataset):
            return TranformedIterableDataset(dataset, transform)
        elif isinstance(dataset, Dataset):
            return TransformedDataset(dataset, transform)
        else:
            raise RuntimeError(
                f"dataset should inherit `Dataset` or `IterableDataset` class"
            )
