import unittest
from torch.utils.data.dataset import IterableDataset, Dataset

from lidra.data.dataset.transformed import Transformed
from lidra.test.util import run_unittest


class MockDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self._data = list(range(10))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        return self._data[index]


class MockIterableDataset(IterableDataset):
    def __iter__(self):
        yield from range(10)


# TODO(Pierre) : add more tests
class UnitTests(unittest.TestCase):
    def _test_dataset(self, dataset):
        dataset_no_transform = Transformed(dataset, transform=None)
        self.assertIs(dataset, dataset_no_transform)
        for i, j in zip(range(10), dataset_no_transform):
            self.assertEqual(i, j)

        dataset_transform = Transformed(dataset, transform=lambda x: x + 10)
        self.assertIsNot(dataset, dataset_transform)
        for i, j in zip(range(10), dataset_transform):
            self.assertEqual(i + 10, j)

    def test_transformed_dataset(self):
        fixed_dataset = MockDataset()
        iterable_dataset = MockIterableDataset()

        self._test_dataset(fixed_dataset)
        self._test_dataset(iterable_dataset)


if __name__ == "__main__":
    run_unittest(UnitTests)
