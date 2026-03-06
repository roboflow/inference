import os
import torch
import unittest

from lidra.test.util import run_unittest, temporary_directory
from lidra.data.dataset.pth import Pth


# TODO(Pierre) : add more tests
class UnitTests(unittest.TestCase):
    ITEMS = [
        "b/item1.pth",
        "b/item0.pth",
        "a/item0.pth",
        "a/item1.pth",
        "item0.pth",
        "item1.pth",
        "a.pth",
    ]

    def _make_item(self, basepath, childpath):
        path = os.path.join(basepath, childpath)
        item = {"path": childpath, "tensor": torch.rand((2, 3, 4))}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(item, path)

    def test_pth_dataset(self):
        with temporary_directory() as dirpath:
            for item in UnitTests.ITEMS:
                self._make_item(dirpath, item)

            dataset = Pth(dirpath)
            items = sorted(UnitTests.ITEMS)

            self.assertEqual(len(dataset), len(items))
            for i, item in enumerate(dataset):
                self.assertEqual(item[0], items[i])
                self.assertEqual(item[1]["path"], items[i])


if __name__ == "__main__":
    run_unittest(UnitTests)
