import unittest

from lidra.data.dataset.tdfy.cadestate.dataset import Dataset
from lidra.test.util import run_unittest, run_only_if_path_exists


class UnitTests(unittest.TestCase):
    def _test_dataset(self, **kwargs):
        dataset = Dataset(
            **kwargs,
        )

        item = dataset[0]
        item = dataset[-1]

    @run_only_if_path_exists("/checkpoint/fujenchu/cadestate/data")
    def test_dataset(self):
        self._test_dataset(
            split="train",
            preload_gt_pts=True,
            add_context_to_bbox=None,
            frustum_visible=True,
            masked_img=True,
            cad_estate_split_json="/checkpoint/fujenchu/cadestate/data/data_split.json",
        )
        self._test_dataset(
            split="val",
            preload_gt_pts=True,
            add_context_to_bbox=None,
            frustum_visible=True,
            masked_img=True,
            cad_estate_split_json="/checkpoint/fujenchu/cadestate/data/data_split.json",
        )


if __name__ == "__main__":
    run_unittest(UnitTests)
