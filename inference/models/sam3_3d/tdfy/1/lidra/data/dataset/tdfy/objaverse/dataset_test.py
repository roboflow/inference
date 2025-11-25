import unittest

from lidra.data.dataset.tdfy.objaverse.dataset import Dataset
from lidra.test.util import run_unittest, run_only_if_path_exists


# TODO(Pierre) : add more tests
class UnitTests(unittest.TestCase):
    def _test_dataset(self, **kwargs):
        dataset = Dataset(
            "/large_experiments/3dfy/datasets/preprocssed_objaverse/objaverse_rendering",
            **kwargs,
        )

        item = dataset[0]
        item = dataset[-1]

    @run_only_if_path_exists(
        "/large_experiments/3dfy/datasets/preprocssed_objaverse/objaverse_rendering"
    )
    def test_dataset(self):
        self._test_dataset(split="train", preprocessed=True)
        self._test_dataset(split="train", preprocessed=False)
        self._test_dataset(split="val", preprocessed=True)
        self._test_dataset(split="val", preprocessed=False)


if __name__ == "__main__":
    run_unittest(UnitTests)
