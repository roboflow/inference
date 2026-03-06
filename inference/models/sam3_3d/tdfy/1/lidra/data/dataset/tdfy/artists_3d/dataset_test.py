import os
import unittest
import torch

from lidra.data.dataset.tdfy.artists_3d.anything_dataset import (
    AnythingDataset as Artist3DAnythingDataset,
    Artist3DAnythingSampleID,
)
from lidra.data.dataset.tdfy.trellis.pose_loader import identity_pose
from lidra.data.dataset.tdfy.artists_3d.dataset import Artist3DDataset
from lidra.data.dataset.tdfy.trellis.dataset import PerSubsetDataset, PreProcessor
from lidra.test.util import run_unittest, run_only_if_path_exists
from lidra.data.dataset.tdfy.img_processing import pad_to_square_centered
from torchvision.transforms import Compose, Resize


def identify_base_dir():
    """Different clusters have different checkpoint directories"""
    for path in {"/fsx-3dfy", "/checkpoint/3dfy"}:
        if os.path.exists(path):
            return path
    raise ValueError(f"Not on A100 or H100 clusters. Base dir not found.")


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up datasets once for all tests."""
        try:
            base_dir = identify_base_dir()
        except ValueError:
            raise unittest.SkipTest("could not identify base dir")

        cls.artist3d_base_dir = os.path.join(base_dir, "shared/datasets/artists_3d")
        if not os.path.exists(cls.artist3d_base_dir):
            return
        metadata_fname = "metadata_latest_052125.csv"
        cls.metadata_fname = os.path.join(cls.artist3d_base_dir, metadata_fname)
        cls.artist3d_dataset = Artist3DDataset(
            path=cls.artist3d_base_dir, metadata_fname=cls.metadata_fname
        )
        cls.per_subset_dataset = PerSubsetDataset(
            path=cls.artist3d_base_dir,
            split="val",
            metadata_fname=metadata_fname,
            preprocessor=cls.get_preprocessor(),
            pose_loader=identity_pose,
        )
        cls.dataset = Artist3DAnythingDataset(
            dataset=cls.artist3d_dataset, latent_loader_dataset=cls.per_subset_dataset
        )

    def get_preprocessor():
        preprocessor = PreProcessor()
        preprocessor.img_transform = Compose(
            transforms=[pad_to_square_centered, Resize(size=518, interpolation=0)]
        )
        preprocessor.mask_transform = Compose(
            transforms=[pad_to_square_centered, Resize(size=518, interpolation=0)]
        )
        preprocessor.img_mask_joint_transform = []

        return preprocessor

    def test_basic_indexing(self):
        """Test basic integer indexing and expected item keys."""
        _, item = self.dataset[0]
        _, item_last = self.dataset[len(self.dataset) - 1]

        expected_keys = {
            "image",
            "mask",
            "mean",
            "logvar",
        }
        self.assertTrue(all(k in item for k in expected_keys))
        self.assertTrue(all(k in item_last for k in expected_keys))

    def test_tuple_indexing(self):
        """Test both raw tuple and namedtuple indexing methods."""
        # Test with raw tuple
        sample_id = ("VTS_WD40", "sa_9519859_church_1", 2)
        uuid, item = self.dataset[sample_id]
        self.assertEqual(uuid.artist, sample_id[0])
        self.assertEqual(uuid.img_obj, sample_id[1])
        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertIsInstance(item["mask"], torch.Tensor)

        # Test with namedtuple
        named_sample_id = Artist3DAnythingSampleID(
            artist="VTS_WD40", img_obj="sa_9519859_church_1", version=2
        )
        uuid, item = self.dataset[named_sample_id]
        self.assertEqual(uuid, named_sample_id)
        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertIsInstance(item["mask"], torch.Tensor)

        # Verify both methods return the same data
        _, item_tuple = self.dataset[sample_id]
        _, item_named = self.dataset[named_sample_id]
        self.assertTrue(torch.equal(item_tuple["image"], item_named["image"]))
        self.assertTrue(torch.equal(item_tuple["mask"], item_named["mask"]))


if __name__ == "__main__":
    run_unittest(UnitTests)
