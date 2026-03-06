import os
import unittest
import torch

from lidra.data.dataset.tdfy.objaverseV1_2024.anything_dataset import AnythingDataset
from lidra.data.dataset.tdfy.objaverseV1_2024.dataset import (
    ObjaverseV1_2024Dataset,
    ObjaverseV1_2024SampleID,
)
from lidra.data.dataset.tdfy.trellis.pose_loader import load_pose_objaversev1old
from lidra.data.dataset.tdfy.trellis.dataset import PerSubsetDataset, PreProcessor
from lidra.test.util import run_unittest, run_only_if_path_exists
from lidra.data.dataset.tdfy.img_processing import pad_to_square_centered
from torchvision.transforms import Compose, Resize


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up datasets once for all tests."""
        cls.base_dir = "/checkpoint/3dfy/shared/datasets/trellis500k/ObjaverseV1_all"
        renders_cond_path = os.path.join(cls.base_dir, "renders_cond")
        if not os.path.exists(cls.base_dir) or not os.path.exists(renders_cond_path):
            raise unittest.SkipTest(
                f"Skipping tests: {cls.base_dir} or its subdirectories do not exist"
            )
        cls.dataset = ObjaverseV1_2024Dataset(data_dir=cls.base_dir, load_voxels=True)
        cls.per_subset_dataset = PerSubsetDataset(
            path=cls.base_dir,
            split="val",
            preprocessor=cls.get_preprocessor(),
            metadata_fname="metadata_eval.csv",
            pose_loader=load_pose_objaversev1old,
        )
        cls.dataset = AnythingDataset(
            dataset=cls.dataset, latent_loader_dataset=cls.per_subset_dataset
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

    @run_only_if_path_exists(
        "/checkpoint/3dfy/shared/datasets/trellis500k/ObjaverseV1_all"
    )
    def test_basic_indexing(self):
        """Test basic integer indexing and expected item keys."""
        _, item = self.dataset[0]
        _, item_last = self.dataset[len(self.dataset) - 1]

        expected_keys = {
            "image",
            "mask",
            "voxels",
            "mean",
            "logvar",
        }
        self.assertTrue(all(k in item for k in expected_keys))
        self.assertTrue(all(k in item_last for k in expected_keys))

    @run_only_if_path_exists(
        "/checkpoint/3dfy/shared/datasets/trellis500k/ObjaverseV1_all"
    )
    def test_tuple_indexing(self):
        """Test both raw tuple and namedtuple indexing methods."""
        # Test with raw tuple
        sample_id = (
            "00dfd7f2111a4597842d28edb09d9bf2",
            "0000",
        )
        uuid, item = self.dataset[sample_id]
        self.assertEqual(uuid.uuid, sample_id[0])
        self.assertEqual(uuid.frame_id, sample_id[1])
        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertIsInstance(item["mask"], torch.Tensor)

        # Test with namedtuple
        named_sample_id = ObjaverseV1_2024SampleID(
            uuid="00dfd7f2111a4597842d28edb09d9bf2",
            frame_id="0000",
        )
        uuid, item = self.dataset[named_sample_id]
        self.assertEqual(uuid, named_sample_id)
        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertIsInstance(item["mask"], torch.Tensor)
        self.assertIsInstance(item["voxels"], torch.Tensor)

        # Verify both methods return the same data
        _, item_tuple = self.dataset[sample_id]
        _, item_named = self.dataset[named_sample_id]
        self.assertTrue(torch.equal(item_tuple["image"], item_named["image"]))
        self.assertTrue(torch.equal(item_tuple["mask"], item_named["mask"]))
        self.assertTrue(torch.equal(item_tuple["voxels"], item_named["voxels"]))


if __name__ == "__main__":
    run_unittest(UnitTests)
