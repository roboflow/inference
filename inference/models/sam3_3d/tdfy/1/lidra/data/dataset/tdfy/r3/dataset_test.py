import os
import unittest
import torch

from lidra.data.dataset.tdfy.r3.dataset_deprecated import Dataset as Dataset_Deprecated
from lidra.data.dataset.tdfy.r3.dataset import R3Dataset, R3SampleID
from lidra.data.dataset.tdfy.r3.anything_dataset import AnythingDataset
from lidra.data.dataset.tdfy.trellis.dataset import PerSubsetDataset, PreProcessor
from lidra.test.util import run_unittest, run_only_if_path_exists
from lidra.data.dataset.tdfy.img_processing import pad_to_square_centered
from torchvision.transforms import Compose, Resize
from lidra.data.dataset.tdfy.trellis.pose_loader import R3
from lidra.data.dataset.tdfy.metadata_filter import (
    custom_metadata_filter,
    data_query_filter,
)


class UnitTests(unittest.TestCase):
    def _test_dataset(self, **kwargs):
        dataset = Dataset_Deprecated(
            **kwargs,
        )

        _ = dataset[0]
        _ = dataset[-1]

    @run_only_if_path_exists("/checkpoint/fujenchu/r3/data")
    @run_only_if_path_exists("/private/home/xingyuchen/xingyuchen/r3_mesh")
    def test_dataset(self):
        self._test_dataset(
            split="train",
            preload_gt_pts=False,
            add_context_to_bbox=0.1,
            frustum_visible=False,
            masked_img=False,
            n_gt_pts=20000,
            r3_split_json="/checkpoint/fujenchu/r3/data/data_split.json",
            r3_dir="/private/home/xingyuchen/xingyuchen/r3_mesh/",
        )
        self._test_dataset(
            split="val",
            preload_gt_pts=False,
            add_context_to_bbox=0.1,
            frustum_visible=False,
            masked_img=False,
            n_gt_pts=20000,
            r3_split_json="/checkpoint/fujenchu/r3/data/data_split.json",
            r3_dir="/private/home/xingyuchen/xingyuchen/r3_mesh/",
        )

    @classmethod
    def setUpClass(cls):
        """Set up datasets once for all tests."""
        cls.base_dir = (
            "/checkpoint/3dfy/shared/datasets/trellis500k/r3_goldenset_25_morphing"
        )
        if not os.path.exists(cls.base_dir):
            return
        cls.dataset = R3Dataset(
            data_dir=cls.base_dir,
            load_voxels=True,
        )
        cls.per_subset_dataset = PerSubsetDataset(
            path=cls.base_dir,
            split="val",
            metadata_fname="metadata_clean.csv",
            preprocessor=cls.get_preprocessor(),
            pose_loader=R3.load_pose,
            metadata_filter=custom_metadata_filter(
                [data_query_filter(query="pose==True")]
            ),
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
        "/checkpoint/3dfy/shared/datasets/trellis500k/r3_goldenset_25_morphing"
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
        "/checkpoint/3dfy/shared/datasets/trellis500k/r3_goldenset_25_morphing"
    )
    def test_tuple_indexing(self):
        """Test both raw tuple and namedtuple indexing methods."""
        # Test with raw tuple
        sample_id = (
            "34bb3a78440f17e9ee34043245f7c491ddffacf4687e059e68998ba50265413f_0",
        )
        uuid, item = self.dataset[sample_id]
        self.assertEqual(uuid.uuid, sample_id[0])
        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertIsInstance(item["mask"], torch.Tensor)

        # Test with namedtuple
        named_sample_id = R3SampleID(
            uuid="34bb3a78440f17e9ee34043245f7c491ddffacf4687e059e68998ba50265413f_0",
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
