import os
import unittest
import torch

from lidra.data.dataset.return_type import (
    extract_data,
    extract_sample_uuid,
)
from lidra.data.dataset.tdfy.aria_digital_twin.anything_dataset import (
    AnythingDataset as ADTAnythingDataset,
    ADTAnythingSampleID,
)
from lidra.data.dataset.tdfy.aria_digital_twin.dataset import ADTDataset
from lidra.data.dataset.tdfy.artists_3d.dataset_test import identify_base_dir
from lidra.data.dataset.tdfy.dataset_test_mixin import AnythingDatasetMixin
from lidra.data.dataset.tdfy.img_processing import pad_to_square_centered
from lidra.data.dataset.tdfy.trellis.dataset import PerSubsetDataset, PreProcessor

from lidra.test.util import run_unittest
from torchvision.transforms import Compose, Resize


class UnitTests(AnythingDatasetMixin):
    @classmethod
    def setUpClass(cls):
        """Set up datasets once for all tests."""
        try:
            base_dir = identify_base_dir()
        except ValueError:
            raise unittest.SkipTest("could not identify base dir")
        cls.adt_base_dir = os.path.join(base_dir, "shared/datasets/aria_digital_twin")

        cls.adt_dataset = ADTDataset(
            path=cls.adt_base_dir, use_synth_img=False, reload_cache=False
        )
        cls.per_subset_dataset = PerSubsetDataset(
            path=cls.adt_base_dir, split="val", preprocessor=cls.get_preprocessor()
        )
        cls.dataset = ADTAnythingDataset(
            dataset=cls.adt_dataset, latent_loader_dataset=cls.per_subset_dataset
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

    def test_tuple_indexing(self):
        """Test both raw tuple and namedtuple indexing methods."""
        # Test with raw tuple
        sample_id = (
            "Apartment_release_clean_seq134_M1292",
            "14594432521025",
            "ApartmentEnv",
        )
        uuid, item = self.dataset[sample_id]
        self.assertEqual(uuid.seq_id, sample_id[0])
        self.assertEqual(uuid.frame_id, sample_id[1])
        self.assertEqual(uuid.instance_id, sample_id[2])
        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertIsInstance(item["mask"], torch.Tensor)

        # Test with namedtuple
        named_sample_id = ADTAnythingSampleID(
            seq_id="Apartment_release_clean_seq134_M1292",
            frame_id="14594432521025",
            instance_id="ApartmentEnv",
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

    def test_pose_loading(self):
        """Test pose loading"""
        sample = self.dataset[123]

        sample_uuid = extract_sample_uuid(sample)
        self._test_uuid_indexing(self.dataset, sample_uuid)

        sample_data = extract_data(sample)
        self._test_sample_basics(sample)
        self._test_sample_instance_pose(sample_data)


if __name__ == "__main__":
    run_unittest(UnitTests)
