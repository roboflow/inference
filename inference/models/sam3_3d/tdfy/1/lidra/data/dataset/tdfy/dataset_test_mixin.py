import unittest
import os

import torch
import pytorch_lightning as pl
import pytorch_lightning.utilities.seed as pl_seed

from lidra.data.dataset.tdfy.trellis.dataset import TrellisDataset
from lidra.test.util import (
    run_unittest,
    run_only_if_path_exists,
    OverwriteTensorEquality,
)
from lidra.data.dataset.tdfy.img_and_mask_transforms import IMAGENET_UNNORMALIZATION
from lidra.utils.notebook.hydra import LidraConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from lidra.data.dataset.return_type import (
    extract_data,
    extract_sample_uuid,
    SampleUuidUtils,
)


class AnythingDatasetMixin(unittest.TestCase):

    def _load_dataset(
        self, config_path, datasets_path="/checkpoint/3dfy/shared/datasets"
    ):
        config = LidraConf.load_config(
            config_path,
            overrides=[
                f'+cluster.path.datasets="{datasets_path}"',
            ],
        )
        return instantiate(config)

    def _test_loads_first_and_last_samples(self, dataset):
        first_item = dataset[0]
        last_item = dataset[-1]
        return first_item, last_item

    def _test_uuid_indexing(self, dataset, sample_uuid):
        item2 = dataset[sample_uuid]
        uuid2, data2 = item2
        self.assertEqual(sample_uuid, uuid2)

    def _test_sample_basics(self, sample):
        self._test_sample_structure(sample)
        sample_data = extract_data(sample)
        self._test_image_normalization(sample_data)

    def _test_sample_structure(self, sample):
        assert len(sample) == 2
        uuid = extract_sample_uuid(sample)
        data = extract_data(sample)

        # Test that the uuid can be demoted to a primitive type
        SampleUuidUtils.demote(uuid)
        self.assertIsInstance(data, dict)

    def _test_image_normalization(self, sample_data):
        image = sample_data["rgb_image"]
        self.assertLessEqual(image.max().item(), 1 + 1e-6)
        self.assertGreaterEqual(image.min().item(), 0)

    def _test_sample_instance_pose(self, sample_data):
        """Test that the sample data contains the keys and shapes for instance pose in L2C frame"""
        self.assertIn("instance_quaternion_l2c", sample_data)
        self.assertIsInstance(sample_data["instance_quaternion_l2c"], torch.Tensor)
        self.assertEqual(sample_data["instance_quaternion_l2c"].shape[-1], 4)
        # Eventually we will load multiple instances per sample
        self.assertEqual(sample_data["instance_quaternion_l2c"].shape[0], 1)

        self.assertIn("instance_position_l2c", sample_data)
        self.assertIsInstance(sample_data["instance_position_l2c"], torch.Tensor)
        self.assertEqual(sample_data["instance_position_l2c"].shape[-1], 3)
        self.assertEqual(sample_data["instance_position_l2c"].shape[0], 1)

        self.assertIn("instance_scale_l2c", sample_data)
        self.assertIsInstance(sample_data["instance_scale_l2c"], torch.Tensor)

    def _test_sample_mesh(self, sample_data):
        self.assertIn("mesh_vertices", sample_data)
        self.assertIn("mesh_faces", sample_data)

    def _test_sample_pointmap(self, sample_data):
        """Test that the sample data contains the keys and shapes for pointmap"""
        self.assertIn("pointmap", sample_data)
        self.assertIsInstance(sample_data["pointmap"], torch.Tensor)

        self.assertIn("pointmap_colors", sample_data)
        self.assertIsInstance(sample_data["pointmap_colors"], torch.Tensor)

        self.assertEqual(
            sample_data["pointmap"].shape, sample_data["pointmap_colors"].shape
        )
