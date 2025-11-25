import unittest
import torch

from lidra.data.dataset.tdfy.kubric.dataset import KubricDataset
from lidra.data.dataset.tdfy.kubric.multi_obj_frame_dataset import (
    KubricMultiObjInFrameDataset,
)
from lidra.test.util import run_unittest, run_only_if_path_exists
from lidra.data.dataset.tdfy.dataset_test_mixin import AnythingDatasetMixin

from lidra.utils.notebook.hydra import LidraConf
from lidra.data.dataset.tdfy.dataset_test_mixin import AnythingDatasetMixin
from lidra.data.dataset.return_type import (
    extract_data,
    extract_sample_uuid,
)


class KubricAnythingTests(AnythingDatasetMixin):

    def _test_basic_functionality(self, config_path):
        dataset = self._load_dataset(config_path)
        first_sample, _ = self._test_loads_first_and_last_samples(dataset)
        sample_uuid = extract_sample_uuid(first_sample)
        self._test_uuid_indexing(dataset, sample_uuid)

        sample_data = extract_data(first_sample)
        self._test_sample_basics(first_sample)
        self._test_sample_instance_pose(sample_data)

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k")
    def test_training_dataset(self):
        self._test_basic_functionality(
            "data/dataset/tdfy/kubric/anything_training.yaml"
        )

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k")
    def test_validation_dataset(self):
        self._test_basic_functionality(
            "data/dataset/tdfy/kubric/anything_validation.yaml"
        )


class KubricBaseTests(unittest.TestCase):
    def _test_dataset(self, **kwargs):
        # Test base dataset
        dataset = KubricDataset(**kwargs)

        # Test length
        self.assertGreater(len(dataset), 0, "Dataset should not be empty")

        # Test getting first item
        first_item = dataset[0][1]

        # Check required keys are present
        required_keys = {
            "video",
            "instances",
            "depth",
            "instance_info",
            "cameras",
            "video_uuid",
            "frame_indices",
        }
        missing_keys = [k for k in required_keys if k not in first_item]
        self.assertTrue(
            all(k in first_item for k in required_keys),
            f"Missing keys in dataset item: {missing_keys}",
        )

        # Test shapes are correct
        self.assertEqual(first_item["video"].dim(), 4)  # B, C, H, W
        self.assertEqual(first_item["instances"].dim(), 3)  # B, H, W
        self.assertEqual(first_item["depth"].dim(), 3)  # B, H, W

        # Test multi-object dataset wrapper
        multi_obj_dataset = KubricMultiObjInFrameDataset(
            dataset=dataset,
            preload_gt_pts=False,
            n_points_per_instance=10000,
            keep_k_instances=2,
        )

        multi_obj_item = multi_obj_dataset[0][1]

        # Check multi-object specific keys
        multi_obj_keys = {
            "rgb_image",
            "instance_masks",
            "instance_positions",
            "instance_quaternions_l2w",
        }
        missing_keys = [k for k in multi_obj_keys if k not in multi_obj_item]
        self.assertTrue(
            all(k in multi_obj_item for k in multi_obj_keys),
            f"Missing keys in multi-object dataset item: {missing_keys}",
        )

        # Test instance masks shape (should be N instances after keep_k)
        self.assertEqual(multi_obj_item["instance_masks"].shape[0], 2)

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/kubric/tdfy_v0")
    def _test_dataset_no_randomness(self, **kwargs):
        dataset = KubricDataset(**kwargs)
        n_random_draw = 10
        sample_uuid = None

        for _ in range(n_random_draw):
            sid = dataset[0][0]
            if sample_uuid is None:
                sample_uuid = sid
            else:
                self.assertEqual(sample_uuid, sid)

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/kubric/tdfy_v0")
    def test_dataset_movi_c(self):
        self._test_dataset(
            data_dir="/checkpoint/3dfy/shared/datasets/kubric/tdfy_v0",
            dataset_name="movi_c/256x256",
            split="train",
            frame_skip=1,
            sequence_length=1,
            skip_first_n_frames=12,
            load_meshes=False,
        )

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/kubric/tdfy_v0")
    def test_dataset_movi_e(self):
        self._test_dataset(
            data_dir="/checkpoint/3dfy/shared/datasets/kubric/tdfy_v0",
            dataset_name="movi_e/256x256",
            split="validation",
            frame_skip=1,
            sequence_length=1,
            skip_first_n_frames=12,
            load_meshes=False,
        )

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/kubric/tdfy_v0")
    def test_dataset_movi_e(self):
        self._test_dataset_no_randomness(
            data_dir="/checkpoint/3dfy/shared/datasets/kubric/tdfy_v0",
            dataset_name="movi_e/256x256",
            split="validation",
            frame_skip=1,
            sequence_length=1,
            skip_first_n_frames=12,
            load_meshes=False,
            random_frame_selection=False,
        )


if __name__ == "__main__":
    run_unittest(KubricAnythingTests)
    run_unittest(KubricBaseTests)
