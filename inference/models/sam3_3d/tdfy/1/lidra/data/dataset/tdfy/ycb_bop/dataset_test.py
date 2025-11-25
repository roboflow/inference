import os
import unittest

from lidra.data.dataset.tdfy.ycb_bop.anything_dataset import AnythingDataset
from lidra.data.dataset.tdfy.trellis.pose_loader import load_trellis_pose_w_scale
from lidra.data.dataset.tdfy.trellis.dataset import PerSubsetDataset, PreProcessor
from lidra.data.dataset.tdfy.img_processing import pad_to_square_centered
from lidra.test.util import run_unittest, run_only_if_path_exists
from torchvision.transforms import Compose, Resize


class UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up datasets once for all tests."""
        cls.base_dir = "/checkpoint/3dfy/shared/datasets/trellis500k/YCBVideo"
        if not os.path.exists(cls.base_dir):
            return
        cls.per_subset_dataset = PerSubsetDataset(
            path=cls.base_dir,
            split="val",
            preprocessor=cls.get_preprocessor(),
            metadata_fname="metadata.csv",
            pose_loader=load_trellis_pose_w_scale,
        )
        cls.dataset = AnythingDataset(
            eval_id_json=os.path.join(cls.base_dir, "eval_random3.json"),
            latent_loader_dataset=cls.per_subset_dataset,
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

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k/YCBVideo")
    def test_basic_indexing(self):
        """Test basic integer indexing and expected item keys."""
        _, item = self.dataset[0]
        _, item_last = self.dataset[len(self.dataset) - 1]

        expected_keys = {
            "image",
            "mask",
            "mean",
            "logvar",
            "pointmap",
            "pointmap_scale",
            "instance_quaternion_l2c",
            "instance_position_l2c",
            "instance_scale_l2c",
            "camera_K",
        }
        self.assertTrue(all(k in item for k in expected_keys))
        self.assertTrue(all(k in item_last for k in expected_keys))


if __name__ == "__main__":
    run_unittest(UnitTests)
