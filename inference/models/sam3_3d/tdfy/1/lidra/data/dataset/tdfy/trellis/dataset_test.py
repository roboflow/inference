import unittest
import os
from loguru import logger
import torch
import lightning.pytorch as pl
import lightning.pytorch.utilities.seed as pl_seed

from lidra.data.dataset.tdfy.trellis.dataset import TrellisDataset
from lidra.test.util import (
    run_unittest,
    run_only_if_path_exists,
    OverwriteTensorEquality,
)
from lidra.data.dataset.tdfy.img_and_mask_transforms import IMAGENET_UNNORMALIZATION
from lidra.utils.notebook.hydra import LidraConf
from lidra.data.dataset.tdfy.kubric.anything_dataset import AnythingDataset
from lidra.data.dataset.tdfy.dataset_test_mixin import AnythingDatasetMixin
from lidra.data.dataset.return_type import (
    extract_data,
    extract_sample_uuid,
)

from hydra.utils import instantiate
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointLights,
)
from pytorch3d.renderer.mesh.shader import HardDepthShader

from lidra.data.dataset.tdfy.flexi.datasets import trellis_from_old_dataset


class UnitTests(unittest.TestCase):
    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k")
    def test_fm_trellis500k_config(self):
        fm_trellis_500k_config_path = os.path.join(
            "run/train/tdfy/trellis/fm-trellis-trellis500k-randPad.yaml"
        )  # config used for the training
        fm_trellis_500k_config = LidraConf.load_config(
            fm_trellis_500k_config_path,
            as_root=False,
            overrides=[
                'cluster.path.datasets="/checkpoint/3dfy/shared/datasets"',
            ],
        )
        dataset = instantiate(fm_trellis_500k_config.loop.dataloaders.training.dataset)
        data = dataset[0][1]
        image = data["image"]
        mask = data["mask"]
        # image should be normalized
        unnormalized_img = IMAGENET_UNNORMALIZATION(image)
        self.assertLessEqual(unnormalized_img.max().item(), 1)
        self.assertGreaterEqual(unnormalized_img.min().item(), 0)
        self.assertLessEqual(mask.max().item(), 1)
        self.assertGreaterEqual(mask.min().item(), 0)

        self._test_flexi_version_dataset(dataset)

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k")
    def test_kubric_fullcond_config(self):
        kubric_fullcond_config_path = os.path.join(
            "run/train/tdfy/trellis/fm-kubric-fullcond.yaml"
        )  # config used for the training
        kubric_fullcond_config = LidraConf.load_config(
            kubric_fullcond_config_path,
            as_root=False,
        )
        dataset = instantiate(kubric_fullcond_config.loop.dataloaders.training.dataset)

        data = dataset[0][1]
        image = data["image"]
        mask = data["mask"]
        # image should not be normalized
        self.assertNotEqual(mask, None)
        self.assertLessEqual(image.max().item(), 1)
        self.assertGreaterEqual(image.min().item(), 0)
        self.assertLessEqual(mask.max().item(), 1)
        self.assertGreaterEqual(mask.min().item(), 0)

        self._test_flexi_version_dataset(dataset)

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k")
    def test_new_dataset(self):
        config_dir_path = "data/dataset/tdfy/trellis500k/subset_train/"
        subsets = os.listdir(os.path.join("etc", "lidra", config_dir_path))
        self.assertTrue(len(subsets) > 0, f"no subsets found in {config_dir_path}.")
        do_not_eval = {
            "base",
            "rp_a_25_morphing_pose",
            "ABO_test",
            "r3_25_trellis_unique_st3d",
            "r3_25_March_morphing",
            "r3_25_morphing",
        }
        for subset_filename in subsets:
            logger.debug(f"loading and testing subset: {subset_filename}")
            config_path = os.path.join(config_dir_path, subset_filename)
            subset_name = os.path.splitext(subset_filename)[0]
            if subset_name in do_not_eval:
                logger.warning(f"skipping subset {subset_name}.")
                continue

            config = LidraConf.load_config(
                config_path,
                overrides=[
                    "+cluster.path.datasets=/checkpoint/3dfy/shared/datasets",
                ],
            )
            dataset = instantiate(config)
            _ = dataset[0]
            _ = dataset[-1]

            self._test_flexi_version_dataset(dataset)

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k")
    def test_old_new_same_result(self):
        # no random augmentation
        new_subset_abo_config = (
            "data/dataset/tdfy/trellis500k/subset_train/ABO_test.yaml"
        )
        abo_config = LidraConf.load_config(
            new_subset_abo_config,
            overrides=[
                'data.dataset.tdfy.trellis500k.subset_train.path="/checkpoint/3dfy/shared/datasets/trellis500k/ABO"',
            ],
        )
        # to be compatible with the old dataset - by default, no padding
        abo_config["preprocessor"]["img_mask_joint_transform"][0][
            "padding_factor"
        ] = 0.0
        new_dataset = instantiate(abo_config)
        old_dataset = TrellisDataset(
            split="train",
            path="/checkpoint/3dfy/shared/datasets/trellis500k",
            subsets=["ABO"],
            latent_type="structure",
            tight_obj_boundary=True,
            random_pad=0.0,
            use_color_aug=False,
            remove_img_bg=True,
            box_size_factor=1.0,
            padding_factor=0.1,
        )
        with pl_seed.isolate_rng():
            pl.seed_everything(seed=0)
            data_new = new_dataset[0][1]
        with pl_seed.isolate_rng():
            pl.seed_everything(seed=0)
            data_old = old_dataset[0][1]
        for data_key in data_old.keys():
            # allow difference in processed mask and image due to rounding when cropping
            if data_key == "mask" or data_key == "image":
                diff = (
                    (((data_new[data_key] - data_old[data_key]) ** 2) > 0.04)
                    .max(dim=0)[0]
                    .sum()
                )
                self.assertLessEqual(diff / data_old[data_key].sum(), 0.05)
            elif type(data_new[data_key]) == torch.Tensor:
                with OverwriteTensorEquality(torch):
                    self.assertEqual(
                        data_new[data_key], data_old[data_key]
                    ), f"unequal results for {data_key}"
            else:
                self.assertEqual(
                    data_new[data_key], data_old[data_key]
                ), f"unequal results for {data_key}"

        self._test_flexi_version_dataset(new_dataset)

    def _test_flexi_version_dataset(self, dataset):
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            # if concat dataset, test all subsets one by one
            for ds in dataset.datasets:
                self._test_flexi_version_dataset(ds)
            return
        if isinstance(dataset, AnythingDataset):
            return  # skip the dataset below, pose format is different
            # wtf is this path ?! nothing simpler was possible ?
            self._test_flexi_version_dataset(dataset.dataset.latent_loader_dataset)
            return

        flexi_dataset = trellis_from_old_dataset(dataset)

        def _check_item_equal(item_flexi, item_old):
            image_path, item_old = item_old
            self.assertEqual(image_path, item_flexi["image_path"])
            with OverwriteTensorEquality(
                torch,
                check_device=True,
                check_dtype=True,
                check_shape=True,
            ):
                for key in item_old:
                    # Skip pointmap_scale and pointmap_shift as flexi dataset doesn't provide these
                    if key in ["pointmap_scale", "pointmap_shift"]:
                        continue
                    self.assertTrue(key in item_flexi)
                    self.assertEqual(item_flexi[key], item_old[key])

        with pl_seed.isolate_rng():
            pl.seed_everything(seed=0)
            data_new_0 = flexi_dataset[0]
            data_new_1 = flexi_dataset[-1]

        with pl_seed.isolate_rng():
            pl.seed_everything(seed=0)
            data_old_0 = dataset[0]
            data_old_1 = dataset[-1]

        _check_item_equal(data_new_0, data_old_0)
        _check_item_equal(data_new_1, data_old_1)

        self._check_projective_consistency(data_new_0)
        self._check_projective_consistency(data_new_1)

    def _check_and_extract_intrinsics_parameters(self, intrinsics_matrix):
        # to not in-place modify intrinsics
        intrinsics_matrix = intrinsics_matrix.clone()

        # extract focal length
        fx = intrinsics_matrix[0, 0].item()
        fy = intrinsics_matrix[1, 1].item()

        # extract principal point
        px = intrinsics_matrix[0, 2].item()
        py = intrinsics_matrix[1, 2].item()

        if fx != fy:
            logger.warning(f"focal length is not isotropic (fx={fx}, fy={fy})")

        # erase focal parameters for easier comparison below
        intrinsics_matrix[0, 0] = intrinsics_matrix[1, 1] = 0
        intrinsics_matrix[0, 2] = intrinsics_matrix[1, 2] = 0

        intrinsics_with_no_params = torch.zeros((3, 3))
        intrinsics_with_no_params[2, 2] = 1

        self.assertTrue(
            torch.allclose(
                intrinsics_matrix,
                intrinsics_with_no_params,
                atol=1e-4,
            )
        )

        return fx, fy, px, py

    def _check_mask_iou(self, image_0, image_1, threshold):
        assert image_0.dtype == torch.bool
        assert image_1.dtype == torch.bool

        intersection = (image_0 & image_1).sum().item()
        union = (image_0 | image_1).sum().item()
        iou = intersection / union if union > 0 else 0.0

        self.assertGreaterEqual(iou, threshold)

    def _check_mask_accuracy(self, image_0, image_1, threshold):
        assert image_0.dtype == torch.bool
        assert image_1.dtype == torch.bool

        accuracy = (image_0 == image_1).float().mean().item()
        self.assertGreaterEqual(accuracy, threshold)

    @staticmethod
    def _render_depth(R, T, fx, fy, px, py, image_size, mesh):
        camera = PerspectiveCameras(
            focal_length=((fx, fy),),
            principal_point=((px, py),),
            R=R,
            T=T,
        )
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            # use depth because other shaders need texture
            shader=HardDepthShader(device="cpu", cameras=camera),
        )
        return renderer(mesh)

    def _check_projective_consistency(self, item):
        # extract necessary data for projective consistency test
        try:
            instance_quaternion = item["instance_quaternion_l2c"]
            instance_position = item["instance_position_l2c"]
            instance_scale = item["instance_scale_l2c"]
            camera_K = item["camera_K"]
            image = item["rgba_image"]
            trellis_mesh = item["trellis_mesh"]
        except KeyError:
            logger.warning(
                "missing keys in data item, skipping projective consistency test ..."
            )
            return

        # get intrinsics
        fx, fy, px, py = self._check_and_extract_intrinsics_parameters(camera_K)

        # get isotropic scale
        scale = instance_scale
        if instance_scale.numel() != 1:
            scale = instance_scale[..., 0]
            assert scale.numel() == 1, "scale is batched for some reason ?!"
            assert torch.allclose(instance_scale, scale), "non isotropic scaling found"
        scale = scale.item()

        rendered_depth = UnitTests._render_depth(
            quaternion_to_matrix(instance_quaternion) / scale,
            instance_position / scale,
            fx,
            fy,
            px,
            py,
            image_size=image.shape[1:3],
            mesh=trellis_mesh,
        )

        mask_0 = rendered_depth.squeeze() < 100.0
        mask_1 = image[3].squeeze() > 0

        # COMMENT(Pierre) depth-produced masks can be different from alpha-produced masks.
        # this occurs when the texture has some transparent pixels, while the depth shader does not use texture
        # thus the following is necessary in order to eliminate transparent pixels from the depth mask
        # the following checks should still be relevant
        mask_0 = mask_0 & mask_1

        self._check_mask_iou(mask_0, mask_1, threshold=0.90)
        self._check_mask_accuracy(mask_0, mask_1, threshold=0.99)


class R3Unittests(AnythingDatasetMixin):

    def _test_per_sample_subset_sample_indexing(self, dataset, sample_uuid):
        # Test alternative ways to index this sample
        self.assertIsInstance(
            sample_uuid, str
        )  # Dataset right now returns path to image
        fpath = sample_uuid
        sample_id = dataset.image_fpath_to_sample_id(fpath)
        reloaded_sample = dataset[sample_id]
        self.assertEqual(fpath, extract_sample_uuid(reloaded_sample))

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k")
    def test_r3_shape(self):
        dataset = self._load_dataset(
            "data/dataset/tdfy/trellis500k/subset_train/r3_25_morphing.yaml"
        )
        first_sample, _ = self._test_loads_first_and_last_samples(dataset)

    @run_only_if_path_exists("/checkpoint/3dfy/shared/datasets/trellis500k")
    def test_rp_with_pose(self):
        dataset = self._load_dataset(
            "data/dataset/tdfy/trellis500k/subset_train/rp_a_25_morphing_pose.yaml"
        )
        first_sample, _ = self._test_loads_first_and_last_samples(dataset)
        sample_uuid = extract_sample_uuid(first_sample)
        self._test_uuid_indexing(dataset, sample_uuid)
        self._test_per_sample_subset_sample_indexing(dataset, sample_uuid)

        # sample_data
        sample_data = extract_data(first_sample)
        self._test_sample_basics(first_sample)
        self._test_sample_instance_pose(sample_data)


if __name__ == "__main__":
    run_unittest(UnitTests)
