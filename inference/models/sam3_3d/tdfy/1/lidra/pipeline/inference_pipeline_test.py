import os
import unittest

from hydra.utils import instantiate
from omegaconf import OmegaConf
import os
from PIL import Image
import torch
from lidra.test.util import (
    run_unittest,
    run_only_if_path_exists,
    run_only_if_cuda_is_available,
)


class UnitTests(unittest.TestCase):
    @run_only_if_cuda_is_available(default_device="cuda")
    @run_only_if_path_exists("/large_experiments/3dfy/")
    def test_3dfy_v2_v100(self):
        os.environ["CUDA_HOME"] = "/public/apps/cuda/12.1/"
        DEMO_VERSION = "demo_fc_3dfy_v2"
        pipeline_config = f"etc/lidra/run/inference/{DEMO_VERSION}.yaml"
        config = OmegaConf.load(pipeline_config)
        pipeline = instantiate(config["inference_pipeline"])

        IMAGE_PATH = "/checkpoint/haotang/dev/lidra/T.png"
        image = Image.open(IMAGE_PATH)
        pipeline.dtype = torch.float16
        output = pipeline.run(
            image, seed=500, with_mesh_postprocess=False, with_texture_baking=False
        )
        self.assertIn("glb", output)
        self.assertIn("translation", output)
        self.assertIn("scale", output)
        self.assertIn("rotation", output)
        del pipeline
        torch.cuda.empty_cache()

    @run_only_if_cuda_is_available(default_device="cuda")
    @run_only_if_path_exists("/large_experiments/3dfy/")
    def test_run_v0_4_3_v100(self):
        os.environ["CUDA_HOME"] = "/public/apps/cuda/12.1/"
        DEMO_VERSION = "demo_fc_v0.4.3"
        pipeline_config = f"etc/lidra/run/inference/{DEMO_VERSION}.yaml"
        config = OmegaConf.load(pipeline_config)
        pipeline = instantiate(config["inference_pipeline"])

        IMAGE_PATH = "/checkpoint/haotang/dev/lidra/T.png"
        image = Image.open(IMAGE_PATH)
        pipeline.dtype = torch.float16
        output = pipeline.run(
            image, seed=500, with_mesh_postprocess=False, with_texture_baking=False
        )
        self.assertIn("glb", output)
        del pipeline
        torch.cuda.empty_cache()

    @run_only_if_cuda_is_available(default_device="cuda")
    @run_only_if_path_exists("/large_experiments/3dfy/")
    def test_run_v0_2_v100(self):
        os.environ["CUDA_HOME"] = "/public/apps/cuda/12.1/"
        DEMO_VERSION = "demo_fc_v0.2"
        pipeline_config = f"etc/lidra/run/inference/{DEMO_VERSION}.yaml"
        config = OmegaConf.load(pipeline_config)
        pipeline = instantiate(config["inference_pipeline"])

        IMAGE_PATH = "/checkpoint/haotang/dev/lidra/T.png"
        image = Image.open(IMAGE_PATH)
        pipeline.dtype = torch.float16
        output = pipeline.run(
            image, seed=500, with_mesh_postprocess=False, with_texture_baking=False
        )
        self.assertIn("glb", output)
        del pipeline
        torch.cuda.empty_cache()

    @run_only_if_cuda_is_available(default_device="cuda")
    @run_only_if_path_exists("/large_experiments/3dfy/")
    def test_run_v1_0_v100(self):
        os.environ["CUDA_HOME"] = "/public/apps/cuda/12.1/"
        DEMO_VERSION = "demo_fc_v1.0"
        pipeline_config = f"etc/lidra/run/inference/{DEMO_VERSION}.yaml"
        config = OmegaConf.load(pipeline_config)
        pipeline = instantiate(config["inference_pipeline"])

        IMAGE_PATH = "/checkpoint/haotang/dev/lidra/T.png"
        image = Image.open(IMAGE_PATH)
        pipeline.dtype = torch.float16
        output = pipeline.run(
            image, seed=500, with_mesh_postprocess=False, with_texture_baking=False
        )
        self.assertIn("glb", output)
        del pipeline
        torch.cuda.empty_cache()

    @run_only_if_cuda_is_available(default_device="cuda")
    @run_only_if_path_exists("/large_experiments/3dfy/")
    def test_run_3dfy_v4_v100(self):
        os.environ["CUDA_HOME"] = "/public/apps/cuda/12.1/"
        DEMO_VERSION = "demo_fc_3dfy_v4"
        pipeline_config = f"etc/lidra/run/inference/{DEMO_VERSION}.yaml"
        config = OmegaConf.load(pipeline_config)
        pipeline = instantiate(config["inference_pipeline"])

        IMAGE_PATH = "/checkpoint/haotang/dev/lidra/T.png"
        image = Image.open(IMAGE_PATH)
        pipeline.dtype = torch.float16
        output = pipeline.run(
            image, seed=500, with_mesh_postprocess=False, with_texture_baking=False
        )
        self.assertIn("glb", output)
        del pipeline
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_unittest(UnitTests)
