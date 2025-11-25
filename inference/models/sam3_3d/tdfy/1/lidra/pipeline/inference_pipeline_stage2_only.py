import numpy as np
import trimesh
import torch
from loguru import logger
from typing import Union
from PIL import Image
from omegaconf import OmegaConf

from lidra.pipeline.inference_pipeline import InferencePipeline
from lidra.model.backbone.trellis.utils import postprocessing_utils
from lidra.pipeline import preprocess_utils
from lidra.pipeline.inference_utils import (
    voxelize_mesh,
    trimesh2o3d_mesh,
)


class InferenceStage2OnlyPipeline(InferencePipeline):
    def __init__(
        self,
        # layout model here
        slat_generator_config_path,
        slat_generator_ckpt_path,
        slat_decoder_gs_config_path,
        slat_decoder_gs_ckpt_path,
        slat_decoder_mesh_config_path,
        slat_decoder_mesh_ckpt_path,
        dtype="bfloat16",
        device="cuda",
        version="v0",
        slat_preprocessor=preprocess_utils.get_default_preprocessor(),
        slat_condition_input_mapping=["image"],
        workspace_dir="",
        use_pretrained_slat=True,
        decode_formats=["mesh", "gaussian"],
    ):
        self.device = torch.device(device)
        with self.device:
            self.version = version
            self.slat_preprocessor = slat_preprocessor
            self.slat_condition_input_mapping = slat_condition_input_mapping
            self.workspace_dir = workspace_dir
            self.use_pretrained_slat = use_pretrained_slat
            self.decode_formats = decode_formats

            if dtype == "bfloat16":
                self.dtype = torch.bfloat16
            elif dtype == "float16":
                self.dtype = torch.float16
            elif dtype == "float32":
                self.dtype = torch.float32
            else:
                raise NotImplementedError
            logger.info("Loading model weights...")

            slat_generator = self.init_slat_generator(
                slat_generator_config_path, slat_generator_ckpt_path
            )

            slat_decoder_gs = self.init_slat_decoder_gs(
                slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
            )
            slat_decoder_mesh = self.init_slat_decoder_mesh(
                slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
            )

            # Load conditioner embedder so that we only load it once
            slat_condition_embedder = self.init_slat_condition_embedder(
                slat_generator_config_path, slat_generator_ckpt_path
            )
            self.condition_embedders = {
                "slat_condition_embedder": slat_condition_embedder,
            }
            self.override_slat_generator_cfg_config(slat_generator)
            self.models = torch.nn.ModuleDict(
                {
                    "slat_generator": slat_generator,
                    "slat_decoder_gs": slat_decoder_gs,
                    "slat_decoder_mesh": slat_decoder_mesh,
                }
            )
            logger.info("Loading model weights completed!")

    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mesh_path: str = None,
        coords: torch.Tensor = None,
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed=42,
        with_mesh_postprocess=True,
        with_texture_baking=True,
    ) -> dict:
        """
        Parameters:
        - image (Image): The input image to be processed.
        - mesh_path (str): The location to a mesh file. Optional. Can pass in coords instead.
        - coords (Tensor): The occupied coordinates from Stage 1 model. Optinal. Can pass in mesh_path instead.
        - seed (int, optional): The random seed for reproducibility. Default is 42.
        - with_mesh_postprocess (bool, optional): If True, performs mesh post-processing. Default is True.
        - with_texture_baking (bool, optional): If True, applies texture baking to the 3D model. Default is True.
        Returns:
        - dict: A dictionary containing the GLB file and additional data from the sparse structure sampling.
        """
        image = self.merge_image_and_mask(image, mask)

        assert (
            mesh_path is not None or coords is not None
        ), "At least need mesh_path or coords for coarse geometry"
        assert (
            mesh_path is None or coords is None
        ), "Cannot consume both coords and mesh_path"

        if mesh_path is not None:
            trimesh_mesh = trimesh.load_mesh(mesh_path, force="mesh")
            if isinstance(trimesh_mesh, trimesh.Scene):
                trimesh_mesh = trimesh_mesh.dump(concatenate=True)
            mesh = trimesh2o3d_mesh(trimesh_mesh)
            ss, _, _ = voxelize_mesh(mesh)

            # switch coordinate to match gs/ mesh
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 1]].int()

        with self.device:
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            torch.manual_seed(seed)

            coords = coords.to(self.device)
            slat = self.sample_slat(slat_input_dict, coords)
            outputs = self.decode_slat(slat, self.decode_formats)
            logger.info(
                f"Postprocessing mesh with option with_mesh_postprocess {with_mesh_postprocess}, with_texture_baking {with_texture_baking}..."
            )
            if "mesh" in outputs:
                glb = postprocessing_utils.to_glb(
                    outputs["gaussian"][0],
                    outputs["mesh"][0],
                    # Optional parameters
                    simplify=0.95,  # Ratio of triangles to remove in the simplification process
                    texture_size=1024,  # Size of the texture used for the GLB
                    verbose=False,
                    with_mesh_postprocess=with_mesh_postprocess,
                    with_texture_baking=with_texture_baking,
                )
            else:
                glb = None
            logger.info("Finished!")

        return {"glb": glb, "gs": outputs["gaussian"][0], **outputs}
