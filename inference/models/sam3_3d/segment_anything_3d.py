import sys
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union
import weakref
from io import BytesIO
import os

import numpy as np
import torch

from inference.core.cache.model_artifacts import get_cache_dir

from inference.core import logger
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.env import DEVICE, MODEL_CACHE_DIR
from inference.core.exceptions import ModelArtefactError
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_roboflow_model_data,
    stream_url_to_cache,
)
from inference.core.utils.image_utils import load_image_rgb
from filelock import FileLock
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.entities.responses.sam3_3d import (
    Sam3_3D_Objects_Metadata,
    Sam3_3D_Objects_Response,
)
from omegaconf import OmegaConf
from hydra.utils import instantiate
from PIL import Image, ImageDraw

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(device_count))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from importlib.resources import files
import tdfy.sam3d_v1

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class Sam3_3D_ObjectsPipelineSingleton:
    """Singleton to cache the heavy 3D pipeline initialization."""

    _instances = weakref.WeakValueDictionary()
    _lock = Lock()

    def __new__(cls, config_key: str):
        with cls._lock:
            if config_key not in cls._instances:
                logger.info(
                    "Creating new SAM3_3D pipeline instance (this may take a while)..."
                )
                instance = super().__new__(cls)
                instance.config_key = config_key
                cls._instances[config_key] = instance
            else:
                logger.info("Using cached SAM3_3D pipeline instance")
            return cls._instances[config_key]


class SegmentAnything3_3D_Objects(RoboflowCoreModel):
    def __init__(
        self,
        *args,
        model_id: str = "sam3-3d-objects",
        torch_compile: bool = False,
        compile_res: int = 518,
        **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)

        self.cache_dir = Path(get_cache_dir(model_id=self.endpoint))

        tdfy_dir = files(tdfy.sam3d_v1)
        pipeline_config_path = tdfy_dir / "checkpoints_configs" / "pipeline.yaml"
        moge_checkpoint_path = self.cache_dir / "moge-vitl.pth"
        ss_generator_checkpoint_path = self.cache_dir / "ss_generator.ckpt"
        slat_generator_checkpoint_path = (
            self.cache_dir / "slat_generator.ckpt"
        )
        ss_decoder_checkpoint_path = self.cache_dir / "ss_decoder.ckpt"
        slat_decoder_checkpoint_path = self.cache_dir / "slat_decoder_gs.ckpt"
        slat_decodergs4_checkpoint_path = (
            self.cache_dir / "slat_decoder_gs_4.ckpt"
        )
        slat_decoder_mesh_checkpoint_path = (
            self.cache_dir / "slat_decoder_mesh.pt"
        )
        dinov2_ckpt_path = self.cache_dir / "dinov2_vitl14_reg4_pretrain.pth"

        config_key = f"{DEVICE}_{pipeline_config_path}"
        singleton = Sam3_3D_ObjectsPipelineSingleton(config_key)

        if not hasattr(singleton, "pipeline"):
            logger.info("Initializing SAM3_3D pipeline...")
            self.pipeline_config = OmegaConf.load(str(pipeline_config_path))
            self.pipeline_config["device"] = DEVICE
            self.pipeline_config["workspace_dir"] = str(tdfy_dir)
            self.pipeline_config["compile_model"] = torch_compile
            self.pipeline_config["compile_res"] = compile_res
            self.pipeline_config["depth_model"]["model"][
                "pretrained_model_name_or_path"
            ] = str(moge_checkpoint_path)
            self.pipeline_config["ss_generator_ckpt_path"] = str(
                ss_generator_checkpoint_path
            )
            self.pipeline_config["slat_generator_ckpt_path"] = str(
                slat_generator_checkpoint_path
            )
            self.pipeline_config["ss_decoder_ckpt_path"] = str(
                ss_decoder_checkpoint_path
            )
            self.pipeline_config["slat_decoder_gs_ckpt_path"] = str(
                slat_decoder_checkpoint_path
            )
            self.pipeline_config["slat_decoder_gs_4_ckpt_path"] = str(
                slat_decodergs4_checkpoint_path
            )
            self.pipeline_config["slat_decoder_mesh_ckpt_path"] = str(
                slat_decoder_mesh_checkpoint_path
            )
            self.pipeline_config["dinov2_ckpt_path"] = str(dinov2_ckpt_path)
            singleton.pipeline = instantiate(self.pipeline_config)
            logger.info("SAM3_3D pipeline initialization complete")

        # Reference the singleton's pipeline
        self.pipeline = singleton.pipeline
        self._state_lock = Lock()

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["environment.json"].
        """
        return ["moge-vitl.pth", "ss_generator.ckpt", "slat_generator.ckpt", "ss_decoder.ckpt", "slat_decoder_gs.ckpt", "slat_decoder_gs_4.ckpt", "slat_decoder_mesh.pt"]

    def download_model_from_roboflow_api(self) -> None:
        """Override parent method to use streaming downloads for large SAM3_3D model files."""
        lock_dir = MODEL_CACHE_DIR + "/_file_locks"
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = os.path.join(lock_dir, f"{os.path.basename(self.cache_dir)}.lock")
        try:
            lock = FileLock(lock_file, timeout = 120)
            with lock:
                api_data = get_roboflow_model_data(
                    api_key=self.api_key,
                    model_id="sam3-3d-weights-vc6vz/1",
                    endpoint_type=ModelEndpointType.ORT,
                    device_id=self.device_id,
                )["ort"]
                if "weights" not in api_data:
                    raise ModelArtefactError(
                        f"`weights` key not available in Roboflow API response while downloading model weights."
                    )
                for weights_url_key in api_data["weights"]:
                    weights_url = api_data["weights"][weights_url_key]
                    filename = weights_url.split("?")[0].split("/")[-1]
                    logger.info(f"Downloading SAM3_3D model file: {filename}")
                    stream_url_to_cache(
                        url=weights_url,
                        filename=filename,
                        model_id=self.endpoint,
                    )
                    logger.info(f"Successfully downloaded: {filename}")
        except Exception as e:
            logger.error(f"Error downloading SAM3_3D model artifacts: {e}")
            raise

    def infer_from_request(
        self, request: Sam3_3D_Objects_InferenceRequest
    ) -> Sam3_3D_Objects_Response:
        with self._state_lock:
            t1 = perf_counter()
            raw_result = self.create_3d(**request.dict())
            inference_time = perf_counter() - t1
            return convert_3d_objects_result_to_api_response(
                raw_result=raw_result,
                inference_time=inference_time,
            )

    def merge_image_and_mask(self, image, mask_input):
        image_np = load_image_rgb(image)
        image_height, image_width = image_np.shape[:2]
        mask = Image.new("L", (image_width, image_height), 0)
        draw = ImageDraw.Draw(mask)
        polygon = [
            (mask_input[i], mask_input[i + 1]) for i in range(0, len(mask_input), 2)
        ]
        draw.polygon(polygon, outline=255, fill=255)
        mask_np = np.array(mask)
        if mask_np.ndim == 2:
            mask_np = mask_np[..., None]
        rgba_image = np.concatenate([image_np, mask_np], axis=-1)
        return rgba_image.astype(np.uint8)

    def create_3d(
        self,
        image: Optional[InferenceRequestImage],
        mask_input: Optional[Union[np.ndarray, List[List[List[float]]]]] = None,
        **kwargs,
    ):

        with torch.inference_mode():
            if image is None or mask_input is None:
                raise ValueError("Must provide image and mask!")

            rgba_image = self.merge_image_and_mask(image, mask_input)

            result = self.pipeline.run(
                image=rgba_image,
                mask=None,
            )
            return result


def convert_tensor_to_list(tensor_data: torch.Tensor) -> Optional[List[float]]:
    if tensor_data is None:
        return None
    if isinstance(tensor_data, torch.Tensor):
        # Flatten the tensor to ensure we get a 1D list of floats
        return tensor_data.cpu().flatten().tolist()
    return tensor_data


def convert_3d_objects_result_to_api_response(
    raw_result: Dict[str, Any],
    inference_time: float,
) -> Sam3_3D_Objects_Response:

    # Extract and convert mesh (GLB)
    mesh_glb_bytes = None
    glb_mesh = raw_result.pop("glb", None)
    if glb_mesh is not None:
        glb_buffer = BytesIO()
        glb_mesh.export(glb_buffer, "glb")
        glb_buffer.seek(0)
        mesh_glb_bytes = glb_buffer.getvalue()

    # Extract and convert Gaussian splatting
    gaussian_ply_bytes = None
    gaussian = raw_result.pop("gs", None)
    if gaussian is not None:
        gaussian_buffer = BytesIO()
        gaussian.save_ply(gaussian_buffer)
        gaussian_buffer.seek(0)
        gaussian_ply_bytes = gaussian_buffer.getvalue()

    metadata = Sam3_3D_Objects_Metadata(
        rotation=convert_tensor_to_list(raw_result.get("rotation")),
        translation=convert_tensor_to_list(raw_result.get("translation")),
        scale=convert_tensor_to_list(raw_result.get("scale")),
    )

    return Sam3_3D_Objects_Response(
        mesh_glb=mesh_glb_bytes,
        gaussian_ply=gaussian_ply_bytes,
        metadata=metadata,
        time=inference_time,
    )
