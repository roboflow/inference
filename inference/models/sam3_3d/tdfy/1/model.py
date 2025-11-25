# entrypoint running inside the Triton server
import triton_python_backend_utils as pb_utils
import os
from cloudpathlib.cloudpath import CloudPath
import tempfile
import io
import numpy as np
import base64
from PIL import Image
import functools
from loguru import logger
import sys
import json
from omegaconf import OmegaConf
from hydra.utils import instantiate
from plyfile import PlyData, PlyElement
from pycocotools import mask as maskUtils
import requests
import threading
import uuid
import urllib.request
from tqdm.auto import tqdm
import ssl

# redirect logs to stdout
logger.remove()
logger.add(sys.stdout)

import certifi


# WARNING: DO NOT IMPORT ANYTHING THAT MIGHT LOAD CUDA_VISIBLE_DEVICES, NO IMPORT lidra!!!!
def import_torch_and_inference():
    import torch
    from lidra.pipeline.inference_pipeline import InferencePipeline
    from lidra.profiler.timeit import timeit

    return torch, InferencePipeline, timeit


def wrap_log_exception(fn):
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        logger.debug(f"> entering function {fn.__qualname__}")
        try:
            return fn(*args, **kwargs)
        except:
            logger.opt(exception=True).error("exception occured")
            raise
        finally:
            logger.debug(f"< exiting function {fn.__qualname__}")

    return wrapped_fn


def export_return_obj_to_tempfile(return_obj, return_type, temp_file):
    if return_type == "glb":
        return_obj.export(temp_file.name, file_type="glb")
    elif return_type == "gs":
        return_obj.save_ply(temp_file.name)
    elif return_type == "gs_4":
        return_obj.save_ply(temp_file.name)
    elif return_type == "voxel":
        voxel_array = return_obj.cpu().numpy()
        # Create structured array using view - most efficient
        vertex = voxel_array.view(
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]
        ).squeeze()
        el = PlyElement.describe(vertex, "vertex")
        PlyData([el], text=False).write(temp_file.name)


def upload_tempfile_to_s3(temp_file, file_prefix, return_type, ext):
    upload_path = (
        f"s3://sam3d-models-us-east-2/returns/{file_prefix}_{return_type}.{ext}"
    )
    cloud_path = CloudPath(upload_path)
    cloud_path.upload_from(temp_file.name)
    return upload_path


def download_image_as_pil(url):
    """
    Download a file from a CDN URL and return its base64-encoded content as a string.
    Args:
        url (str): The CDN URL of the file to download.
    Returns:
        Image: Image from PIL
    """
    try:
        # Open the URL and get the total size of the video
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=context) as response:
            total_size = int(response.info().get("Content-Length", 0))
            # Fallback to be a video
            content_type = response.info().get("Content-Type", "image/png")
            extension = content_type.split("/")[1]
            # Initialize a progress bar
            block_size = 1024
            t = tqdm(total=total_size, unit="B", unit_scale=True)
            # Save the video to a local file
            with tempfile.NamedTemporaryFile(
                suffix=extension, delete=True
            ) as temp_file:
                while True:
                    data = response.read(block_size)
                    if not data:
                        break
                    t.update(len(data))
                    temp_file.write(data)
                t.close()
                img = Image.open(temp_file)
                img.load()
                return img
    except Exception as e:
        logger.info(str(e))
        return None


class TritonPythonModel:
    SUFFIX_MAP = {
        "glb": ".glb",
        "gs": ".ply",
        "gs_4": ".ply",
        "voxel": ".ply",
    }

    def checkpoint_path(self, module_name, ext):
        return os.path.join(self._current_folder, "checkpoints", f"{module_name}.{ext}")

    def config_path(self, module_name):
        return os.path.join(self._current_folder, "checkpoints", f"{module_name}.yaml")

    def load_kwargs(self):
        filepath = os.path.join(self._current_folder, "checkpoints", "kwargs.json")
        with open(filepath, "r") as f:
            return json.load(f)

    @wrap_log_exception
    def initialize(self, args):
        logger.info(f"{args=}")
        assert args["model_instance_kind"] == "GPU"  # only run on GPU

        # hacky fix of spconv
        # spconv crashes when running on anything other than cuda:0
        # error : "merge_sort: failed to synchronize: cudaErrorIllegalAddress: an illegal memory access was encountered"
        os.environ["CUDA_VISIBLE_DEVICES"] = args["model_instance_device_id"]
        global torch, InferencePipeline, timeit
        torch, InferencePipeline, timeit = import_torch_and_inference()
        device = torch.device("cuda", 0)

        logger.info(f"model will use device {device}")
        self._current_folder = os.path.dirname(__file__)

        # point torch hub cached files to local folder
        torch.hub.set_dir(os.path.join(self._current_folder, "..", "hub"))

        # kwargs = self.load_kwargs()

        pipeline_config = OmegaConf.load(
            os.path.join(self._current_folder, "checkpoints", f"pipeline.yaml")
        )
        logger.info(f"self._current_folder {self._current_folder}")
        pipeline_config["device"] = "cuda"
        pipeline_config["workspace_dir"] = self._current_folder
        self.keep_result = ["rotation", "translation", "scale"]
        self._model = None
        self._model_ready = False  # Optional: flag to check if model is ready

        def instantiate_model():
            self._model = instantiate(pipeline_config)
            logger.info(f"Model version {self._model.version}")
            self._model_ready = True  # Optional: set flag when done

        # Start model instantiation in a separate thread
        threading.Thread(target=instantiate_model, daemon=True).start()
        logger.info("Model instantiation started in background thread.")
        # Return immediately to avoid timeout
        return

    def encode_buffer(self, buffer: bytes):
        buffer_str = base64.b64encode(buffer.getvalue())
        buffer_str = buffer_str.decode("utf8")
        buffer_arr = np.array(buffer_str)
        buffer_arr = buffer_arr.astype(object)
        return buffer_arr

    def decode_image_from_string(self, string: str):
        buffer = base64.b64decode(string)
        buffer = io.BytesIO(buffer)
        return Image.open(buffer)

    def convert_tensor_to_list(self, result: dict):
        res = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().tolist()

            res[k] = v

        return res

    def load_image_from_request(self, request, kwargs: dict):
        # due to legacy reason, we need to support three ways of passing in images
        # (a) image cdn url and mask rle encoding
        # (b) image and mask base64 encoding
        # (c) image and mask together as an RGB-A file in S3 bucket
        if "cdn_url" in kwargs and "mask_rle" in kwargs:
            # if user provides img_cdn_url, then we should load using cdn_url
            img_cdn_url = kwargs["cdn_url"]
            mask_rle_raw = kwargs["mask_rle"]
            image = download_image_as_pil(img_cdn_url)

            # decode rle encoding into a mask numpy array
            width, height = image.size
            mask_rle_obj = {
                "size": [width, height],
                "counts": mask_rle_raw,
            }
            mask_np = maskUtils.decode(mask_rle_obj) * 255
            # flip and rotate to accomodate row/column major order
            mask = Image.fromarray(mask_np.astype(np.uint8))
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.rotate(-90, expand=True)
        elif "s3_path" in kwargs:
            # if user provides s3_path, then we should load from s3 bucket
            s3_path = kwargs["s3_path"]
            logger.info(f"s3_path received {s3_path}")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                cloud_path = CloudPath(s3_path)
                cloud_path.download_to(temp_path)
                image = Image.open(temp_path)
                mask = None
        else:
            # otherwise user needs to provide image and mask base64 encoding
            image = pb_utils.get_input_tensor_by_name(request, "image")
            mask = pb_utils.get_input_tensor_by_name(request, "mask")
            if mask is not None:
                mask = mask.as_numpy()
                mask = self.decode_image_from_string(mask.item())
            if image is not None:
                image = image.as_numpy()
                image = self.decode_image_from_string(image.item())
        kwargs.pop("cdn_url", None)
        kwargs.pop("mask_rle", None)
        kwargs.pop("s3_path", None)
        return image, mask

    def handle_request(self, request):
        logger.info(certifi.__version__)
        # Check if model is ready
        if not getattr(self, "_model_ready", False):
            # Return an error response indicating the model is still loading
            error_message = "Model is still loading. Please try again later."
            logger.warning(error_message)
            # If using Triton Inference Server, you might use pb_utils.InferenceResponse
            # Otherwise, adapt this to your serving framework
            raise ModelNotReadyError(error_message)

        # prepare inputs
        seed = pb_utils.get_input_tensor_by_name(request, "seed")
        seed = seed.as_numpy().item() if seed else 0
        kwargs = pb_utils.get_input_tensor_by_name(request, "kwargs")
        kwargs = kwargs.as_numpy().item() if kwargs else "{}"
        kwargs = json.loads(kwargs)
        image, mask = self.load_image_from_request(request, kwargs)
        logger.info(f"kwargs: {kwargs}")

        # run model
        result = self._model.run(image, mask=mask, seed=seed, **kwargs)

        # iterate through potential output path
        s3_path_map = {}
        for return_type in ["glb", "gs", "gs_4", "voxel"]:
            s3_path_map[return_type] = None
            if "s3_path" in kwargs:
                file_prefix = kwargs["s3_path"].split("/")[-1].replace(".png", "")
            else:
                file_prefix = uuid.uuid4()
            return_obj = result.pop(return_type, None)
            if return_obj is not None:
                with tempfile.NamedTemporaryFile(
                    suffix=TritonPythonModel.SUFFIX_MAP[return_type], delete=True
                ) as temp_file:
                    export_return_obj_to_tempfile(return_obj, return_type, temp_file)
                    return_path = upload_tempfile_to_s3(
                        temp_file,
                        file_prefix,
                        return_type,
                        TritonPythonModel.SUFFIX_MAP[return_type],
                    )
                    s3_path_map[return_type] = return_path

        result = {k: result.get(k, None) for k in self.keep_result}
        result = self.convert_tensor_to_list(result)
        response = pb_utils.InferenceResponse(
            output_tensors=[
                pb_utils.Tensor(
                    "mesh",
                    np.array(s3_path_map["glb"]).astype(object),
                ),
                pb_utils.Tensor(
                    "gaussian",
                    np.array(s3_path_map["gs"]).astype(object),
                ),
                pb_utils.Tensor(
                    "gaussian_4",
                    np.array(s3_path_map["gs_4"]).astype(object),
                ),
                pb_utils.Tensor(
                    "voxel",
                    np.array(s3_path_map["voxel"]).astype(object),
                ),
                pb_utils.Tensor(
                    "others",
                    np.array(json.dumps(result)).astype(object),
                ),
            ],
        )
        return response

    @wrap_log_exception
    def execute(self, requests):
        logger.info(f"{requests=}")
        responses = [self.handle_request(req) for req in requests]
        return responses


class ModelNotReadyError(Exception):
    pass
