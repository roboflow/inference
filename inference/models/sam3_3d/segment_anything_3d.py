import os
import sys
import weakref
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import cv2
import numpy as np
import torch
from filelock import FileLock
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from inference.core.cache.model_artifacts import get_cache_dir
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam3_3d import Sam3_3D_Objects_InferenceRequest
from inference.core.entities.responses.sam3_3d import (
    Sam3_3D_Object_Item,
    Sam3_3D_Objects_Metadata,
    Sam3_3D_Objects_Response,
)
from inference.core.env import DEVICE, MODEL_CACHE_DIR
from inference.core.exceptions import ModelArtefactError
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_roboflow_model_data,
    stream_url_to_cache,
)
from inference.core.utils.image_utils import load_image_rgb

try:
    import pycocotools.mask as mask_utils

    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False


def convert_mask_to_binary(mask_input: Any, image_shape: Tuple[int, int]) -> np.ndarray:
    """Convert polygon, RLE, or binary mask to binary mask (H, W) with values 0/255."""
    height, width = image_shape

    if isinstance(mask_input, np.ndarray):
        return _normalize_binary_mask(mask_input, image_shape)

    if isinstance(mask_input, Image.Image):
        return _normalize_binary_mask(np.array(mask_input.convert("L")), image_shape)

    if isinstance(mask_input, dict) and "counts" in mask_input:
        if not PYCOCOTOOLS_AVAILABLE:
            raise ImportError(
                "pycocotools required for RLE. Install: pip install pycocotools"
            )
        rle = dict(mask_input)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        return _normalize_binary_mask(mask_utils.decode(rle), image_shape)

    if isinstance(mask_input, list):
        points = _parse_polygon_to_points(mask_input)
        if not points or len(points) < 3:
            return np.zeros((height, width), dtype=np.uint8)
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).polygon(points, outline=255, fill=255)
        return np.array(mask, dtype=np.uint8)

    raise TypeError(f"Unsupported mask type: {type(mask_input)}")


def _normalize_binary_mask(
    mask: np.ndarray, image_shape: Tuple[int, int]
) -> np.ndarray:
    """Normalize mask to uint8 with values 0/255. Returns input unchanged if already correct."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    h, w = image_shape
    needs_resize = mask.shape[0] != h or mask.shape[1] != w

    # Check if already in correct format (uint8, 0/255 range, correct size)
    if mask.dtype == np.uint8 and mask.max() > 1 and not needs_resize:
        return mask

    # Convert to uint8 0/255
    if mask.dtype == np.bool_:
        mask = mask.astype(np.uint8) * 255
    elif mask.dtype != np.uint8:
        mask = ((mask > 0).astype(np.uint8)) * 255
    elif mask.max() <= 1:
        mask = mask * 255

    if needs_resize:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask


def _parse_polygon_to_points(polygon: List) -> List[Tuple[float, float]]:
    if polygon is None or (isinstance(polygon, list) and len(polygon) == 0):
        return []
    if isinstance(polygon, np.ndarray):
        if polygon.size == 0:
            return []
        if polygon.ndim == 2 and polygon.shape[1] == 2:
            return [(float(p[0]), float(p[1])) for p in polygon]
        return [
            (float(polygon[i]), float(polygon[i + 1]))
            for i in range(0, len(polygon), 2)
        ]
    if isinstance(polygon[0], (int, float)):
        return [
            (float(polygon[i]), float(polygon[i + 1]))
            for i in range(0, len(polygon), 2)
        ]
    if isinstance(polygon[0], (list, tuple, np.ndarray)):
        return [(float(p[0]), float(p[1])) for p in polygon]
    return []


def _is_single_mask_input(mask_input: Any) -> bool:
    """Check if input is single mask vs list of masks."""
    if mask_input is None or (
        isinstance(mask_input, (list, np.ndarray)) and len(mask_input) == 0
    ):
        return True
    if isinstance(mask_input, np.ndarray):
        return mask_input.ndim == 2
    if isinstance(mask_input, dict) and "counts" in mask_input:
        return True
    if isinstance(mask_input, list):
        first = mask_input[0]
        # Flat polygon: [x1, y1, x2, y2, ...]
        if isinstance(first, (int, float)):
            return True
        # List of RLE dicts
        if isinstance(first, dict) and "counts" in first:
            return False
        # List of 2D numpy arrays (binary masks) -> multiple masks
        if isinstance(first, np.ndarray) and first.ndim == 2:
            return False
        # Check list/tuple elements
        if isinstance(first, (list, tuple)):
            # [[x1, y1], [x2, y2], ...] -> single polygon as points
            if len(first) == 2 and isinstance(first[0], (int, float)):
                return True
            # [[x1, y1, x2, ...], [x1, y1, x2, ...]] -> multiple flat polygons
            if len(first) > 2 and isinstance(first[0], (int, float)):
                return False
    return True


if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(device_count))
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from importlib.resources import files

import tdfy.sam3d_v1

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

import trimesh
from pytorch3d.transforms import (
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)
from tdfy.sam3d_v1.inference_utils import make_scene, ready_gaussian_for_video_rendering


def apply_gaussian_view_correction(scene_gs):
    """
    Apply view correction to Gaussian scene to match GLB orientation.
    Used for combined scene PLY.
    """
    xyz = scene_gs.get_xyz
    device = xyz.device
    dtype = xyz.dtype

    R_view_zup = torch.tensor(
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    )

    new_xyz = xyz @ R_view_zup
    scene_gs.from_xyz(new_xyz)

    q_correction = matrix_to_quaternion(R_view_zup.unsqueeze(0)).squeeze(0)
    old_rotations = scene_gs.get_rotation
    new_rotations = quaternion_multiply(
        q_correction.unsqueeze(0).expand(old_rotations.shape[0], -1), old_rotations
    )
    scene_gs.from_rotation(new_rotations)

    return scene_gs


def prepare_individual_object_for_export(gs):
    """
    Prepare an individual object Gaussian for PLY export.
    """
    from copy import deepcopy

    gs_copy = deepcopy(gs)
    gs_copy = ready_gaussian_for_video_rendering(gs_copy)

    xyz = gs_copy.get_xyz
    device = xyz.device
    dtype = xyz.dtype

    R_view = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], device=device, dtype=dtype
    )

    new_xyz = xyz @ R_view
    gs_copy.from_xyz(new_xyz)

    q_correction = matrix_to_quaternion(R_view.unsqueeze(0)).squeeze(0)
    old_rotations = gs_copy.get_rotation
    new_rotations = quaternion_multiply(
        q_correction.unsqueeze(0).expand(old_rotations.shape[0], -1), old_rotations
    )
    gs_copy.from_rotation(new_rotations)

    return gs_copy


class Sam3_3D_ObjectsPipelineSingleton:
    """Singleton to cache the heavy 3D pipeline initialization."""

    _instances = weakref.WeakValueDictionary()
    _lock = Lock()

    def __new__(cls, config_key: str):
        with cls._lock:
            if config_key not in cls._instances:
                instance = super().__new__(cls)
                instance.config_key = config_key
                cls._instances[config_key] = instance
            return cls._instances[config_key]


class SegmentAnything3_3D_Objects(RoboflowCoreModel):

    task_type = "3d-reconstruction"

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
        slat_generator_checkpoint_path = self.cache_dir / "slat_generator.ckpt"
        ss_decoder_checkpoint_path = self.cache_dir / "ss_decoder.ckpt"
        slat_decoder_checkpoint_path = self.cache_dir / "slat_decoder_gs.ckpt"
        slat_decodergs4_checkpoint_path = self.cache_dir / "slat_decoder_gs_4.ckpt"
        slat_decoder_mesh_checkpoint_path = self.cache_dir / "slat_decoder_mesh.pt"
        dinov2_ckpt_path = self.cache_dir / "dinov2_vitl14_reg4_pretrain.pth"

        config_key = f"{DEVICE}_{pipeline_config_path}"
        singleton = Sam3_3D_ObjectsPipelineSingleton(config_key)

        if not hasattr(singleton, "pipeline"):
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

        # Reference the singleton's pipeline
        self.pipeline = singleton.pipeline
        self._state_lock = Lock()

    def get_infer_bucket_file_list(self) -> list:
        """Get the list of required files for inference.

        Returns:
            list: A list of required files for inference, e.g., ["environment.json"].
        """
        return [
            "moge-vitl.pth",
            "ss_generator.ckpt",
            "slat_generator.ckpt",
            "ss_decoder.ckpt",
            "slat_decoder_gs.ckpt",
            "slat_decoder_gs_4.ckpt",
            "slat_decoder_mesh.pt",
        ]

    def download_model_from_roboflow_api(self) -> None:
        """Override parent method to use streaming downloads for large SAM3_3D model files."""
        lock_dir = MODEL_CACHE_DIR + "/_file_locks"
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = os.path.join(lock_dir, f"{os.path.basename(self.cache_dir)}.lock")
        lock = FileLock(lock_file, timeout=120)
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
                stream_url_to_cache(
                    url=weights_url,
                    filename=filename,
                    model_id=self.endpoint,
                )

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

    def create_3d(
        self,
        image: Optional[InferenceRequestImage],
        mask_input: Optional[Any] = None,
        **kwargs,
    ):
        """
        Generate 3D from image and mask(s).

        Args:
            image: Input image
            mask_input: Mask in any supported format:
                - np.ndarray (H,W) or (N,H,W): Binary mask(s)
                - List[float]: COCO polygon [x1,y1,x2,y2,...]
                - List[List[float]]: Multiple polygons
                - Dict with 'counts'/'size': RLE mask
                - List[Dict]: Multiple RLE masks
        """
        with torch.inference_mode():
            if image is None or mask_input is None:
                raise ValueError("Must provide image and mask!")

            image_np = load_image_rgb(image)
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)
            image_shape = (image_np.shape[0], image_np.shape[1])

            if _is_single_mask_input(mask_input):
                masks = [convert_mask_to_binary(mask_input, image_shape)]
            elif isinstance(mask_input, np.ndarray) and mask_input.ndim == 3:
                masks = [convert_mask_to_binary(m, image_shape) for m in mask_input]
            else:
                masks = [convert_mask_to_binary(m, image_shape) for m in mask_input]

            outputs = []
            for mask in masks:
                result = self.pipeline.run(image=image_np, mask=mask)
                outputs.append(result)

            if len(outputs) == 1:
                result = outputs[0]
                scene_gs = ready_gaussian_for_video_rendering(result["gs"])
                return {
                    "gs": scene_gs,
                    "glb": result["glb"],
                    "objects": outputs,
                }
            else:
                scene_gs = make_scene(*outputs)
                scene_gs = ready_gaussian_for_video_rendering(scene_gs)
                scene_gs = apply_gaussian_view_correction(scene_gs)
                scene_glb = make_scene_glb(*outputs)
                return {
                    "gs": scene_gs,
                    "glb": scene_glb,
                    "objects": outputs,
                }


def convert_tensor_to_list(tensor_data: torch.Tensor) -> Optional[List[float]]:
    if tensor_data is None:
        return None
    if isinstance(tensor_data, torch.Tensor):
        return tensor_data.cpu().flatten().tolist()
    return tensor_data


def convert_3d_objects_result_to_api_response(
    raw_result: Dict[str, Any],
    inference_time: float,
) -> Sam3_3D_Objects_Response:

    mesh_glb_bytes = None
    glb = raw_result.pop("glb", None)
    if glb is not None:
        glb_buffer = BytesIO()
        glb.export(glb_buffer, "glb")
        glb_buffer.seek(0)
        mesh_glb_bytes = glb_buffer.getvalue()

    gaussian_ply_bytes = None
    gaussian = raw_result.pop("gs", None)
    if gaussian is not None:
        gaussian_buffer = BytesIO()
        gaussian.save_ply(gaussian_buffer)
        gaussian_buffer.seek(0)
        gaussian_ply_bytes = gaussian_buffer.getvalue()

    objects = []
    outputs_list = raw_result.pop("objects", [])
    for output in outputs_list:
        obj_glb_bytes = None
        obj_glb = output.get("glb")
        if obj_glb is not None:
            obj_glb_buffer = BytesIO()
            obj_glb.export(obj_glb_buffer, "glb")
            obj_glb_buffer.seek(0)
            obj_glb_bytes = obj_glb_buffer.getvalue()

        obj_ply_bytes = None
        obj_gs = output.get("gs")
        if obj_gs is not None:
            obj_gs_export = prepare_individual_object_for_export(obj_gs)
            obj_ply_buffer = BytesIO()
            obj_gs_export.save_ply(obj_ply_buffer)
            obj_ply_buffer.seek(0)
            obj_ply_bytes = obj_ply_buffer.getvalue()

        obj_metadata = Sam3_3D_Objects_Metadata(
            rotation=convert_tensor_to_list(output.get("rotation")),
            translation=convert_tensor_to_list(output.get("translation")),
            scale=convert_tensor_to_list(output.get("scale")),
        )

        objects.append(
            Sam3_3D_Object_Item(
                mesh_glb=obj_glb_bytes,
                gaussian_ply=obj_ply_bytes,
                metadata=obj_metadata,
            )
        )

    return Sam3_3D_Objects_Response(
        mesh_glb=mesh_glb_bytes,
        gaussian_ply=gaussian_ply_bytes,
        objects=objects,
        time=inference_time,
    )


def transform_glb_to_world(glb_mesh, rotation, translation, scale):
    """
    Transform a GLB mesh from local to world coordinates.
    """
    quat = rotation.squeeze()
    quat_normalized = quat / quat.norm()
    R_layout = quaternion_to_matrix(quat_normalized).cpu().numpy()
    t = translation.squeeze().cpu().numpy()
    s = scale.squeeze().cpu().numpy()[0]

    z_to_y_up = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    y_to_z_up = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)

    verts = glb_mesh.vertices.copy().astype(np.float32)

    verts = verts @ y_to_z_up

    verts = verts * s
    verts = verts @ R_layout
    verts = verts + t

    verts = verts @ z_to_y_up

    glb_mesh.vertices = verts

    if (
        hasattr(glb_mesh, "vertex_normals")
        and glb_mesh.vertex_normals is not None
        and len(glb_mesh.vertex_normals) > 0
    ):
        normals = glb_mesh.vertex_normals.copy().astype(np.float32)
        normals = normals @ y_to_z_up
        normals = normals @ R_layout
        normals = normals @ z_to_y_up
        glb_mesh.vertex_normals = normals

    return glb_mesh


def make_scene_glb(*outputs):
    """
    Combine multiple GLB meshes into a single scene.
    Applies layout transforms and a final view correction rotation.
    """
    scene = trimesh.Scene()

    for i, output in enumerate(outputs):
        glb = output["glb"]
        glb = glb.copy()

        glb = transform_glb_to_world(
            glb,
            output["rotation"],
            output["translation"],
            output["scale"],
        )
        scene.add_geometry(glb, node_name=f"object_{i}")

    R_view = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)
    for geom_name in scene.geometry:
        mesh = scene.geometry[geom_name]
        mesh.vertices = (mesh.vertices.astype(np.float32)) @ R_view
        if (
            hasattr(mesh, "vertex_normals")
            and mesh.vertex_normals is not None
            and len(mesh.vertex_normals) > 0
        ):
            mesh.vertex_normals = (mesh.vertex_normals.astype(np.float32)) @ R_view

    return scene
