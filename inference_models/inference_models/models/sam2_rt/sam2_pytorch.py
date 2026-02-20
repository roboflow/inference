from pathlib import Path
from threading import RLock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.errors import MissingDependencyError, ModelRuntimeError
from inference_models.models.common.model_packages import get_model_package_contents

try:
    import hydra
    from sam2.build_sam import build_sam2_camera_predictor
    from sam2.sam2_camera_predictor import SAM2CameraPredictor
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import SAM2 model, please contact Roboflow for further instructions.",
        help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
    ) from import_error


class SAM2ForStream:
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "SAM2ForStream":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "weights.pt",
                "sam2-rt.yaml",
            ],
        )
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(
            config_dir=Path(model_package_content["sam2-rt.yaml"]).parent.as_posix(),
            version_base=None,
        )
        predictor: SAM2CameraPredictor = build_sam2_camera_predictor(
            config_file=Path(model_package_content["sam2-rt.yaml"]).name,
            ckpt_path=model_package_content["weights.pt"],
            device=device,
        )
        return cls(predictor=predictor, device=device)

    def __init__(self, predictor: SAM2CameraPredictor, device: torch.device):
        self._predictor = predictor
        self._device = device
        self._lock = RLock()

    def prompt(
        self,
        image: Union[np.ndarray, torch.Tensor],
        bboxes: Union[Tuple[int, int, int, int], List[Tuple[int, int, int, int]]],
        state_dict: Optional[dict] = None,
        clear_old_points: bool = True,
        normalize_coords: bool = True,
        frame_idx: int = 0,
    ) -> tuple:
        with self._lock:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            if clear_old_points or not self._predictor.condition_state:
                self._predictor.load_first_frame(image)
            if state_dict is not None:
                self._predictor.load_state_dict(state_dict)
            obj_id = 0
            if (
                self._predictor.condition_state
                and self._predictor.condition_state["obj_ids"]
            ):
                obj_id = max(self._predictor.condition_state["obj_ids"]) + 1
            if not isinstance(bboxes, list):
                bboxes = [bboxes]
            for pts in bboxes:
                if len(pts) < 4:
                    continue
                x1, y1, x2, y2 = pts[:4]
                x_lt = int(round(min(x1, x2)))
                y_lt = int(round(min(y1, y2)))
                x_rb = int(round(max(x1, x2)))
                y_rb = int(round(max(y1, y2)))
                xyxy = np.array([[x_lt, y_lt, x_rb, y_rb]])

                _, object_ids, mask_logits = self._predictor.add_new_prompt(
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    bbox=xyxy,
                    clear_old_points=clear_old_points,
                    normalize_coords=normalize_coords,
                )
                obj_id += 1
            masks = (mask_logits > 0.0).cpu().numpy()
            masks = np.squeeze(masks).astype(bool)
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, axis=0)
            object_ids = np.array(object_ids)
            return masks, object_ids, self._predictor.state_dict()

    def track(
        self,
        image: Union[np.ndarray, torch.Tensor],
        state_dict: Optional[dict] = None,
    ) -> tuple:
        with self._lock:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            if state_dict is not None:
                self._predictor.load_state_dict(state_dict)
            if not self._predictor.condition_state:
                raise ModelRuntimeError(
                    "Attempt to track with no prior call to prompt; prompt must be called first"
                )
            object_ids, mask_logits = self._predictor.track(image)
            masks = (mask_logits > 0.0).cpu().numpy()
            masks = np.squeeze(masks).astype(bool)
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, axis=0)
            object_ids = np.array(object_ids)
            return masks, object_ids, self._predictor.state_dict()
