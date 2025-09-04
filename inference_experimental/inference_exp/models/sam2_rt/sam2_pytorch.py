from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.errors import MissingDependencyError
from inference_exp.models.common.model_packages import get_model_package_contents

try:
    import hydra
    from sam2.build_sam import build_sam2_camera_predictor
except ImportError as import_error:
    raise MissingDependencyError(
        message=f"Could not import SAM2 model, please consult README for installation instructions.",
    ) from import_error


class SAM2ForInstanceSegmentationPyTorch:
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "SAM2ForInstanceSegmentationPyTorch":
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
        predictor = build_sam2_camera_predictor(
            config_file=Path(model_package_content["sam2-rt.yaml"]).name,
            ckpt_path=model_package_content["weights.pt"],
            device=device,
        )
        return cls(predictor=predictor, device=device)

    def __init__(self, predictor, device: torch.device):
        self._predictor = predictor
        self._device = device

    def prompt(
        self,
        image: Union[np.ndarray, torch.Tensor],
        prompts: List[List[Tuple[int, int, int, int]]],
        state_dict: Optional[dict] = None,
    ) -> tuple:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        self._predictor.load_first_frame(image)
        if state_dict is not None:
            self._predictor.load_state_dict(state_dict)
        for i, pts in enumerate(prompts):
            if len(pts) < 4:
                continue
            x1, y1, x2, y2 = pts[:4]
            x_lt = int(round(min(x1, x2)))
            y_lt = int(round(min(y1, y2)))
            x_rb = int(round(max(x1, x2)))
            y_rb = int(round(max(y1, y2)))
            xyxy = np.array([[x_lt, y_lt, x_rb, y_rb]])

            _, object_ids, mask_logits = self._predictor.add_new_prompt(
                frame_idx=0, obj_id=i, bbox=xyxy
            )
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, axis=0)
        object_ids = np.array(object_ids)
        return masks, object_ids, self._predictor.state_dict()

    def track(
        self,
        image: Union[np.ndarray, torch.Tensor],
        state_dict: dict,
    ) -> tuple:
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        self._predictor.load_state_dict(state_dict)
        object_ids, mask_logits = self._predictor.track(image)
        masks = (mask_logits > 0.0).cpu().numpy()
        masks = np.squeeze(masks).astype(bool)
        if len(masks.shape) == 2:
            masks = np.expand_dims(masks, axis=0)
        object_ids = np.array(object_ids)
        return masks, object_ids, self._predictor.state_dict()
