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

    def __call__(
        self,
        image: np.ndarray,
        prompts: Optional[List[List[Tuple[int, int, int, int]]]] = None,
    ) -> Tuple:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if prompts:
                self._predictor.load_first_frame(image)

                for i, pts in enumerate(prompts):
                    if len(pts) < 4:
                        continue
                    x1, y1, x2, y2 = pts[:4]
                    x_lt = min(x1, x2)
                    y_lt = min(y1, y2)
                    x_rt = max(x1, x2)
                    y_rt = max(y1, y2)
                    xyxy = np.array([[x_lt, y_lt, x_rt, y_rt]])

                    _, object_ids, mask_logits = self._predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=i,
                        bbox=xyxy
                    )
            else:
                object_ids, mask_logits = self._predictor.track(image)

            masks = (mask_logits > 0.0).cpu().numpy()
            masks = np.squeeze(masks).astype(bool)
            if len(masks.shape) == 2:
                masks = np.expand_dims(masks, axis=0)
            object_ids = np.array(object_ids)
            return object_ids, masks
