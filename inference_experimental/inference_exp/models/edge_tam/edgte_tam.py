from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import InstanceDetections
from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.errors import ModelRuntimeError
from inference_exp.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)
from inference_exp.models.common.model_packages import get_model_package_contents
from inference_exp.models.common.roboflow.pre_processing import (
    images_to_pillow,
)


from transformers import Sam2Processor, EdgeTamModel


class EdgeTAM:
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "EdgeTAM":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "model.safetensors",
                "processor_config.json",
                "video_preprocessor_config.json",
                "preprocessor_config.json",
                "config.json",
            ],
        )
        model = EdgeTamModel.from_pretrained(
            model_name_or_path, local_files_only=True
        ).to(device)
        processor = Sam2Processor.from_pretrained(
            model_name_or_path, local_files_only=True
        )
        return cls(model=model, processor=processor, device=device)

    def __init__(
        self, model: EdgeTamModel, processor: Sam2Processor, device: torch.device
    ):
        self._model = model
        self._processor = processor
        self._device = device

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_points: Optional[Union[List, torch.Tensor]] = None,
        input_labels: Optional[Union[List, torch.Tensor]] = None,
        input_boxes: Optional[Union[List, torch.Tensor]] = None,
        multimask_output: bool = False,
        mask_threshold: float = 0.0,
        binarize: bool = True,
        **kwargs,
    ) -> List[InstanceDetections]:
        pre_processed_images, pre_processing_meta = self.pre_process(
            images=images,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            **kwargs,
        )
        model_results = self.forward(
            pre_processed_images=pre_processed_images,
            multimask_output=multimask_output,
            **kwargs,
        )
        return self.post_process(
            model_results=model_results,
            pre_processing_meta=pre_processing_meta,
            mask_threshold=mask_threshold,
            binarize=binarize,
            **kwargs,
        )

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_points: Optional[Union[List, torch.Tensor]] = None,
        input_labels: Optional[Union[List, torch.Tensor]] = None,
        input_boxes: Optional[Union[List, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:

        # Prepare inputs using the processor
        encoding = self._processor(
            images=images,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self._device)
        # Use processor-produced original_sizes as our metadata for post-processing
        original_sizes = encoding["original_sizes"]
        return encoding, original_sizes

    def forward(
        self,
        pre_processed_images: PreprocessedInputs,
        multimask_output: bool = False,
        **kwargs,
    ) -> RawPrediction:
        with torch.inference_mode():
            # Allow caller to override multimask_output etc. via kwargs
            outputs = self._model(
                **pre_processed_images, multimask_output=multimask_output, **kwargs
            )
        return outputs

    def post_process(
        self,
        model_results: RawPrediction,
        pre_processing_meta: PreprocessingMetadata,
        mask_threshold: float = 0.0,
        binarize: bool = True,
        **kwargs,
    ) -> List[InstanceDetections]:
        # Post-process masks back to original image sizes
        all_masks = self._processor.post_process_masks(
            getattr(model_results, "pred_masks").detach().cpu(),
            pre_processing_meta,
            mask_threshold=mask_threshold,
            binarize=binarize,
        )
        results: List[InstanceDetections] = []
        for masks in all_masks:
            # masks: (num_objects, H, W)
            if isinstance(masks, np.ndarray):
                masks_t = torch.from_numpy(masks)
            else:
                masks_t = (
                    masks if isinstance(masks, torch.Tensor) else torch.tensor(masks)
                )
            # Handle potential temporal/batch singleton dimension: (1, N, H, W) -> (N, H, W)
            if masks_t.ndim == 4:
                if masks_t.shape[0] == 1:
                    masks_t = masks_t.squeeze(0)
                elif masks_t.shape[1] == 1:
                    masks_t = masks_t.squeeze(1)
                else:
                    # Fallback: merge first two dims
                    masks_t = masks_t.flatten(0, 1)
            if masks_t.dtype != torch.bool and binarize:
                masks_t = masks_t > 0
            num_objs = masks_t.shape[0] if masks_t.ndim == 3 else 0
            if num_objs == 0:
                results.append(
                    InstanceDetections(
                        xyxy=torch.zeros((0, 4), dtype=torch.int32),
                        class_id=torch.zeros((0,), dtype=torch.int64),
                        confidence=torch.zeros((0,), dtype=torch.float32),
                        mask=torch.zeros((0, 0, 0), dtype=torch.bool),
                    )
                )
                continue
            xyxy_list = []
            keep_masks = []
            for i in range(num_objs):
                m = masks_t[i]
                if m.dtype != torch.bool and binarize:
                    m = m > 0
                ys, xs = torch.where(m)
                if ys.numel() == 0:
                    # Skip empty masks
                    continue
                x_min, x_max = xs.min().item(), xs.max().item()
                y_min, y_max = ys.min().item(), ys.max().item()
                xyxy_list.append([x_min, y_min, x_max, y_max])
                keep_masks.append(m.unsqueeze(0))
            if len(xyxy_list) == 0:
                results.append(
                    InstanceDetections(
                        xyxy=torch.zeros((0, 4), dtype=torch.int32),
                        class_id=torch.zeros((0,), dtype=torch.int64),
                        confidence=torch.zeros((0,), dtype=torch.float32),
                        mask=torch.zeros((0, 0, 0), dtype=torch.bool),
                    )
                )
                continue
            xyxy_tensor = torch.tensor(xyxy_list, dtype=torch.int32)
            masks_tensor = torch.cat(keep_masks, dim=0)
            # No class predictions from EdgeTAM; default to class 0 with confidence 1.0
            class_ids = torch.zeros((xyxy_tensor.shape[0],), dtype=torch.int64)
            confidences = torch.ones((xyxy_tensor.shape[0],), dtype=torch.float32)
            results.append(
                InstanceDetections(
                    xyxy=xyxy_tensor,
                    class_id=class_ids,
                    confidence=confidences,
                    mask=masks_tensor,
                )
            )
        return results

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_points: Optional[Union[List, torch.Tensor]] = None,
        input_labels: Optional[Union[List, torch.Tensor]] = None,
        input_boxes: Optional[Union[List, torch.Tensor]] = None,
        multimask_output: bool = False,
        mask_threshold: float = 0.0,
        binarize: bool = True,
        **kwargs,
    ) -> List[InstanceDetections]:
        return self.infer(
            images=images,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            multimask_output=multimask_output,
            mask_threshold=mask_threshold,
            binarize=binarize,
            **kwargs,
        )
