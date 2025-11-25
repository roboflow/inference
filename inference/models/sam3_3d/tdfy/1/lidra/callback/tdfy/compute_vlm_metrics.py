from pathlib import Path
import torch
from typing import List, Any, Tuple, Sequence, Optional, Callable, Dict, Iterator
import pandas as pd
from lightning.pytorch.callbacks import Callback
from lightning.pytorch import LightningModule, Trainer
from loguru import logger

from lidra.data.collator import auto_uncollate
from lidra.data.dataset.return_type import (
    extract_sample_uuid,
    extract_data,
    SampleUuidUtils,
    AbstractDatasetReturnType,
)
from lidra.utils.visualization.image_mask import get_masked_sample_cv2
from lidra.data.utils import (
    build_batch_extractor,
    empty_mapping,
)
from lidra.metrics.tdfy.metric_collection_per_sample import (
    TdfyPerSample,
    NanFoundInMetricException,
)

from lidra.data.dataset.tdfy.pose_target import PoseTargetConvention
from lidra.metrics.tdfy.vlmscore import GPTScorer
from lidra.callback.tdfy.compute_metrics import TdfySingleTrialMetricsCallback


class TdfySingleTrialVLMMetricsCallback(TdfySingleTrialMetricsCallback):
    def __init__(
        self,
        output_dir: str,
        evaluation_fn: Callable,
        scorer: Optional[GPTScorer] = None,
        *args,
        **kwargs,
    ):
        super().__init__(output_dir, evaluation_fn, *args, **kwargs)

        if scorer:
            self.gptscorer = scorer
            TdfySingleTrialVLMMetricsCallback._current_scorer = self.gptscorer

    @classmethod
    def _get_current_scorer(cls):
        """Get the current scorer instance for static methods"""
        if hasattr(cls, "_current_scorer"):
            return cls._current_scorer
        return None

    @staticmethod
    def em_vlm_evaluation_fn(
        pred_dict: Dict[str, torch.Tensor], sample: Dict[str, torch.Tensor], **kwargs
    ):
        assert (
            "trial_idx" in pred_dict
        ), "Trial index must be present in the predictions"

        scorer = TdfySingleTrialVLMMetricsCallback._get_current_scorer()
        score, rationale = scorer.score(sample)
        return {
            "trial": pred_dict["trial_idx"],
            "GPT_score": score,
            "GPT_rationale": rationale,
        }

    @staticmethod
    def pre_vlm_preprocess_fn(
        module: LightningModule,
        batch: AbstractDatasetReturnType,
        remaining_key_mapping: Optional[Dict[str, str]] = None,
    ):
        sample_uuids = extract_sample_uuid(batch)
        output_dict = extract_data(batch)

        full_rgbs = output_dict["img_rgb_full"].unbind(0)
        full_masks = output_dict["img_mask_full"].unbind(0)
        croppeds = output_dict["img_cropped_rgb"].unbind(0)
        cropped_masks = output_dict["img_cropped_mask"].unbind(0)

        full_highlighteds = [
            get_masked_sample_cv2(full_rgbs[i], full_masks[i], add_bbox=False)
            for i in range(len(full_rgbs))
        ]  # C,H,W in [0,1]
        cropped_highlighteds = [
            get_masked_sample_cv2(croppeds[i], cropped_masks[i], add_bbox=False)
            for i in range(len(croppeds))
        ]  # C,H,W in [0,1]

        if remaining_key_mapping is None:
            remaining_key_mapping = {}
        else:
            remaining_key_mapping = {
                k: output_dict[v] for k, v in remaining_key_mapping.items()
            }

        return (
            sample_uuids,
            {
                "trial_idx": tuple(output_dict["trial_idx"]),
                "input_full_highlighted": torch.stack(full_highlighteds),
                "input_cropped_highlighted": torch.stack(cropped_highlighteds),
                **remaining_key_mapping,
            },
        )


def batch_preprocess_fn_for_vlm_scoring(
    module: LightningModule,
    batch: AbstractDatasetReturnType,
    **kwargs,
):
    sample_uuids = extract_sample_uuid(batch)
    data = extract_data(batch)
    pred_data = data["prediction"]
    pred_data["trial_idx"] = data["trial_idx"]
    batch = (sample_uuids, pred_data)
    return TdfySingleTrialVLMMetricsCallback.pre_vlm_preprocess_fn(
        module,
        batch,
        **kwargs,
    )
