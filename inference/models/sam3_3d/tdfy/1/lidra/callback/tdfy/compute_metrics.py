from pathlib import Path
import torch
from typing import List, Any, Tuple, Optional, Callable, Dict, Iterator
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
from lidra.data.utils import (
    build_batch_extractor,
)
from lidra.metrics.tdfy.metric_collection_per_sample import (
    TdfyPerSample,
    NanFoundInMetricException,
)


class MetricsCallback(Callback):
    def __init__(
        self,
        output_dir: str,
        evaluation_fn: Callable,
        prediction_preprocess_fn: Optional[Callable] = None,
        batch_preprocess_fn: Optional[Callable] = None,
        sample_extractor_fn: Dict[str, Any] = ["trial_idx"],
        prediction_extractor_fn: Dict[str, Any] = {"prediction": "volume_pred"},
        uncollate_fn: Optional[Callable] = None,
        metrics_fstem: str = "metrics",
        attempt_merge: bool = True,
    ):
        super().__init__()

        # Callables
        self.evaluation_fn = evaluation_fn
        self.prediction_preprocess_fn = (
            prediction_preprocess_fn
            if prediction_preprocess_fn is not None
            else lambda pl_module, pred: pred
        )
        self.batch_preprocess_fn = (
            batch_preprocess_fn
            if batch_preprocess_fn is not None
            else lambda pl_module, batch: batch
        )
        self.uncollate_fn = (
            uncollate_fn if uncollate_fn is not None else auto_uncollate()
        )

        self.sample_extractor_fn = build_batch_extractor(sample_extractor_fn)
        self.prediction_extractor_fn = build_batch_extractor(prediction_extractor_fn)

        # Paths
        self.output_dir = Path(output_dir)
        if output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_fstem = metrics_fstem
        self.output_fpath = self.output_dir / f"{metrics_fstem}.csv"

        # State
        self.attempt_merge = attempt_merge
        self.stored_metrics = []

    def compute_metrics_per_sample(
        self,
        pl_module: LightningModule,
        outputs: torch.Tensor | Dict[str, Any] | None,
        batch: Any,
    ) -> pd.DataFrame:
        """Returns a dataframe of metrics per sample"""
        metrics_df = [
            metrics_dict
            for _, metrics_dict in self._compute_metrics_per_sample(
                pl_module, outputs, batch
            )
        ]
        return pd.DataFrame(metrics_df)

    def _compute_metrics_per_sample(
        self,
        pl_module: LightningModule,
        outputs: torch.Tensor | Dict[str, Any] | None,
        batch: Any,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        sample_uuids, samples = self.batch_preprocess_fn(pl_module, batch)
        _, samples = self.sample_extractor_fn(samples)

        prediction = self.prediction_preprocess_fn(pl_module, outputs)
        _, prediction = self.prediction_extractor_fn(prediction)

        uncollated_sample_uuids = self.uncollate_fn(sample_uuids)
        uncollated_samples = self.uncollate_fn(samples)
        uncollated_prediction = self.uncollate_fn(prediction)

        for example_uuid, pred_dict, sample in zip(
            uncollated_sample_uuids, uncollated_prediction, uncollated_samples
        ):
            example_uuid = SampleUuidUtils.demote(example_uuid)
            try:
                metrics_dict = self.evaluation_fn(pred_dict, sample)
            except NanFoundInMetricException as e:
                logger.opt(exception=e).warning(
                    f"NaN found in metrics for sample {example_uuid}: {e}"
                )
                continue
            yield example_uuid, metrics_dict

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for example_uuid, metrics_dict in self._compute_metrics_per_sample(
            pl_module, outputs, batch
        ):
            self.stored_metrics.append((example_uuid, metrics_dict))

    def on_predict_epoch_end(self, trainer, pl_module):
        metrics_df = self._make_metrics_df(self.stored_metrics)
        metrics_df.to_csv(self._rank_to_filepath(trainer.global_rank), index=False)
        self.stored_metrics = []
        if self.attempt_merge:
            self._attempt_merge(trainer)

    @staticmethod
    def _make_metrics_df(stored_metrics: List[Tuple[str, Dict[str, Any]]]):
        metrics_df = [
            {"sample_uuid": SampleUuidUtils.demote(sample_uuid), **metrics_dict}
            for sample_uuid, metrics_dict in stored_metrics
        ]
        return pd.DataFrame(metrics_df)

    def _rank_to_filepath(self, rank: int):
        return self.output_dir / f"{self.output_fstem}_{rank}.csv"

    def _attempt_merge(self, trainer: Trainer):
        trainer.strategy.barrier(
            f"{self.__class__.__name__}.{self.output_fstem}.on_validation_epoch_end"
        )  # wait for all processes

        if trainer.global_rank == 0:
            # some of those files might not be local, this is why we "attempt"
            try:
                all_data = [
                    pd.read_csv(self._rank_to_filepath(rank))
                    for rank in range(trainer.world_size)
                ]
                all_data = pd.concat(all_data, ignore_index=True)
                all_data.to_csv(self.output_fpath, index=False)
            except:
                logger.opt(exception=True).warning(f"could not merge prediction files")


class TdfySingleTrialMetricsCallback(MetricsCallback):
    def __init__(
        self,
        output_dir: str,
        evaluation_fn: Optional[Callable],
        *args,
        **kwargs,
    ):
        if evaluation_fn is None:
            evaluation_fn = TdfySingleTrialMetricsCallback.em_structure_evaluation_fn
        super().__init__(output_dir, evaluation_fn, *args, **kwargs)

    @staticmethod
    def em_structure_evaluation_fn(
        pred_dict: Dict[str, torch.Tensor],
        sample: Dict[str, torch.Tensor],
        occupancy_volume_resolution: Optional[int] = None,
    ):
        assert (
            "trial_idx" in pred_dict
        ), "Trial index must be present in the predictions"

        metrics_dict = TdfyPerSample.evaluate(
            prediction=pred_dict,
            target=sample,
            occupancy_volume_resolution=occupancy_volume_resolution,
        )
        metrics_dict["trial"] = pred_dict["trial_idx"]
        return metrics_dict

    @staticmethod
    def em_batch_preprocess_fn(
        module: LightningModule,
        batch: AbstractDatasetReturnType,
        module_extractor_str: Optional[str] = None,
        batch_key: str = "mean",
        output_key: str = "occupancy_volume",
        remaining_key_mapping: Optional[Dict[str, str]] = None,
        pose_target_convention: Optional[str] = None,
    ):
        """Adds sample['occupancy_volume']=module(samples['mean']) to the batch"""
        if pose_target_convention is not None:
            raise NotImplementedError("This should be moved to the evalaution")
        sample_uuids = extract_sample_uuid(batch)
        samples = extract_data(batch)
        return (
            sample_uuids,
            TdfySingleTrialMetricsCallback.decode_shape_latents_and_passthrough_keys(
                module,
                samples,
                module_extractor_str,
                batch_key,
                output_key,
                remaining_key_mapping,
            ),
        )

    @staticmethod
    def decode_shape_latents_and_passthrough_keys(
        module: LightningModule,
        samples: Dict[str, Any],
        module_extractor_str: Optional[str] = None,
        batch_key: str = "mean",
        output_key: str = "occupancy_volume",
        remaining_key_mapping: Optional[Dict[str, str]] = None,
    ):
        module = extract_module(module, module_extractor_str)
        volume = module(samples[batch_key])

        if remaining_key_mapping is None:
            remaining_key_mapping = samples
        else:
            remaining_key_mapping = {
                k: samples[v] for k, v in remaining_key_mapping.items()
            }
        return {
            output_key: volume,
            **remaining_key_mapping,
        }


def inference_pipeline_prediction_preprocess_fn(
    module: LightningModule,
    samples: Dict[str, Any],
    module_extractor_str: Optional[str] = None,
    batch_key: str = "x_shape_latent",
    output_key: str = "occupancy_volume",
    remaining_key_mapping: Optional[Dict[str, str]] = None,
):
    """
    Process inference pipeline predictions for metrics computation.

    This function is similar to decode_shape_latents_and_passthrough_keys
    but handles the case where shape latent might be None or already decoded.

    Args:
        module: Lightning module containing the decoder
        samples: Dictionary of predictions from inference pipeline
        module_extractor_str: Path to decoder module within the Lightning module
        batch_key: Key for shape latent in samples dict
        output_key: Key to use for decoded shape volume
        remaining_key_mapping: Mapping for additional keys to pass through

    Returns:
        Dictionary with decoded shape volume and mapped remaining keys
    """
    # Decode shape latent if present
    volume = None
    if batch_key in samples and samples[batch_key] is not None:
        shape_latent = samples[batch_key]
        # Only decode if we have a valid shape latent
        if shape_latent.numel() > 0:
            decoder = extract_module(module, module_extractor_str)
            volume = decoder(shape_latent)

    # Handle remaining keys
    if remaining_key_mapping is None:
        # Pass through all keys except the batch_key
        remaining_keys = {k: v for k, v in samples.items() if k != batch_key}
    else:
        # Map specific keys according to the mapping
        remaining_keys = {}
        for k, v in remaining_key_mapping.items():
            if v in samples:
                remaining_keys[k] = samples[v]

    return {
        output_key: volume,
        **remaining_keys,
    }


def coords_prediction_preprocess_fn(
    module: LightningModule,
    samples: Dict[str, Any],
    module_extractor_str: Optional[str] = None,
    batch_key: str = "x_shape_latent",
    output_key: str = "occupancy_volume",
    remaining_key_mapping: Optional[Dict[str, str]] = None,
):
    """
    Process inference pipeline predictions for metrics computation.

    This function is similar to decode_shape_latents_and_passthrough_keys
    but handles the case where shape latent might be None or already decoded.

    Args:
        module: Lightning module containing the decoder
        samples: Dictionary of predictions from inference pipeline
        module_extractor_str: Path to decoder module within the Lightning module
        batch_key: Key for shape latent in samples dict
        output_key: Key to use for decoded shape volume
        remaining_key_mapping: Mapping for additional keys to pass through

    Returns:
        Dictionary with decoded shape volume and mapped remaining keys
    """
    from lidra.metrics.tdfy.occupancy.pointcloud import create_occupancy_volume

    # Decode shape latent if present
    volumes = []
    for i in range(samples["coords"].shape[0]):
        volume = create_occupancy_volume(
            (samples["coords"][i, ..., 1:] + 0.5) / 64 - 0.5, n_voxels=64
        )
        volumes.append(volume)
    volume = torch.stack(volumes, dim=0).unsqueeze(1)

    # Handle remaining keys
    if remaining_key_mapping is None:
        # Pass through all keys except the batch_key
        remaining_keys = {k: v for k, v in samples.items() if k != batch_key}
    else:
        # Map specific keys according to the mapping
        remaining_keys = {}
        for k, v in remaining_key_mapping.items():
            if v in samples:
                remaining_keys[k] = samples[v]

    return {
        output_key: volume,
        **remaining_keys,
    }


def extract_module(module: LightningModule, module_extractor_str: Optional[str] = None):
    """
    Get a module extractor from a string

    Args:
        module: The module to get the extractor from
        module_extractor_str: The string to get the extractor from.
            If None, the identity function is returned.
            If '.', the module itself is returned.
            If a string, it is assumed to be a nested attribute of the module:
                e.g. 'backbone.decoder' returns module.backbone.decoder
    Returns:
        nn.Module
    """
    if module_extractor_str is None:
        return lambda x: x
    elif module_extractor_str == ".":
        return module
    else:
        # Handle nested attributes like 'backbone.decoder'
        parts = module_extractor_str.split(".")
        result = module
        for part in parts:
            result = getattr(result, part)
        return result


# Used for decoding cached predictions
def batch_preprocess_fn_for_cached_predictions(
    module: LightningModule,
    batch: AbstractDatasetReturnType,
    **kwargs,
):
    sample_uuids = extract_sample_uuid(batch)
    data = extract_data(batch)
    data = data["sample"]
    batch = (sample_uuids, data)
    return TdfySingleTrialMetricsCallback.em_batch_preprocess_fn(
        module,
        batch,
        **kwargs,
    )


def extract_trial_prediction_from_cached_prediction(
    batch: AbstractDatasetReturnType,
):
    data = extract_data(batch)
    return {
        "trial_idx": data["trial_idx"],
        **data["prediction"],
    }


def extract_trial_prediction_from_inference_pipeline(
    batch: AbstractDatasetReturnType,
    key_mapping: Optional[Dict[str, str]] = None,
    dimension_adjustments: Optional[Dict[str, str]] = None,
    passthrough_keys: Optional[List[str]] = None,
    pose_convention: Optional[str] = "ScaleShiftInvariant",
    convert_instance_pose: bool = True,
):
    """
    Extract and transform inference pipeline predictions to match expected format.

    The inference pipeline outputs have different key names and tensor dimensions
    compared to the training model outputs. This function maps between the formats.

    Args:
        batch: The batch containing predictions from inference pipeline
        key_mapping: Dict mapping from pipeline keys to expected keys
                    Default maps common inference pipeline outputs
        dimension_adjustments: Dict of keys that need dimension adjustment
                             Values are adjustment operations like "squeeze_0", "squeeze_-1", "squeeze_1"
        passthrough_keys: List of keys to pass through unchanged
        pose_convention: Pose target convention to use if converting instance pose
        convert_instance_pose: Whether to convert instance pose to pose targets

    Returns:
        Transformed prediction dict matching expected format
    """
    from lidra.data.dataset.tdfy.pose_target import PoseTargetConverter

    data = extract_data(batch)
    pred = data["prediction"]

    # Default key mappings if not provided
    if key_mapping is None:
        key_mapping = {
            "shape": "x_shape_latent",
            "translation": "x_instance_translation",
            "scale": "x_instance_scale",
            "rotation": "x_instance_rotation",
            "translation_scale": "x_translation_scale",
            "6drotation_normalized": "x_instance_rotation_6d_normalized",
        }

    # Default dimension adjustments
    if dimension_adjustments is None:
        dimension_adjustments = {
            "x_shape_latent": "squeeze_0",  # Remove batch dim [1, 4096, 8] -> [4096, 8]
            "x_translation_scale": "squeeze_-1",  # [1, 1, 1] -> [1, 1]
            "x_instance_rotation_6d_normalized": "squeeze_1",  # [1, 1, 6] -> [1, 6]
        }

    # Default passthrough keys
    if passthrough_keys is None:
        passthrough_keys = ["coords", "coords_original", "pointmap", "pointmap_colors"]

    # Build the transformed prediction dict
    result = {
        "trial_idx": data["trial_idx"],
    }

    # Apply key mappings
    for src_key, dst_key in key_mapping.items():
        if src_key in pred:
            value = pred[src_key]

            # Apply dimension adjustment if needed
            if dst_key in dimension_adjustments:
                adjustment = dimension_adjustments[dst_key]
                if adjustment.startswith("squeeze_"):
                    dim = int(adjustment.split("_")[1])
                    value = value.squeeze(dim)

            result[dst_key] = value

    # Pass through additional keys unchanged
    for key in passthrough_keys:
        if key in pred:
            result[key] = pred[key]

    # Convert instance pose to pose targets if needed
    if convert_instance_pose and pose_convention:
        # Check if we have instance pose components
        has_instance_pose = any(
            k in result
            for k in [
                "x_instance_translation",
                "x_instance_scale",
                "x_instance_rotation",
                "x_instance_rotation_6d_normalized",
            ]
        )

        if has_instance_pose:
            # Add pose convention metadata
            result["pose_target_convention"] = pose_convention
            result["pose_normalize"] = True

            # Optionally convert to pose targets if the convention class is available
            # This would require additional scene information from the sample
            # For now, we just ensure the metadata is correct

    return result
