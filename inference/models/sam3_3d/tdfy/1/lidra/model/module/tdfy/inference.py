"""
InferencePipeline wrapper for Lidra module interface.

This module provides a minimal wrapper to adapt the standalone InferencePipeline
to work within Lidra's PyTorch Lightning-based infrastructure.
"""

from typing import Optional, Dict, Any, List, Union, Callable
import torch
from loguru import logger

from lidra.model.module.base import Base, FrozenBackbone
from lidra.data.collator import auto_uncollate, auto_collate
from lidra.data.dataset.return_type import extract_data, extract_sample_uuid


class InferenceWrapper(Base):
    """
    Minimal wrapper to adapt InferencePipeline to Lidra's module interface.

    This module serves as a bridge between the standalone InferencePipeline
    and Lidra's Lightning-based infrastructure, enabling:
    - Distributed batch processing via SLURM
    - Integration with Lidra callbacks (metrics, caching, visualization)
    - Consistent configuration through Hydra

    The wrapper follows composition over inheritance, taking an instantiated
    pipeline as input rather than creating one internally. This maintains
    separation of concerns and allows the pipeline to be configured independently.
    """

    def __init__(
        self,
        pipeline: Any,  # The instantiated InferencePipeline
        input_preprocessor: Optional[Callable] = None,
        output_postprocessor: Optional[Callable] = None,
        max_trials: int = 1,  # Number of inference runs per sample
        uncollate_fn: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the InferenceWrapper.

        Args:
            pipeline: Pre-configured InferencePipeline instance
            input_preprocessor: Optional function to preprocess inputs before pipeline
                               Signature: (image, mask, **kwargs) -> dict
            output_postprocessor: Optional function to postprocess pipeline outputs
                                 Signature: (output_dict) -> dict
            max_trials: Number of times to run inference per sample (for stochastic models)
            uncollate_fn: Function to uncollate batched inputs into individual samples
            inference_kwargs: Additional keyword arguments to pass to pipeline.run()
            **kwargs: Additional arguments passed to Base class
        """
        # Wrap pipeline models as FrozenBackbone, filtering out None values
        # Some models (like layout_model) might be None if not configured
        models = {
            k: FrozenBackbone(v, None)
            for k, v in pipeline.models.items()
            if v is not None
        }
        if not models:
            raise ValueError("Pipeline must have at least one non-None model")
        super().__init__(models, **kwargs)

        self._pipeline = pipeline

        self.input_preprocessor = input_preprocessor or self._default_preprocessor
        self.output_postprocessor = output_postprocessor or (
            lambda x: x
        )  # self._default_postprocessor
        self.max_trials = max_trials
        self.inference_kwargs = inference_kwargs or {}

        self.uncollate_fn = uncollate_fn or auto_uncollate()
        self.collate_fn = collate_fn or auto_collate()

        self.configure_model()
        logger.info(f"InferenceWrapper initialized with max_trials={max_trials}")

    @staticmethod
    def _default_preprocessor(batch, include_pointmap=False):
        batch = extract_data(batch)
        return_dict = {}

        image = batch["rgb_image"].cpu()
        image = image.permute(0, 2, 3, 1)
        image = (image * 255.0).to(torch.uint8)
        return_dict["image"] = image

        mask = batch["rgb_image_mask"].cpu()
        mask = mask.squeeze(1)
        mask = (mask * 255.0).to(torch.uint8)
        return_dict["mask"] = mask

        if include_pointmap:
            pointmap = batch["rgb_pointmap"].cpu()
            pointmap = pointmap.permute(0, 2, 3, 1)
            return_dict["pointmap"] = pointmap

        return return_dict

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Run inference pipeline on a batch of inputs.

        This method is called by PyTorch Lightning during the prediction loop.
        It handles the standard Lidra batch format and runs the pipeline on each sample.

        Args:
            batch: Tuple of (sample_uuids, inputs_dict) where:
                   - sample_uuids: List of sample identifiers
                   - inputs_dict: Dict of batched tensors
            batch_idx: Current batch index
            dataloader_idx: Dataloader index (for multiple dataloaders)

        Returns:
            For max_trials == 1: List of prediction dicts
            For max_trials > 1: List of lists (trials) of prediction dicts
        """
        inputs = self.input_preprocessor(batch)
        inputs = self.uncollate_fn(inputs)

        # Process each sample
        all_predictions = []
        for i in range(self.max_trials):
            trial_predictions = []
            for processed_inputs in inputs:
                sample_predictions = self._run_single_trial(
                    processed_inputs,
                    seed=i,
                )
                trial_predictions.append(sample_predictions)

            trial_predictions = self.collate_fn(trial_predictions)
            all_predictions.append(trial_predictions)

        return all_predictions

    def _run_single_trial(self, inputs, seed=None):
        """Run a single inference trial for a single sample."""
        pipeline_kwargs = {**inputs, **self.inference_kwargs}
        pipeline_kwargs["seed"] = seed
        output = self._pipeline.run(**pipeline_kwargs)
        output = self.output_postprocessor(output)
        return output

    def forward(self, *args, **kwargs):
        """
        Forward pass (not used for inference-only module).

        This method is required by the Base class but not used for inference.
        """
        raise NotImplementedError(
            "InferenceWrapper is for prediction only, not training"
        )

    def training_step(self, batch, batch_idx):
        """Training step (not supported)."""
        raise NotImplementedError("InferenceWrapper does not support training")

    def validation_step(self, batch, batch_idx):
        """Validation step (could be implemented to run predict_step)."""
        return self.predict_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        """Test step (could be implemented to run predict_step)."""
        return self.predict_step(batch, batch_idx)
