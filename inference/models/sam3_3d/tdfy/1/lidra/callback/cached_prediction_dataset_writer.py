import os
from pathlib import Path
import torch
from typing import List, Any, Tuple, Sequence, Optional, Callable, Union, Dict
from lightning.pytorch.callbacks import BasePredictionWriter
from loguru import logger

from lidra.data.collator import auto_uncollate, auto_collate
from lidra.data.dataset.tdfy.with_cached_predictions_dataset import (
    WithCachedPredictionDataset,
)

from collections import Counter


class CachedPredictionDatasetWriter(BasePredictionWriter):
    """Lightning callback that writes predictions to disk in a format compatible with DatasetWithCachedPrediction.

    Each prediction is saved as a .pt file containing a tuple of (example_uuid, prediction_dict).
    """

    def __init__(
        self,
        output_dir: str,
        uncollate_fn: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        write_interval: str = "batch",
        pickle_module: Optional[str] = None,  # 'dill' or None for default pickle
    ):
        super().__init__(write_interval=write_interval)
        assert write_interval == "batch"

        self.uncollate_fn = uncollate_fn
        if self.uncollate_fn is None:
            self.uncollate_fn = auto_uncollate()

        self.collate_fn = collate_fn
        if self.collate_fn is None:
            self.collate_fn = auto_collate()

        self.write_interval = write_interval
        self.output_dir = Path(output_dir)

        # Setup pickle module
        self.pickle_module = None
        if pickle_module == "dill":
            import dill  # Will raise ImportError if not available

            self.pickle_module = dill

        logger.info(
            f"Writing predictions to {self.output_dir}"
            + (f" using {pickle_module}" if pickle_module else "")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_trials = Counter()

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction: Union[List, Dict],
        batch_indices: List[int],
        batch: Tuple[Sequence[Any], Sequence[Any]],
        batch_idx: int,
        dataloader_idx: int,
    ):
        if self.write_interval != "batch":
            return

        sample_uuids, inputs = batch
        sample_uuids = self.uncollate_fn(sample_uuids)

        if isinstance(prediction, (tuple, list)):
            sample_uuids, prediction = (
                CachedPredictionDatasetWriter.reshape_wide_to_long(
                    sample_uuids, prediction
                )
            )

        for sample_uuid, sample_pred in zip(sample_uuids, prediction):
            self.n_trials[sample_uuid] += 1
            trial_num = self.n_trials[sample_uuid]
            fpath = WithCachedPredictionDataset.cache_prediction(
                self.output_dir,
                sample_uuid,
                trial_num,
                sample_pred,
                pickle_module=self.pickle_module,
            )

    def _make_example_fpath(self, example_uuid, trial_num):
        fname = WithCachedPredictionDataset.make_safe_filename(example_uuid, trial_num)
        fpath = self.output_dir / fname
        return fpath

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions: List[List[tuple[Any, dict]]],
        batch_indices: List[List[int]],
    ):
        """Write predictions at the end of each epoch.

        Args:
            predictions: List of lists of (example_uuid, prediction_dict) tuples
            batch_indices: List of lists of batch indices
        """
        if self.write_interval != "epoch":
            self.n_trials.clear()
            return

        raise NotImplementedError("Epoch writing not implemented")

    @staticmethod
    def reshape_wide_to_long(sample_uuids, data_all_steps):
        """
        Reshapes data from wide format (multiple steps) to long format (flattened).

        Args:
            sample_uuids: List of unique identifiers for samples
            data_all_steps: List of outputs for each step, where each element contains a list of outputs for hat sample

        Returns:
            tuple: (repeated sample_uuids matching the flattened outputs, flattened outputs from all steps)

        Example:
            >>> sample_uuids = ['uuid1', 'uuid2']
            >>> n_uuids = len(sample_uuids)
            >>> data_all_steps = [
            ...     [{
            ...        "key1": tensor: [n_uuids, ...]
            ...        "key2": tensor: [n_uuids, ...]
            ...      },
            ...      ...
            ...      {
            ...        "key1": tensor: [n_uuids, ...]
            ...        "key2": tensor: [n_uuids, ...]
            ...      },
            ... ]
             >>> uuids, outputs = reshape_wide_to_long(sample_uuids, data_all_steps)
            >>> uuids # repeated sample_uuids matching the flattened outputs
            ['uuid1', 'uuid2', 'uuid1', 'uuid2']
            >>> outputs
            [{'key1': tensor: [...], 'key2': tensor: [...]}, ...] # len(outputs) == n_uuids * n_steps
        """
        assert isinstance(
            data_all_steps, list
        ), f"data_all_steps is not a list: {type(data_all_steps)}"

        # If data_all_steps is a list of dictionaries with tensors
        if all(isinstance(step_data, dict) for step_data in data_all_steps):
            n_steps = len(data_all_steps)
            n_samples = len(sample_uuids)
            long_outputs = []

            # Process each dictionary and extract per-sample outputs
            for step_dict in data_all_steps:
                # For each sample in this step
                for i in range(n_samples):
                    sample_dict = {}
                    # Extract the i-th sample from each tensor in the dict
                    for key, tensor in step_dict.items():
                        sample_dict[key] = tensor[i]
                    long_outputs.append(sample_dict)

            # Repeat sample_uuids to match the new format (uuids are arranged as uuid1, uuid2, uuid1, uuid2, ...)
            repeated_uuids = []
            for _ in range(n_steps):
                repeated_uuids.extend(sample_uuids)

            assert len(repeated_uuids) == len(long_outputs)
            return repeated_uuids, long_outputs

        # Original implementation for list of lists
        sample_uuids = sample_uuids * len(data_all_steps)
        long_outputs = []
        for step_outputs in data_all_steps:
            long_outputs.extend(step_outputs)

        assert len(sample_uuids) == len(long_outputs)
        return sample_uuids, long_outputs

    @staticmethod
    def reshape_long_to_wide(x):
        raise NotImplementedError("Not implemented")
