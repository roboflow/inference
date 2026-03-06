import os
import torch
from typing import Callable, Any, Dict, List, Tuple, Iterator, Optional
from loguru import logger
import pandas as pd
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from lidra.data.utils import build_batch_extractor
from lidra.data.dataset.return_type import (
    SampleUuidUtils,
    extract_sample_uuid,
    extract_data,
)
from lidra.data.collator import auto_uncollate


class ScorerCallback(Callback):
    def __init__(
        self,
        output_dir: str,
        scorer_fn: Callable,
        uncollate_fn: Optional[Callable] = None,
        name_prefix: str = "scores",
        attempt_merge: bool = True,
        write_every: Optional[int] = None,
    ):
        super().__init__()
        # Callables
        self.scorer_fn = scorer_fn
        self.write_every = write_every

        # Paths
        self.output_dir = Path(output_dir)
        if output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_fstem = name_prefix
        self.output_fpath = self.output_dir / f"{name_prefix}.csv"
        self.uncollate_fn = (
            uncollate_fn if uncollate_fn is not None else auto_uncollate()
        )

        # State
        self.attempt_merge = attempt_merge
        self.stored_metrics = []

    def _compute_per_sample(
        self,
        pl_module: LightningModule,
        outputs: torch.Tensor | Dict[str, Any] | None,
        batch: Any,
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        sample_uuids = extract_sample_uuid(batch)
        uncollated_sample_uuids = self.uncollate_fn(sample_uuids)

        metrics = self.scorer_fn(pl_module, batch)
        metrics = self.uncollate_fn(metrics)
        for example_uuid, metrics_dict in zip(uncollated_sample_uuids, metrics):
            example_uuid = SampleUuidUtils.demote(example_uuid)
            yield example_uuid, metrics_dict

    def compute_per_sample(
        self,
        pl_module: LightningModule,
        outputs: torch.Tensor | Dict[str, Any] | None,
        batch: Any,
    ) -> pd.DataFrame:
        """Returns a dataframe of metrics per sample"""
        stored_metrics = []
        for example_uuid, metrics_dict in self._compute_per_sample(
            pl_module, outputs, batch
        ):
            stored_metrics.append((example_uuid, metrics_dict))
        return self._make_metrics_df(stored_metrics)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for example_uuid, metrics_dict in self._compute_per_sample(
            pl_module, outputs, batch
        ):
            self.stored_metrics.append((example_uuid, metrics_dict))

        if batch_idx % self.write_every == 0:
            metrics_df = self._make_metrics_df(self.stored_metrics)
            metrics_df.to_csv(self._rank_to_filepath(trainer.global_rank), index=False)

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


class LogLikelihoodScorer(ScorerCallback):
    def __init__(
        self, scorer_fn: Callable, remove_predict_step: bool = False, **kwargs
    ):
        if scorer_fn is None:
            scorer_fn = self.compute_log_likelihood
        self.remove_predict_step = remove_predict_step
        super().__init__(scorer_fn=scorer_fn, **kwargs)

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str):
        super().setup(trainer, pl_module, stage)
        if self.remove_predict_step:
            logger.warning("Removing predict_step from pl_module; it will return None")
            pl_module.predict_step = constant

    @staticmethod
    def compute_log_likelihood(module, batch):
        with torch.autograd.grad_mode.inference_mode(mode=False):
            ll = module.log_likelihood(batch, 0)
        return {"log_likelihood": [_ll.item() for _ll in ll]}


def constant(*args, **kwargs):
    return None
