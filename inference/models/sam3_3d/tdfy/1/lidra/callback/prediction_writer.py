import os
import torch
from loguru import logger

from lightning.pytorch.callbacks import BasePredictionWriter


class PredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, attempt_merge=True):
        super().__init__(write_interval="epoch")
        self.output_dir = output_dir
        self.attempt_merge = attempt_merge
        os.makedirs(output_dir, exist_ok=True)

    def _rank_to_filepath(self, rank):
        return os.path.join(
            self.output_dir,
            f"predictions_{rank}.pt",
        )

    def _merge(self, predictions):
        # TODO(Pierre) : handles more generic use cases (other than lists)
        return sum(predictions, start=[])

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = self._merge(predictions)
        filepath = self._rank_to_filepath(trainer.global_rank)
        torch.save(predictions, filepath)

        if self.attempt_merge:
            trainer.strategy.barrier(
                "PredictionWriter.write_on_epoch_end"
            )  # wait for all processes

            if trainer.global_rank == 0:
                # some of those files might not be local, this is why we "attempt"
                try:
                    all_data = [
                        torch.load(self._rank_to_filepath(rank))
                        for rank in range(trainer.world_size)
                    ]
                    all_data = self._merge(all_data)
                    torch.save(all_data, self._rank_to_filepath("all"))
                except:
                    logger.opt(exception=True).warning(
                        f"could not merge prediction files"
                    )
