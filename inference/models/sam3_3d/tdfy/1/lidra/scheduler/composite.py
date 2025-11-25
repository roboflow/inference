import torch
from loguru import logger


class Composite(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        composite_optimizer,
        *schedulers,
        last_epoch=-1,
        verbose=False,
        check_consistency=True
    ):
        self._schedulers = list(schedulers)
        self.optimizer = composite_optimizer
        self._check_consistency = check_consistency

        super().__init__(self.optimizer, last_epoch=last_epoch, verbose=verbose)

    @property
    def schedulers(self):
        return self._schedulers

    def step(self):
        for scheduler in self._schedulers:
            scheduler.step()

    def get_last_lr(self):
        return tuple(sche.get_last_lr() for sche in self._schedulers)

    def get_lr(self):
        return tuple(sche.get_lr() for sche in self._schedulers)

    def state_dict(self):
        return {"_schedulers": self._schedulers}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

        if self._check_consistency:
            sch_ids = set()
            opt_ids = set()
            for sch in self.schedulers:
                sch_ids |= set([id(sch)])

            for opt in self.optimizer._optimizers:
                opt_ids |= set([id(opt)])

            if sch_ids != opt_ids:
                logger.warning(
                    "Incosistency found after reloading optimizer and scheduler. [scheduler.optimizer] is not the same as self.optimizer. Force set the optimizer of scheduler to be the new ones."
                )
                # Also need to update the optimizer
                for opt, sch in zip(self.optimizer._optimizers, self.schedulers):
                    sch.optimizer = opt
