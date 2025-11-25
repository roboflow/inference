import torch
import math


class MCCScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        min_factor,
        max_factor=1.0,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_factor = min_factor
        self.max_factor = max_factor

        super().__init__(
            optimizer,
            lr_lambda=self.lr_update,
            last_epoch=last_epoch,
            verbose=False,
        )

    def lr_update(self, epoch):
        if epoch < self.warmup_epochs:
            factor = epoch / self.warmup_epochs
        else:
            t_factor = (epoch - self.warmup_epochs) / (
                self.max_epochs - 1 - self.warmup_epochs
            )
            cos_factor = math.cos(math.pi * t_factor)
            factor = (1 + cos_factor) / 2
        return self.min_factor + (self.max_factor - self.min_factor) * factor
