import torch

from lidra.model.module.base import Base, TrainableBackbone


class Classification(Base):
    def __init__(self, model: TrainableBackbone, **kwargs):
        super().__init__(model, **kwargs)

        self._criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.base_model(x)
        loss = self._criterion(y_hat, y)

        self.log("loss", loss, prog_bar=True)

        return loss
