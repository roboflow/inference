import torch

from lidra.model.module.base import Base, TrainableBackbone


class AutoEncoder(Base):
    def __init__(
        self,
        encoder: TrainableBackbone,
        decoder: TrainableBackbone,
        regularization_lambda: float = 0.01,
        **kwargs,
    ):
        super().__init__({"encoder": encoder, "decoder": decoder}, **kwargs)

        self._regularization_lambda = regularization_lambda
        self._regression_criterion = torch.nn.MSELoss()
        self._regularization_criterion = torch.nn.L1Loss()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        h = self.base_models["encoder"](x)
        x_hat = self.base_models["decoder"](h)

        _0 = torch.tensor(0.0, dtype=h.dtype, device=h.device)
        regression_term = self._regression_criterion(x_hat, x)
        regularization_term = self._regularization_criterion(h, _0)
        loss = regression_term + self._regularization_lambda * regularization_term

        self.log("regression", regression_term, prog_bar=True)
        self.log("regularization", regularization_term)
        self.log("loss", loss, prog_bar=True)

        return loss
