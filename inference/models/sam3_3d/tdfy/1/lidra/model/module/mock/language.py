import torch

from lidra.model.module.base import Base, TrainableBackbone


class Language(Base):
    def __init__(self, model: TrainableBackbone, **kwargs):
        super().__init__(model, **kwargs)

        self._criterion = torch.nn.CrossEntropyLoss()

    # hugging face references
    # https://github.com/huggingface/transformers/blob/2cc8cf6ce7ae0416561acbb639df4bbc5f409b6f/src/transformers/trainer.py#L2876
    # https://github.com/huggingface/transformers/blob/2cc8cf6ce7ae0416561acbb639df4bbc5f409b6f/src/transformers/trainer.py#L2915
    def training_step(self, batch, batch_idx):
        if not "labels" in batch:
            batch["labels"] = batch["input_ids"]

        # TODO(Pierre) : loss computation should be exported out of the model
        # remark : loss computation disabled when "label" isn't present in batch
        # c.f. LabelSmoother from HuggingFace
        outputs = self.base_model(**batch)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        self.log("loss", loss, prog_bar=True)

        return loss
