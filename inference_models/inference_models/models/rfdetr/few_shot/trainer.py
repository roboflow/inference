"""LoRA fine-tuning trainer for RF-DETR few-shot object detection."""

import copy
import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from inference_models.models.rfdetr.few_shot.criterion import SetCriterion
from inference_models.models.rfdetr.few_shot.dataset import FewShotDataset, collate_fn
from inference_models.models.rfdetr.few_shot.matcher import HungarianMatcher
from inference_models.models.rfdetr.rfdetr_base_pytorch import (
    LWDETR,
    ModelConfig,
    build_model,
)

logger = logging.getLogger(__name__)

# Default LoRA target modules for RF-DETR
BACKBONE_LORA_TARGETS = ["query", "value"]
DECODER_LORA_TARGETS = ["self_attn.out_proj", "linear1", "linear2"]


class RFDETRFewShotTrainer:
    """Trains a LoRA adapter on top of an RF-DETR model for few-shot detection.

    The trainer:
    1. Clones the base model weights
    2. Reinitialises the detection head for the new class set
    3. Applies LoRA to backbone + decoder attention layers
    4. Trains for ``num_epochs`` using DETR loss (focal + L1 + GIoU)
    5. Returns the merged model + adapter state for caching
    """

    def __init__(
        self,
        device: torch.device,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        learning_rate: float = 2e-3,
        num_epochs: int = 15,
    ):
        self.device = device
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def train(
        self,
        base_model: LWDETR,
        config: ModelConfig,
        training_data: list,
        class_names: List[str],
    ) -> Tuple[LWDETR, Dict]:
        """Run LoRA fine-tuning and return (merged_model, adapter_state).

        Args:
            base_model: Pretrained LWDETR model (will NOT be mutated).
            config: Model config for the variant being used.
            training_data: List of training image dicts with boxes.
            class_names: Ordered list of class names for this few-shot task.

        Returns:
            (merged_model, adapter_state) where merged_model is eval-mode
            LWDETR with LoRA merged in, and adapter_state contains the
            LoRA adapter weights + head weights + class names for caching.
        """
        from peft import LoraConfig, get_peft_model

        num_classes = len(class_names)

        # Build a fresh model from config with the right number of classes
        fresh_config = config.model_copy(update={"num_classes": num_classes})
        model = build_model(config=fresh_config)

        # Load base weights, then reinitialise head for new class count
        base_state = base_model.state_dict()
        # Filter out mismatched keys (class_embed, enc_out_class_embed)
        model_state = model.state_dict()
        filtered_state = {
            k: v
            for k, v in base_state.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(filtered_state, strict=False)
        model.reinitialize_detection_head(num_classes + 1)
        model = model.to(self.device)

        # Apply LoRA to backbone + decoder
        all_targets = BACKBONE_LORA_TARGETS + DECODER_LORA_TARGETS
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.0,
            target_modules=all_targets,
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)

        # Make detection head fully trainable
        for name, param in peft_model.named_parameters():
            if "class_embed" in name or "bbox_embed" in name:
                param.requires_grad = True
            if "enc_out_class_embed" in name or "enc_out_bbox_embed" in name:
                param.requires_grad = True

        trainable_count = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in peft_model.parameters())
        logger.info(
            "LoRA trainable params: %d / %d (%.2f%%)",
            trainable_count,
            total_count,
            100.0 * trainable_count / total_count,
        )

        # Build dataset + dataloader
        dataset = FewShotDataset(
            training_data=training_data,
            class_names=class_names,
            resolution=config.resolution,
            augment=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=min(len(dataset), 4),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Build loss
        matcher = HungarianMatcher(
            cost_class=2, cost_bbox=5, cost_giou=2, focal_alpha=0.25
        )
        weight_dict = {
            "loss_ce": 2,
            "loss_bbox": 5,
            "loss_giou": 2,
        }
        # Aux losses for each decoder layer
        num_aux = fresh_config.dec_layers - 1
        aux_weight_dict = {}
        for i in range(num_aux):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

        criterion = SetCriterion(
            num_classes=num_classes + 1,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
            group_detr=getattr(config, "group_detr", 1),
            ia_bce_loss=getattr(config, "ia_bce_loss", True),
        )
        criterion = criterion.to(self.device)

        # Optimizer + cosine annealing schedule
        optimizer = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=self.learning_rate * 0.01,
        )

        # Training loop
        peft_model.train()
        criterion.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for images, targets in dataloader:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    outputs = peft_model(images)
                    loss_dict = criterion(outputs, targets)
                    losses = sum(
                        loss_dict[k] * weight_dict[k]
                        for k in loss_dict
                        if k in weight_dict
                    )

                losses.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                epoch_loss += losses.item()

            scheduler.step()
            logger.info("Epoch %d/%d, loss: %.4f", epoch + 1, self.num_epochs, epoch_loss)

        # Save adapter state before merging
        adapter_state = {
            "lora_state_dict": {
                k: v.cpu()
                for k, v in peft_model.state_dict().items()
                if "lora_" in k
            },
            "head_state_dict": {
                "class_embed": {
                    k: v.cpu()
                    for k, v in peft_model.base_model.model.class_embed.state_dict().items()
                },
                "bbox_embed": {
                    k: v.cpu()
                    for k, v in peft_model.base_model.model.bbox_embed.state_dict().items()
                },
            },
            "class_names": class_names,
            "num_classes": num_classes,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
        }

        # Also save enc_out_class_embed if two_stage
        if hasattr(peft_model.base_model.model, "transformer") and hasattr(
            peft_model.base_model.model.transformer, "enc_out_class_embed"
        ):
            adapter_state["head_state_dict"]["enc_out_class_embed"] = {
                k: v.cpu()
                for k, v in peft_model.base_model.model.transformer.enc_out_class_embed.state_dict().items()
            }

        # Merge LoRA weights for fast inference
        merged_model = peft_model.merge_and_unload()
        merged_model = merged_model.eval()

        return merged_model, adapter_state
