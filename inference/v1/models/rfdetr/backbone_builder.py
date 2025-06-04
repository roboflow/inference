from typing import Callable

import torch
from torch import nn

from inference.v1.models.rfdetr.misc import NestedTensor
from inference.v1.models.rfdetr.position_encoding import build_position_encoding
from inference.v1.models.rfdetr.rfdetr_backbone_pytorch import Backbone


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        """ """
        x = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
        for name, m in self.named_modules():
            if (
                hasattr(m, "export")
                and isinstance(m.export, Callable)
                and hasattr(m, "_export")
                and not m._export
            ):
                m.export()

    def forward_export(self, inputs: torch.Tensor):
        feats, masks = self[0](inputs)

        # Optimization: Try batch-processing masks if possible
        # If feats and masks are lists/tuples of tensors, stack them if feasible
        if isinstance(masks, (list, tuple)):
            try:
                # Try to stack masks for batch processing
                masks_batch = torch.stack(masks, dim=0)
                poss_batch = self[1](masks_batch, align_dim_orders=False)
                if not isinstance(poss_batch, torch.Tensor):
                    # Fallback to original for-loop if output is not batched
                    raise Exception("Output is not a batch")
                # If position embedding returns a single tensor for batch
                # Convert once to feats[0].dtype and split accordingly
                poss_batch = poss_batch.to(feats[0].dtype)
                poss = [p for p in poss_batch]
            except Exception:
                # If stacking or batch processing fails, fallback to for-loop
                poss = [
                    self[1](mask, align_dim_orders=False).to(feat.dtype)
                    for feat, mask in zip(feats, masks)
                ]
        else:
            # Not a sequence, fallback to original logic
            poss = [
                self[1](masks, align_dim_orders=False).to(feats.dtype) for feat in feats
            ]
        return feats, None, poss


def build_backbone(
    encoder,
    vit_encoder_num_layers,
    pretrained_encoder,
    window_block_indexes,
    drop_path,
    out_channels,
    out_feature_indexes,
    projector_scale,
    use_cls_token,
    hidden_dim,
    position_embedding,
    freeze_encoder,
    layer_norm,
    target_shape,
    rms_norm,
    backbone_lora,
    force_no_pretrain,
    gradient_checkpointing,
    load_dinov2_weights,
):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(hidden_dim, position_embedding)

    backbone = Backbone(
        encoder,
        pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights,
    )

    model = Joiner(backbone, position_embedding)
    return model
