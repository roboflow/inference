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
        # Try to batch compute position embeddings and type-cast all at once
        # Most position_embeddings can be vectorized over all masks in a batch
        # If not, fallback to per-sample loop

        try:
            # Attempt to batch process (common for transformer position embeddings)
            poss_batched = self[1](masks, align_dim_orders=False)
            if isinstance(poss_batched, (list, tuple)):
                # Some implementations return list, convert to list of correct dtype
                poss = [p.to(f.dtype) for p, f in zip(poss_batched, feats)]
            else:
                # Assume batch tensor (B, ...), split and cast all at once
                poss = [p.to(f.dtype) for p, f in zip(poss_batched.unbind(0), feats)]
        except Exception:
            # Fallback: legacy sequential per-sample logic
            poss = []
            for feat, mask in zip(feats, masks):
                poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
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
