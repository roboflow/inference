"""
This file contains the implementation of a stateless Sam3ImageModel and a corresponding 
model builder function. This is part of a refactoring to separate the core model 
from the interactive session logic.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Union, Dict, Optional, List, Tuple

# Imports from the original sam3 library, ensuring no modifications to original code
from sam3.model.decoder import TransformerDecoder, TransformerDecoderLayer
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import FusedMaskEncoder, SequenceGeometryEncoder, Prompt
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import CXBlock, SimpleFuser, SimpleMaskDownSampler
from sam3.model.model_misc import DotProductScoring, MLP, TransformerWrapper, NestedTensor, inverse_sigmoid
from sam3.model.necks import OriginalViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import NonFusionVLBackbone
from sam3.model.box_ops import box_cxcywh_to_xyxy, box_xywh_to_cxcywh, box_xyxy_to_xywh
from torch.nn import MultiheadAttention


def build_sam3_model(
    bpe_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    image_size=1008,
    image_mean=(0.5, 0.5, 0.5),
    image_std=(0.5, 0.5, 0.5),
):
    """
    This function builds the Sam3ImageModel by constructing its components.
    It is an adaptation of the original build_sam3_image_model from the sam3 library,
    designed to be self-contained and instantiate our new stateless model.
    """
    # Create position encoding for visual backbone
    position_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)
    # Create ViT backbone
    vit_backbone = ViT(img_size=1008, pretrain_img_size=336, patch_size=14, embed_dim=1024, depth=32, num_heads=16, mlp_ratio=4.625, norm_layer="LayerNorm", drop_path_rate=0.1, qkv_bias=True, use_abs_pos=True, tile_abs_pos=True, global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(), use_rope=True, use_interp_rope=True, window_size=24, pretrain_use_cls_token=True, retain_cls_token=False, ln_pre=True, ln_post=False, return_interm_layers=False, bias_patch_embed=False, compile_mode="default")
    # Create ViT neck
    vit_neck = OriginalViTDetNeck(position_encoding=position_encoding, d_model=256, scale_factors=[4.0, 2.0, 1.0, 0.5], trunk=vit_backbone)
    # Create text tokenizer & encoder
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    text_encoder = VETextEncoder(tokenizer=tokenizer, d_model=256, width=1024, heads=16, layers=24)
    # Create visual-language backbone
    backbone = NonFusionVLBackbone(visual=vit_neck, text=text_encoder, scalp=1)
    # Create transformer encoder
    encoder_layer = TransformerEncoderLayer(activation="relu", d_model=256, dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=True, pos_enc_at_cross_attn_keys=False, pos_enc_at_cross_attn_queries=False, pre_norm=True, self_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, batch_first=True), cross_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, batch_first=True))
    encoder = TransformerEncoderFusion(layer=encoder_layer, num_layers=6, d_model=256, num_feature_levels=1, frozen=False, use_act_checkpoint=True, add_pooled_text_to_img_feat=False, pool_text_with_mask=True)
    # Create transformer decoder
    decoder_layer = TransformerDecoderLayer(activation="relu", d_model=256, dim_feedforward=2048, dropout=0.1, cross_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256), n_heads=8, use_text_cross_attention=True)
    decoder = TransformerDecoder(layer=decoder_layer, num_layers=6, num_queries=200, return_intermediate=True, box_refine=True, num_o2m_queries=0, dac=True, boxRPB="log", d_model=256, frozen=False, interaction_layer=None, dac_use_selfatt_ln=True, use_act_checkpoint=True, instance_query=True, num_instances=4)
    # Create transformer
    transformer = TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)
    # Create dot product scorer
    prompt_mlp = MLP(input_dim=256, hidden_dim=2048, output_dim=256, num_layers=2, dropout=0.1, residual=True, out_norm=nn.LayerNorm(256))
    dot_prod_scoring = DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)
    # Create segmentation head
    pixel_decoder = PixelDecoder(num_upsampling_stages=3, interpolation_mode="nearest", hidden_dim=256, compile_mode="default")
    cross_attend_prompt = MultiheadAttention(num_heads=8, dropout=0, embed_dim=256)
    segmentation_head = UniversalSegmentationHead(hidden_dim=256, upsampling_stages=3, aux_masks=False, presence_head=True, dot_product_scorer=DotProductScoring(d_model=256,d_proj=256,prompt_mlp=MLP(input_dim=256,hidden_dim=2048,output_dim=256,num_layers=2,dropout=0.1,residual=True,out_norm=nn.LayerNorm(256),),),act_ckpt=True,cross_attend_prompt=cross_attend_prompt,pixel_decoder=pixel_decoder)
    # Create geometry encoder
    geo_pos_enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)
    mask_downsampler = SimpleMaskDownSampler(interpol_size=[288, 288], kernel_size=3, stride=2, padding=1, total_stride=4)
    cx_block = CXBlock(dim=256, kernel_size=7, padding=3, layer_scale_init_value=1.0e-06, use_dwconv=True)
    fuser = SimpleFuser(layer=cx_block, num_layers=2)
    mask_encoder = FusedMaskEncoder(out_dim=256, position_encoding=PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000, precompute_resolution=1008), mask_downsampler=mask_downsampler, fuser=fuser)
    geo_layer = TransformerEncoderLayer(activation="relu", d_model=256, dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False, pre_norm=True, self_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, batch_first=False), pos_enc_at_cross_attn_queries=False, pos_enc_at_cross_attn_keys=True, cross_attention=MultiheadAttention(num_heads=8, dropout=0.1, embed_dim=256, batch_first=False))
    input_geometry_encoder = SequenceGeometryEncoder(pos_enc=geo_pos_enc, encode_boxes_as_points=False, points_direct_project=True, points_pool=True, points_pos_enc=True, boxes_direct_project=True, boxes_pool=True, boxes_pos_enc=True, d_model=256, num_layers=3, layer=geo_layer, use_act_ckpt=True, add_cls=True, add_post_encode_proj=True, mask_encoder=mask_encoder)

    model = Sam3ImageModel(
        backbone=backbone,
        transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head,
        dot_prod_scoring=dot_prod_scoring,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std
    )

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt, strict=True)

    model.to(device)
    if eval_mode:
        model.eval()

    return model

class Sam3ImageModel(nn.Module):
    """A stateless SAM3 model for single image segmentation."""
    def __init__(
        self,
        backbone: nn.Module,
        transformer: nn.Module,
        input_geometry_encoder: nn.Module,
        segmentation_head: nn.Module,
        dot_prod_scoring: nn.Module,
        image_size: int = 1008,
        image_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.geometry_encoder = input_geometry_encoder
        self.segmentation_head = segmentation_head
        self.dot_prod_scoring = dot_prod_scoring
        self.image_size = image_size
        self.image_mean = torch.tensor(image_mean).view(3, 1, 1)
        self.image_std = torch.tensor(image_std).view(3, 1, 1)
        self.num_feature_levels = 1 # From original builder
        self.use_instance_query = True # From original builder
        
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def preprocess_image(
        self, image: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocesses the input image for the model.
        Accepts a NumPy array (H, W, C) or a PyTorch tensor (C, H, W).
        Returns the processed image tensor and the original image size.
        """
        if isinstance(image, np.ndarray):
            # Convert HWC, uint8 [0, 255] to CHW, float32 [0, 1]
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        image = image.to(self.device)
        original_size = image.shape[-2:]
        
        # Resize
        resized_image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        # Normalize
        processed_image = (resized_image - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        
        return processed_image, original_size

    @torch.inference_mode()
    def encode_image(self, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encodes the image using the vision backbone."""
        image_tensor = image_tensor.unsqueeze(0) # Create a batch of 1
        image_nested_tensor = NestedTensor(tensors=image_tensor, mask=None)
        image_features = self.backbone.forward_image(image_nested_tensor)
        return {
            "backbone_fpn": image_features["backbone_fpn"],
            "vision_pos_enc": image_features["vision_pos_enc"]
        }

    @torch.inference_mode()
    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Encodes a text string."""
        return self.backbone.forward_text([text], device=self.device)

    def _get_img_feats(self, backbone_out: Dict) -> Tuple:
        """Helper to extract features from backbone output."""
        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]
        img_feats = [x.tensors.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_masks = [x.mask.flatten(1) if x.mask is not None else None for x in vis_feats]
        img_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vis_pos_enc]
        return img_feats, img_masks, img_pos_embeds, vis_feat_sizes

    def _init_instance_queries(self, B: int, multimask_output: bool = False) -> Dict:
        """Initializes queries for a single image, equivalent to the first frame of a video."""
        if self.use_instance_query:
            query_embed = self.transformer.decoder.instance_query_embed.weight
            query_embed = query_embed[1:] if multimask_output else query_embed[:1]
            reference_boxes = self.transformer.decoder.instance_reference_points.weight
            reference_boxes = reference_boxes[1:] if multimask_output else reference_boxes[:1]
        else:
            query_embed = self.transformer.decoder.query_embed.weight
            reference_boxes = self.transformer.decoder.reference_points.weight
        
        reference_boxes = reference_boxes.unsqueeze(1).expand(-1, B, -1).sigmoid()
        tgt = query_embed.unsqueeze(1).expand(-1, B, -1)
        
        return {"embed": tgt, "reference_boxes": reference_boxes}

    @torch.inference_mode()
    def predict(
        self,
        image_features: Dict,
        text_features: Optional[Dict],
        geometric_prompt: Prompt,
        visual_prompt: Optional[Prompt] = None,
        multimask_output: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        The core stateless forward pass for single image segmentation.
        """
        img_feats, img_masks, img_pos_embeds, vis_feat_sizes = self._get_img_feats(image_features)

        # 1. Encode all prompts
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=geometric_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        visual_prompt_embed = torch.zeros((0, geo_feats.shape[1], geo_feats.shape[2]), device=self.device)
        visual_prompt_mask = torch.zeros((geo_masks.shape[0], 0), device=self.device, dtype=torch.bool)
        if visual_prompt is not None:
             visual_prompt_embed, visual_prompt_mask = self.geometry_encoder(
                 geo_prompt=visual_prompt,
                 img_feats=img_feats,
                 img_sizes=vis_feat_sizes,
                 img_pos_embeds=img_pos_embeds,
             )

        prompt_list = [geo_feats, visual_prompt_embed]
        prompt_mask_list = [geo_masks, visual_prompt_mask]
        
        if text_features:
            prompt_list.insert(0, text_features["language_features"].permute(1,0,2))
            prompt_mask_list.insert(0, text_features["language_mask"])
            
        prompt = torch.cat(prompt_list, dim=0)
        prompt_mask = torch.cat(prompt_mask_list, dim=1)
        prompt_pos_embed = torch.zeros_like(prompt)

        # 2. Run Transformer Encoder
        memory = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=img_masks.copy(),
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )
        encoder_hidden_states = memory["memory"]
        pos_embed = memory["pos_embed"]
        src_mask = memory["padding_mask"]

        # 3. Initialize queries for decoder
        B = encoder_hidden_states.size(1)
        tracking_queries = self._init_instance_queries(B, multimask_output)
        
        # 4. Run Transformer Decoder
        hs, reference_boxes = self.transformer.decoder(
            tgt=tracking_queries["embed"],
            memory=encoder_hidden_states,
            memory_key_padding_mask=src_mask,
            pos=pos_embed,
            reference_boxes=tracking_queries["reference_boxes"],
            level_start_index=memory["level_start_index"],
            spatial_shapes=memory["spatial_shapes"],
            valid_ratios=memory["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)

        # 5. Get scores and boxes
        # These heads expect the full stack of decoder layer outputs (4D tensors)
        assert hs.ndim == 4, f"Expecting 4D hs tensor from decoder, got {hs.ndim}"
        outputs_class = self.dot_prod_scoring(hs, prompt, prompt_mask)
        anchor_box_offsets = self.transformer.decoder.bbox_embed(hs)
        outputs_coord = (inverse_sigmoid(reference_boxes) + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        out = {
            "pred_logits": outputs_class,
            "pred_boxes": outputs_coord,
            "pred_boxes_xyxy": outputs_boxes_xyxy,
        }

        # 6. Run Segmentation Head
        # This head also expects the full 4D hs tensor, but only uses the last layer internally
        seg_head_outputs = self.segmentation_head(
            backbone_feats=image_features["backbone_fpn"],
            obj_queries=hs,
            image_ids=torch.arange(B, device=self.device),
            encoder_hidden_states=encoder_hidden_states,
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        out.update(seg_head_outputs)

        return out

    @torch.no_grad()
    def postprocess_outputs(
        self,
        model_outputs: Dict,
        original_size: Tuple[int, int],
        output_prob_thresh: float = 0.5,
        multimask_output: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Post-processes the model's raw outputs."""
        # Select outputs from the final decoder layer
        out_logits = model_outputs["pred_logits"][-1]
        out_boxes_xyxy = model_outputs["pred_boxes_xyxy"][-1]
        out_masks = model_outputs["pred_masks"]

        out_probs = out_logits.sigmoid()
        
        if not self.training and multimask_output and out_probs.shape[1] > 1:
            # Multi-mask post-processing
            best_mask_idx = out_logits.argmax(1).squeeze(1)
            batch_idx = torch.arange(len(best_mask_idx), device=best_mask_idx.device)
            
            multi_out = {
                "multi_pred_logits": out_logits.cpu().numpy(),
                "multi_pred_masks": out_masks.cpu().numpy(),
                "multi_pred_boxes_xyxy": out_boxes_xyxy.cpu().numpy(),
            }
            # Select best for standard keys
            out_logits = out_logits[batch_idx, best_mask_idx].unsqueeze(1)
            out_masks = out_masks[batch_idx, best_mask_idx].unsqueeze(1)
            out_boxes_xyxy = out_boxes_xyxy[batch_idx, best_mask_idx].unsqueeze(1)
        else:
             multi_out = {}
        
        # Standard post-processing
        out_scores = out_logits.squeeze(-1).sigmoid()
        keep = out_scores > output_prob_thresh
        
        out_probs = out_scores[keep]
        out_boxes_xyxy = out_boxes_xyxy[keep]
        out_masks = out_masks[keep]

        if out_masks.size(0) > 0:
            out_masks_orig_size = torch.nn.functional.interpolate(
                out_masks.unsqueeze(1), # Add channel dim
                size=original_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1) # Remove channel dim
            out_binary_masks = out_masks_orig_size > 0.0
        else:
            out_binary_masks = torch.zeros(0, *original_size, dtype=torch.bool, device=self.device)

        frame_outputs = {
            "out_probs": out_probs.float().cpu().numpy(),
            "out_boxes_xywh": box_xyxy_to_xywh(out_boxes_xyxy).float().cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
        }
        frame_outputs.update(multi_out)

        return frame_outputs

