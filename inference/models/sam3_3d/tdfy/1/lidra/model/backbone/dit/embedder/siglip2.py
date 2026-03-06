import torch
from typing import Optional, Dict, Any
import warnings
from torchvision.transforms import Normalize
import torch.nn.functional as F
from transformers import AutoModel


class SigLIP2(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 384,
        siglip2_model: str = "google/siglip2-large-patch16-384",
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        prenorm_features: bool = True,
        patch_token: bool = False,
    ):
        super().__init__()
        if backbone_kwargs is None:
            backbone_kwargs = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone = AutoModel.from_pretrained(
                siglip2_model,
                torch_dtype=torch.bfloat16,
                **backbone_kwargs,
            )

        self.resize_input_size = (input_size, input_size)
        self.embed_dim = self.backbone.config.vision_config.hidden_size
        self.input_size = input_size
        self.input_channels = 3
        self.normalize_images = normalize_images
        self.prenorm_features = prenorm_features
        self.patch_token = patch_token

        # freeze
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def extract_patch_token(self, input_img):
        feats = self.backbone.vision_model(pixel_values=input_img).last_hidden_state
        return feats

    @torch.no_grad()
    def extract_cls_token(self, input_img):
        feats = self.backbone.get_image_features(pixel_values=input_img)
        # unsqueeze to get a token dimension
        return feats[:, None]

    def forward(self, x, **kwargs):
        _resized_images = torch.nn.functional.interpolate(
            x,
            size=self.resize_input_size,
            mode="bilinear",
            align_corners=False,
        )

        if x.shape[1] == 1:
            _resized_images = _resized_images.repeat(1, 3, 1, 1)

        if self.normalize_images:
            _resized_images = Normalize(
                [0.5] * 3,
                [0.5] * 3,
            )(_resized_images)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.patch_token:
                features = self.extract_patch_token(_resized_images)
            else:
                features = self.extract_cls_token(_resized_images)
        if self.prenorm_features:
            tokens = torch.nn.functional.normalize(features, dim=-1)
        else:
            tokens = features
        return tokens.to(x.dtype)
