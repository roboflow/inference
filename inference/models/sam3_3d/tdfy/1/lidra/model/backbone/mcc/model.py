import torch
from functools import partial

from lidra.model.backbone.mcc.encoder import ImageEncoder
from lidra.model.backbone.mcc.decoder import DensityDecoder, DecoderBlock
from lidra.model.backbone.mcc.common import Transformer, Block


def square_image(image):
    _, _, h, w = image.shape  # assuming image is in (B, C, H, W) format
    if h == w:
        return image  # image is already square

    # calculate padding
    diff = abs(h - w)
    pad2 = diff

    # pad image to make it square
    if h > w:
        padding = (0, pad2, 0, 0)  # Pad width (left, right, top, bottom)
    else:
        padding = (0, 0, 0, pad2)  # Pad height

    # Apply padding
    padded_image = torch.nn.functional.pad(image, padding, mode="constant", value=0)
    return padded_image


def reshape_image(x, size, mode="bilinear"):
    x = torch.nn.functional.interpolate(
        x,
        size=(size, size),
        mode=mode,
    )
    return x


def normalize_image(
    x,
    mean=(0.485, 0.456, 0.406),  # default weights are resnet weights
    std=(0.229, 0.224, 0.225),
):
    mean = torch.tensor(mean, device=x.device).reshape((1, 3, 1, 1))
    std = torch.tensor(std, device=x.device).reshape((1, 3, 1, 1))
    mean = mean.reshape((1, 3, 1, 1))
    std = std.reshape((1, 3, 1, 1))
    x = (x - mean) / std
    return x


def preprocess_image(
    x,
    size=224,
    normalize=False,
    mode="bilinear",
):
    if x.shape[1] != x.shape[2]:
        x = square_image(x)

    x = reshape_image(x, size, mode=mode)

    if normalize:
        x = normalize_image(x)

    return x


class MCC(torch.nn.Module):
    DEFAULT_LAYER_NORM = partial(torch.nn.LayerNorm, eps=1e-6)
    IMAGE_SIZE = 224
    IMAGE_PATCH_SIZE = 16
    IMAGE_CHANNELS = 3

    # TODO(Pierre) : set encoder / decoder as arguments
    def __init__(
        self,
        encoder_embed_dim=768,
        encoder_n_blocks=12,
        encoder_n_heads=12,
        encoder_n_cls_tokens=1,
        decoder_embed_dim=512,
        decoder_n_blocks=8,
        decoder_n_heads=16,
        mlp_ratio=4,
        drop_path=0.1,
        norm_layer=DEFAULT_LAYER_NORM,
        max_n_queries=256000,
        color_prediction=False,
    ):
        super().__init__()

        # encode input image to token latents
        self.encoder = ImageEncoder(
            transformer=Transformer(
                embed_dim=encoder_embed_dim,
                n_blocks=encoder_n_blocks,
                n_heads=encoder_n_heads,
                n_cls_tokens=encoder_n_cls_tokens,
                norm_layer=norm_layer,
                drop_path=drop_path,
                mlp_ratio=mlp_ratio,
                block_fn=Block,
            ),
            image_size=MCC.IMAGE_SIZE,
            image_patch_size=MCC.IMAGE_PATCH_SIZE,
            image_channels=MCC.IMAGE_CHANNELS,
        )

        # map encoder tokens to decoder tokens
        self.embed_mapper = torch.nn.Linear(
            encoder_embed_dim,
            decoder_embed_dim,
            bias=True,
        )

        # decode latent tokens to output tokens
        self.decoder = DensityDecoder(
            transformer=Transformer(
                embed_dim=decoder_embed_dim,
                n_blocks=decoder_n_blocks,
                n_heads=decoder_n_heads,
                n_cls_tokens=0,
                norm_layer=norm_layer,
                drop_path=drop_path,
                mlp_ratio=mlp_ratio,
                block_fn=DecoderBlock,
            ),
            image_size=MCC.IMAGE_SIZE,
            image_patch_size=MCC.IMAGE_PATCH_SIZE,
            image_channels=MCC.IMAGE_CHANNELS,
            input_n_cls_tokens=encoder_n_cls_tokens,
            max_n_queries=max_n_queries,
        )

        # map tokens to occupancy logits + color (optional)
        self.color_prediction = color_prediction
        self.output_dim = 1
        if color_prediction:
            self.output_dim += 256 * MCC.IMAGE_CHANNELS
        self.logit_mapper = torch.nn.Linear(
            decoder_embed_dim,
            self.output_dim,
            bias=True,
        )

    def extract_color(self, logits):
        return logits[..., 1:]

    def extract_occupancy(self, logits):
        return logits[..., 0]

    def keep_xyz_tokens_only(self, x, size):
        x = x[:, -size:]
        return x

    def forward(self, image, xyz):
        x = preprocess_image(image)
        y = self.encoder(x)
        y = self.embed_mapper(y)
        z = self.decoder(y, xyz)
        z = self.keep_xyz_tokens_only(z, xyz.shape[1])
        logits = self.logit_mapper(z)
        if self.color_prediction:
            return self.extract_occupancy(logits), self.extract_color(logits)
        return self.extract_occupancy(logits)
