"""Experimental, fixed-shape ONNX export of the native sam3==0.1.4 model."""

import hashlib
import json
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

IMAGE_SIZE = 1008
TEXT_LENGTH = 32
VISION_OUTPUTS = ("fpn_0", "fpn_1", "fpn_2", "position_2")
TEXT_OUTPUTS = ("language_features", "language_mask")
DECODER_OUTPUTS = ("pred_masks", "pred_boxes", "pred_logits", "presence_logit_dec")
PARTS = ("vision", "text", "decoder")


def apply_rotary_real(query, key, frequencies):
    def rotate(tensor):
        pairs = tensor.float().reshape(*tensor.shape[:-1], -1, 2)
        real, imaginary = pairs[..., 0], pairs[..., 1]
        cosine, sine = frequencies[..., 0], frequencies[..., 1]
        return (
            torch.stack(
                (real * cosine - imaginary * sine, real * sine + imaginary * cosine),
                dim=-1,
            )
            .flatten(-2)
            .type_as(tensor)
        )

    return rotate(query), rotate(key)


def _apply_real_rope(attention, query, key):
    return apply_rotary_real(query, key, attention.freqs_cis)


@contextmanager
def sam3_onnx_rope(model):
    from sam3.model.vitdet import Attention

    original = []
    try:
        for module in model.modules():
            if isinstance(module, Attention) and module.use_rope:
                original.append(
                    (module, module.freqs_cis, module.__dict__.get("_apply_rope"))
                )
                # ONNX cannot represent native complex tensors or complex multiplication.
                module.freqs_cis = torch.view_as_real(module.freqs_cis).contiguous()
                module._apply_rope = MethodType(_apply_real_rope, module)
        yield
    finally:
        for module, frequencies, instance_method in original:
            module.freqs_cis = frequencies
            if instance_method is None:
                del module._apply_rope
            else:
                module._apply_rope = instance_method


class SAM3VisionExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = self.backbone.forward_image(images)
        return (*output["backbone_fpn"], output["vision_pos_enc"][-1])


class SAM3TextExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.language = model.backbone.language_backbone

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, tokens = self.language.encoder(input_ids)
        features = self.language.resizer(tokens.transpose(0, 1))
        return features, input_ids.eq(0)


class SAM3DecoderExport(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        if model.num_feature_levels != 1 or model.num_interactive_steps_val != 0:
            raise ValueError(
                "Split export requires one feature level and no refinement steps"
            )
        self.register_buffer("query_ids", torch.zeros(1, dtype=torch.long))

    def forward(
        self,
        fpn_0: torch.Tensor,
        fpn_1: torch.Tensor,
        fpn_2: torch.Tensor,
        position_2: torch.Tensor,
        language_features: torch.Tensor,
        language_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        backbone = {
            "backbone_fpn": [fpn_0, fpn_1, fpn_2],
            "vision_pos_enc": [position_2],
            "language_features": language_features,
            "language_mask": language_mask,
        }
        query = SimpleNamespace(img_ids=self.query_ids, text_ids=self.query_ids)
        # Native grounding retains the empty geometry token and joint presence score.
        output = self.model.forward_grounding(
            backbone_out=backbone,
            find_input=query,
            find_target=None,
            geometric_prompt=self.model._get_dummy_prompt(num_prompts=1),
        )
        return tuple(output[name] for name in DECODER_OUTPUTS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_sam3_onnx(model_package: Path, output_dir: Path) -> Dict:
    from inference_models.models.sam3.sam3_torch import SAM3Torch

    if version("sam3") != "0.1.4":
        raise ValueError("Native split export currently supports sam3==0.1.4 only")
    elif not torch.cuda.is_available():
        raise RuntimeError("Export SAM3 on a CUDA device with sufficient memory")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(
            "Export into an empty directory to avoid stale engine artifacts"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", torch.cuda.current_device())
    loaded = SAM3Torch.from_pretrained(
        str(model_package),
        device=device,
        image_size=IMAGE_SIZE,
        enable_inst_interactivity=False,
    )
    model = loaded._model.float().eval()
    tokenizer = model.backbone.language_backbone.tokenizer
    input_ids = tokenizer(["person"], context_length=TEXT_LENGTH).to(device)
    images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    vision = SAM3VisionExport(model).eval()
    text = SAM3TextExport(model).eval()
    decoder = SAM3DecoderExport(model).to(device).eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=False), sdpa_kernel(
        SDPBackend.MATH
    ), sam3_onnx_rope(model):
        image_features = vision(images)
        text_features = text(input_ids)
        if len(image_features) != len(VISION_OUTPUTS):
            raise ValueError("Expected three native SAM3 feature maps")
        exports = (
            ("vision", vision, (images,), ("images",), VISION_OUTPUTS),
            ("text", text, (input_ids,), ("input_ids",), TEXT_OUTPUTS),
            (
                "decoder",
                decoder,
                (*image_features, *text_features),
                (*VISION_OUTPUTS, *TEXT_OUTPUTS),
                DECODER_OUTPUTS,
            ),
        )
        for name, wrapper, inputs, input_names, output_names in exports:
            torch.onnx.export(
                wrapper,
                inputs,
                str(output_dir / f"{name}.onnx"),
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=17,
                dynamo=False,
                external_data=True,
            )

    manifest = {
        "schema_version": 1,
        "sam3_version": version("sam3"),
        "torch_version": torch.__version__,
        "image_size": IMAGE_SIZE,
        "text_length": TEXT_LENGTH,
        "batch_size": 1,
        "prompt_mode": "text_only",
        "weights_sha256": file_sha256(model_package / "weights.pt"),
        "tokenizer_sha256": file_sha256(model_package / "bpe_simple_vocab_16e6.txt.gz"),
    }
    (output_dir / "export.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
