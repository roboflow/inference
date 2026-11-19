import json
from pathlib import Path
from typing import Dict

import numpy as np
import pytest
import timm
import torch

from inference_models import AutoModel
from inference_models.errors import MissingDependencyError
from inference_models.models.auto_loaders.core import (
    MODEL_TYPES_TO_LOAD_FROM_CHECKPOINT,
)
from inference_models.models.rfdetr import pe_core_t_backbone
from inference_models.models.rfdetr.pe_core_t_backbone import (
    PE_CORE_T_ARCHITECTURE,
    PECoreTEncoder,
    _apply_windowing_with_rope,
    _undo_windowing,
)
from inference_models.models.rfdetr.rfdetr_base_pytorch import (
    RFDETRAttoConfig,
    RFDETRFemtoConfig,
    RFDETRPicoConfig,
    build_model,
)
from inference_models.models.rfdetr.rfdetr_object_detection_pytorch import (
    CONFIG_FOR_MODEL_TYPE,
)

OUT_FEATURE_INDEXES = [2, 5, 8, 11]


def _timm_reference(
    encoder: PECoreTEncoder, *, patch_size: int, shape
) -> torch.nn.Module:
    # An unmodified timm trunk with the encoder's weights: timm itself skips RoPE for the class
    # token, which the encoder reproduces with an identity rotary embedding.
    reference = timm.create_model(
        PE_CORE_T_ARCHITECTURE,
        pretrained=False,
        patch_size=patch_size,
        img_size=encoder.model.patch_embed.img_size[0],
    )
    # The encoder drops the CLIP attention pool and classifier, which the trunk never runs.
    incompatible = reference.load_state_dict(encoder.model.state_dict(), strict=False)
    assert incompatible.missing_keys
    assert all(
        key.startswith(("attn_pool.", "head.")) for key in incompatible.missing_keys
    )
    assert not incompatible.unexpected_keys
    reference.set_input_size(img_size=shape)
    return reference.eval()


@pytest.mark.parametrize(
    ("trained_size", "input_size"),
    [((380, 380), (380, 380)), ((380, 380), (400, 360)), ((440, 440), (440, 440))],
    ids=["trained-size", "other-size", "trained-at-non-default-resolution"],
)
def test_unwindowed_encoder_matches_timm_forward_intermediates(
    trained_size, input_size
) -> None:
    torch.manual_seed(0)
    encoder = PECoreTEncoder(
        shape=trained_size,
        out_feature_indexes=OUT_FEATURE_INDEXES,
        patch_size=20,
        num_windows=1,
        positional_encoding_size=380 // 20,
    ).eval()
    reference = _timm_reference(encoder, patch_size=20, shape=input_size)
    x = torch.randn(2, 3, *input_size)

    with torch.no_grad():
        features = encoder(x)
        expected = reference.forward_intermediates(
            x,
            indices=OUT_FEATURE_INDEXES,
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )

    assert len(features) == len(expected)
    for actual, wanted in zip(features, expected):
        torch.testing.assert_close(actual, wanted, rtol=1e-4, atol=1e-4)


def test_other_input_size_leaves_trained_size_output_unchanged() -> None:
    torch.manual_seed(0)
    encoder = PECoreTEncoder(
        shape=(384, 384),
        out_feature_indexes=OUT_FEATURE_INDEXES,
        patch_size=16,
        num_windows=2,
        positional_encoding_size=384 // 16,
    ).eval()
    x = torch.randn(1, 3, 384, 384)

    with torch.no_grad():
        before = encoder(x)
        other = encoder(torch.randn(1, 3, 448, 320))
        after = encoder(x)

    assert other[0].shape == (1, 192, 28, 20)
    for first, second in zip(before, after):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("num_windows", [1, 2, 4])
def test_windowing_round_trips_with_rope_in_window_order(num_windows: int) -> None:
    side, channels, rope_dim = 8, 6, 4
    x = torch.randn(2, side * side, channels)
    ropes = torch.arange(side * side * rope_dim, dtype=torch.float32).view(-1, rope_dim)
    per_window = side // num_windows

    windowed, windowed_ropes = _apply_windowing_with_rope(
        x,
        ropes=ropes,
        num_windows=num_windows,
        patch_size=16,
        height=side * 16,
        width=side * 16,
    )
    restored = _undo_windowing(
        windowed,
        num_windows=num_windows,
        patch_size=16,
        height=side * 16,
        width=side * 16,
    )

    grid = x.view(2, side, side, channels)
    rope_grid = ropes.view(side, side, rope_dim)
    for window_row in range(num_windows):
        for window_col in range(num_windows):
            window = window_row * num_windows + window_col
            rows = slice(window_row * per_window, (window_row + 1) * per_window)
            cols = slice(window_col * per_window, (window_col + 1) * per_window)
            torch.testing.assert_close(
                windowed[window], grid[0, rows, cols].reshape(-1, channels)
            )
            torch.testing.assert_close(
                windowed[num_windows**2 + window],
                grid[1, rows, cols].reshape(-1, channels),
            )
            torch.testing.assert_close(
                windowed_ropes[window], rope_grid[rows, cols].reshape(-1, rope_dim)
            )
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("model_type", "config_class", "pos_embed_tokens", "decoder_layers", "num_queries"),
    [
        ("rfdetr-atto", RFDETRAttoConfig, 19 * 19 + 1, 0, 300),
        ("rfdetr-femto", RFDETRFemtoConfig, 24 * 24 + 1, 2, 200),
        ("rfdetr-pico", RFDETRPicoConfig, 28 * 28 + 1, 3, 200),
    ],
)
def test_pe_core_t_sizes_build_the_released_architecture(
    model_type, config_class, pos_embed_tokens, decoder_layers, num_queries
) -> None:
    config = config_class(device=torch.device("cpu"), num_classes=4)

    model = build_model(config=config).eval()
    state = model.state_dict()
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, config.resolution, config.resolution))

    assert CONFIG_FOR_MODEL_TYPE[model_type] is config_class
    assert model_type in MODEL_TYPES_TO_LOAD_FROM_CHECKPOINT
    assert state["backbone.0.encoder.model.pos_embed"].shape == (
        1,
        pos_embed_tokens,
        192,
    )
    assert len(model.transformer.decoder.layers) == decoder_layers
    assert outputs["pred_boxes"].shape == (1, num_queries, 4)
    assert outputs["pred_logits"].shape == (1, num_queries, 5)


def test_old_timm_without_pe_core_t_raises_missing_dependency(monkeypatch) -> None:
    monkeypatch.setattr(pe_core_t_backbone.timm, "list_models", lambda pattern: [])

    with pytest.raises(MissingDependencyError, match="timm>=1.0.18"):
        PECoreTEncoder(
            shape=(384, 384),
            out_feature_indexes=OUT_FEATURE_INDEXES,
            patch_size=16,
            num_windows=2,
            positional_encoding_size=384 // 16,
        )


NON_DEFAULT_RESOLUTIONS = pytest.mark.parametrize(
    ("config_class", "model_type", "resolution"),
    [
        (RFDETRAttoConfig, "rfdetr-atto", 440),
        (RFDETRFemtoConfig, "rfdetr-femto", 448),
        (RFDETRPicoConfig, "rfdetr-pico", 640),
    ],
    ids=["atto", "femto", "pico"],
)
CLASS_NAMES = ["background", "a", "b", "c"]


def _trained_state_dict(config_class, *, resolution: int) -> Dict[str, torch.Tensor]:
    torch.manual_seed(0)
    config = config_class(
        device=torch.device("cpu"),
        num_classes=len(CLASS_NAMES) - 1,
        resolution=resolution,
    )
    model = build_model(config=config)

    return model.state_dict()


def _write_torch_package(
    package_dir: Path,
    *,
    state_dict: Dict[str, torch.Tensor],
    model_type: str,
    resolution: int,
    divisibility: int,
) -> Path:
    package_dir.mkdir()
    torch.save({"model": state_dict}, package_dir / "weights.pth")
    (package_dir / "class_names.txt").write_text("\n".join(CLASS_NAMES))
    (package_dir / "model_type.json").write_text(json.dumps({"model_type": model_type}))
    (package_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_architecture": "rfdetr",
                "task_type": "object-detection",
                "backend_type": "torch",
            }
        )
    )
    network_input = {
        "training_input_size": {"height": resolution, "width": resolution},
        "dynamic_spatial_size_supported": True,
        "dynamic_spatial_size_mode": {
            "type": "pad-to-be-divisible",
            "value": divisibility,
        },
        "color_mode": "rgb",
        "resize_mode": "stretch",
        "input_channels": 3,
        "scaling_factor": 255,
        "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    }
    (package_dir / "inference_config.json").write_text(
        json.dumps({"network_input": network_input})
    )

    return package_dir


def _loaded_state_dict(model) -> Dict[str, torch.Tensor]:
    return model._model.state_dict()


@NON_DEFAULT_RESOLUTIONS
def test_resolution_sizes_the_position_embedding_grid(
    config_class, model_type, resolution: int
) -> None:
    # Training does the same: the grid follows an explicit resolution, an explicit grid is kept.
    default_grid = config_class.model_fields["positional_encoding_size"].default
    device = torch.device("cpu")

    assert (
        config_class(device=device, resolution=resolution).positional_encoding_size
        == resolution // config_class(device=device).patch_size
    )
    assert (
        config_class(
            device=device,
            resolution=resolution,
            positional_encoding_size=default_grid,
        ).positional_encoding_size
        == default_grid
    )
    assert config_class(device=device).positional_encoding_size == default_grid


@NON_DEFAULT_RESOLUTIONS
def test_package_fine_tuned_at_non_default_resolution_loads(
    config_class, model_type, resolution: int, tmp_path: Path
) -> None:
    # A model fine-tuned at another resolution stores resolution // patch_size patches per side.
    state_dict = _trained_state_dict(config_class, resolution=resolution)
    config = config_class(device=torch.device("cpu"))
    package = _write_torch_package(
        tmp_path / "package",
        state_dict=state_dict,
        model_type=model_type,
        resolution=resolution,
        divisibility=config.patch_size * config.num_windows,
    )

    model = AutoModel.from_pretrained(str(package), device=torch.device("cpu"))
    predictions = model(np.zeros((resolution, resolution, 3), dtype=np.uint8))

    grid = resolution // config.patch_size
    loaded_pos_embed = _loaded_state_dict(model)["backbone.0.encoder.model.pos_embed"]
    assert loaded_pos_embed.shape == (1, grid * grid + 1, 192)
    torch.testing.assert_close(
        loaded_pos_embed, state_dict["backbone.0.encoder.model.pos_embed"]
    )
    assert len(predictions) == 1


def _platform_layout(
    state_dict: Dict[str, torch.Tensor], *, patch_size: int
) -> Dict[str, torch.Tensor]:
    # Platform checkpoints also keep the PE-CLIP attention pool and classifier weights.
    clip = timm.create_model(
        PE_CORE_T_ARCHITECTURE, pretrained=False, patch_size=patch_size
    )
    unused = {
        f"backbone.0.encoder.model.{key}": value
        for key, value in clip.state_dict().items()
        if key.startswith(("attn_pool.", "head."))
    }
    assert unused

    return {**state_dict, **unused}


@pytest.mark.parametrize("resolution", [None, 440], ids=["default", "440"])
def test_platform_and_rfdetr_plus_checkpoint_layouts_load_to_the_same_model(
    resolution, tmp_path: Path
) -> None:
    trained_resolution = (
        resolution or RFDETRAttoConfig.model_fields["resolution"].default
    )
    rfdetr_plus_layout = _trained_state_dict(
        RFDETRAttoConfig, resolution=trained_resolution
    )
    platform_layout = _platform_layout(rfdetr_plus_layout, patch_size=20)
    image = np.random.default_rng(0).integers(
        0, 255, (trained_resolution, trained_resolution, 3), dtype=np.uint8
    )

    predictions = []
    for name, state_dict in [
        ("platform", platform_layout),
        ("rfdetr_plus", rfdetr_plus_layout),
    ]:
        checkpoint = tmp_path / f"{name}.pth"
        torch.save({"model": state_dict}, checkpoint)
        model = AutoModel.from_pretrained(
            str(checkpoint),
            model_type="rfdetr-atto",
            labels=CLASS_NAMES,
            resolution=resolution,
            device=torch.device("cpu"),
        )
        assert not any(
            key.startswith("backbone.0.encoder.model.attn_pool.")
            or key.startswith("backbone.0.encoder.model.head.")
            for key in _loaded_state_dict(model)
        )
        predictions.append(model(image, confidence=0.0)[0])

    torch.testing.assert_close(predictions[0].xyxy, predictions[1].xyxy)
    torch.testing.assert_close(predictions[0].confidence, predictions[1].confidence)
    torch.testing.assert_close(predictions[0].class_id, predictions[1].class_id)


def test_checkpoint_missing_a_used_encoder_weight_still_fails_to_load(
    tmp_path: Path,
) -> None:
    state_dict = _trained_state_dict(RFDETRAttoConfig, resolution=380)
    del state_dict["backbone.0.encoder.model.blocks.0.attn.qkv.weight"]
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": state_dict}, checkpoint)

    with pytest.raises(RuntimeError, match="blocks.0.attn.qkv.weight"):
        AutoModel.from_pretrained(
            str(checkpoint),
            model_type="rfdetr-atto",
            labels=CLASS_NAMES,
            device=torch.device("cpu"),
        )
