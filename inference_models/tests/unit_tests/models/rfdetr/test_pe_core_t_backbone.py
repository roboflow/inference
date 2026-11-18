import pytest
import timm
import torch

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
    reference.load_state_dict(encoder.model.state_dict())
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


@pytest.mark.parametrize(
    ("config_class", "resolution"),
    [(RFDETRAttoConfig, 440), (RFDETRFemtoConfig, 448), (RFDETRPicoConfig, 640)],
    ids=["atto", "femto", "pico"],
)
def test_model_trained_at_non_default_resolution_loads_default_sized_position_embedding(
    config_class, resolution: int
) -> None:
    # Training stores the position embedding at positional_encoding_size patches per side
    # whatever the training resolution, and timm resamples it to the input grid.
    torch.manual_seed(0)
    default_model = build_model(config=config_class(device=torch.device("cpu")))
    config = config_class(device=torch.device("cpu"))
    config.resolution = resolution
    model = build_model(config=config).eval()

    model.load_state_dict(default_model.state_dict())
    with torch.no_grad():
        outputs = model(torch.randn(1, 3, resolution, resolution))

    pos_embed = model.state_dict()["backbone.0.encoder.model.pos_embed"]
    assert pos_embed.shape[1] == config.positional_encoding_size**2 + 1
    assert outputs["pred_boxes"].shape == (1, config.num_queries, 4)
