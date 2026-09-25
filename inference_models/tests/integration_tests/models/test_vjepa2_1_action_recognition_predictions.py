"""Load and predict with a real fine-tuned V-JEPA package.

From inference_models/, with CUDA available for the prediction cases:

    VJEPA_ACTION_RECOGNITION_PACKAGE_DIR=/path/to/flat/package \
    python -m pytest tests/integration_tests/models/test_vjepa2_1_action_recognition_predictions.py -m slow

Without the override, the fixture downloads the synthetic-dataset t7 package.
"""

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open

from inference_models import AutoModel
from inference_models.models.vjepa2_1.model import VJepaActionRecognition

pytestmark = [pytest.mark.slow, pytest.mark.torch_models]


@pytest.fixture(scope="module")
def loaded_model(vjepa_action_recognition_package, tmp_path_factory):
    """Load once through the registry without modifying the exported package.

    Args:
        vjepa_action_recognition_package: Directory containing the trained export.
        tmp_path_factory: Pytest factory for temporary directories.

    Returns:
        Model loaded with the package's real encoder and head weights.
    """
    package_dir = tmp_path_factory.mktemp("vjepa-automodel")
    for name in ("model.safetensors", "inference_config.json", "class_names.txt"):
        source = vjepa_action_recognition_package / name
        assert source.is_file(), f"Missing V-JEPA package artifact: {source}"
        (package_dir / name).symlink_to(source)

    # The weights provider normally supplies this manifest, not the trainer export.
    (package_dir / "model_config.json").write_text(
        json.dumps(
            {
                "model_architecture": "vjepa2_1",
                "task_type": "action-recognition",
                "backend_type": "torch",
            }
        )
    )
    model = AutoModel.from_pretrained(
        str(package_dir),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    return model


def test_real_package_loads_through_automodel(
    loaded_model, vjepa_action_recognition_package: Path
) -> None:
    config = json.loads(
        (vjepa_action_recognition_package / "inference_config.json").read_text()
    )
    classes = (
        (vjepa_action_recognition_package / "class_names.txt").read_text().splitlines()
    )

    assert isinstance(loaded_model, VJepaActionRecognition)
    assert loaded_model.class_names == classes == config["class_names"]
    assert loaded_model.video_sampling.max_frames == config["network_input"]["frames"]
    assert loaded_model.video_sampling.sample_fps == config["network_input"]["fps"]
    assert (
        loaded_model.confidence_threshold
        == config["post_processing"]["confidence_threshold"]
    )
    with safe_open(
        str(vjepa_action_recognition_package / "model.safetensors"),
        framework="pt",
        device="cpu",
    ) as checkpoint:
        for key in ("encoder.patch_embed.proj.weight", "head.classifier.weight"):
            torch.testing.assert_close(
                loaded_model._model.state_dict()[key].cpu(),
                checkpoint.get_tensor(key).float(),
                rtol=0,
                atol=0,
            )


@pytest.mark.gpu_only
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for predictions"
)
@pytest.mark.parametrize(
    "input_kind", ["numpy_full_window", "cuda_tensor_padded_window"]
)
def test_real_weights_predict_scored_spans_and_filter_them(
    loaded_model, input_kind
) -> None:
    model = loaded_model
    count = model.video_sampling.max_frames
    if input_kind == "cuda_tensor_padded_window":
        count = max(1, count // 2)
    frames = []
    for x in np.linspace(0, 96, num=count, dtype=int):
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        frame[32:64, x : x + 32] = 255
        if input_kind == "cuda_tensor_padded_window":
            frame = torch.from_numpy(frame).permute(2, 0, 1).to("cuda")
        frames.append(frame)

    predictions = model.infer(
        frames=frames, fps=model.video_sampling.sample_fps, confidence=0
    )

    assert len(predictions) == count * len(model.class_names)
    for prediction in predictions:
        assert prediction.class_name in model.class_names
        assert prediction.end_exclusive is True
        assert 0 <= prediction.start_frame_idx < prediction.end_frame_idx <= count
        assert math.isfinite(prediction.confidence)
        assert 0 <= prediction.confidence <= 1

    selected_class = model.class_names[0]
    class_predictions = [p for p in predictions if p.class_name == selected_class]
    threshold = sorted(p.confidence for p in class_predictions)[
        len(class_predictions) // 2
    ]
    filtered = model.infer(
        frames=frames,
        fps=model.video_sampling.sample_fps,
        confidence=threshold,
        class_names=[selected_class],
    )

    assert filtered == [p for p in class_predictions if p.confidence >= threshold]
    assert filtered


@pytest.mark.gpu_only
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for predictions"
)
def test_t7_predictions_match_pinned_reference(
    loaded_model, vjepa_action_recognition_package, vjepa_prediction_frames
) -> None:
    expected = json.loads(
        (
            Path(__file__).parent / "fixtures" / "vjepa2_1_t7_predictions.json"
        ).read_text()
    )
    frames = vjepa_prediction_frames
    assert len(frames) == expected["frame_count"]
    assert sha256(frames.tobytes()).hexdigest() == expected["frames_sha256"]
    weights_hash = sha256()
    with (vjepa_action_recognition_package / "model.safetensors").open("rb") as weights:
        for block in iter(lambda: weights.read(1024 * 1024), b""):
            weights_hash.update(block)
    assert weights_hash.hexdigest() == expected["weights_sha256"]

    predictions = loaded_model.infer(
        frames=list(frames), fps=expected["sample_fps"], confidence=0
    )

    assert len(predictions) == expected["prediction_count"]
    # Serving uses BF16; the independent reference was captured in FP32.
    for reference in expected["predictions"]:
        prediction = predictions[reference["index"]]
        assert prediction.class_name == reference["class_name"]
        assert prediction.end_exclusive is True
        np.testing.assert_allclose(
            [prediction.start_frame_idx, prediction.end_frame_idx],
            [reference["start_frame"], reference["end_frame"]],
            rtol=0,
            atol=0.5,
        )
        assert prediction.confidence == pytest.approx(
            reference["confidence"], rel=0, abs=0.01
        )
