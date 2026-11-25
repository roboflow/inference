import json
import os

import pytest

pytest.importorskip(
    "inference_models",
    reason="inference_models is required to exercise the compiler's package verification",
)
pytest.importorskip(
    "tensorrt",
    reason="the compilation handlers import the TensorRT engine builder",
)

from inference_cli.lib.enterprise.inference_compiler.core.compilation_handlers.default import (
    verify_model_package,
)
from inference_cli.lib.enterprise.inference_compiler.core.entities import TRTConfig
from inference_models.models.common.roboflow.oriented_box_metadata import (
    OrientedBoxMetadata,
)
from inference_models.weights_providers.entities import ModelMetadata


def _verified_model_config(tmp_path, oriented_box_metadata):
    """The model_config.json the compiler's verification loads a compiled package with."""
    (tmp_path / "inference_config.json").write_text(json.dumps({"network_input": {}}))
    for name in ("class_names.txt", "engine.plan"):
        (tmp_path / name).write_text("x")
    seen = {}

    def verify_model(package_dir):
        with open(os.path.join(package_dir, "model_config.json")) as file:
            seen.update(json.load(file))

    verify_model_package(
        model_metadata=ModelMetadata(
            model_id="workspace/obb",
            model_architecture="rfdetr",
            model_packages=[],
            task_type="obb-detection",
            oriented_box_metadata=oriented_box_metadata,
        ),
        model_package_id="package",
        trt_config=TRTConfig(static_batch_size=1),
        inference_config_path=str(tmp_path / "inference_config.json"),
        class_names_path=str(tmp_path / "class_names.txt"),
        engine_path=str(tmp_path / "engine.plan"),
        verify_model=verify_model,
        keypoints_metadata_path=None,
    )
    return seen


def test_verification_loads_oriented_models_with_their_class_heading_policy(tmp_path):
    policy = OrientedBoxMetadata.model_validate(
        {"classHeading": {"ship": True, "tank": False}, "classHeadingRevision": 1}
    )

    config = _verified_model_config(tmp_path, policy)

    assert config["oriented_box_metadata"] == policy.model_dump(mode="json")


def test_verification_config_of_other_models_is_unchanged(tmp_path):
    config = _verified_model_config(tmp_path, None)

    assert "oriented_box_metadata" not in config
