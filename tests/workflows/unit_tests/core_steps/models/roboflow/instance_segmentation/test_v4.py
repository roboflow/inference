import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.roboflow.instance_segmentation.v4 import (
    BlockManifest,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_instance_segmentation_model_v4_validation_when_minimalistic_config_is_provided(
    images_field_alias: str,
) -> None:
    data = {
        "type": "roboflow_core/roboflow_instance_segmentation_model@v4",
        "name": "some",
        images_field_alias: "$inputs.image",
        "model_id": "some/1",
    }

    result = BlockManifest.model_validate(data)

    assert result == BlockManifest(
        type="roboflow_core/roboflow_instance_segmentation_model@v4",
        name="some",
        images="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "images", "model_id"])
def test_instance_segmentation_model_v4_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    data = {
        "type": "roboflow_core/roboflow_instance_segmentation_model@v4",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_instance_segmentation_model_v4_predictions_output_advertises_rle_kind_first() -> None:
    outputs = {o.name: o for o in BlockManifest.describe_outputs()}

    predictions_kinds = outputs["predictions"].kind

    assert predictions_kinds[0] is RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND, (
        "RLE kind must be listed first so the RLE serializer is attempted before "
        "the polygon fallback — otherwise the block silently emits polygon points "
        "even though it requested RLE from the model."
    )
    assert INSTANCE_SEGMENTATION_PREDICTION_KIND in predictions_kinds, (
        "Polygon kind must remain as a fallback for code paths where the model "
        "did not produce RLE (e.g. USE_INFERENCE_MODELS=False)."
    )


def test_instance_segmentation_model_v4_ui_version_label_matches_type() -> None:
    schema_extra = BlockManifest.model_config["json_schema_extra"]

    assert schema_extra["version"] == "v4"
