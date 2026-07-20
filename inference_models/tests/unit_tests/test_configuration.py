import pytest

from inference_models.configuration import (
    DEFAULT_QWEN_IMAGE_EDIT_CPU_OFFLOAD,
    DEFAULT_RFDETR_PIPELINE_DEPTH,
    MAX_RFDETR_PIPELINE_DEPTH,
    get_rfdetr_pipeline_depth,
    parse_qwen_image_edit_cpu_offload,
    parse_rfdetr_pipeline_depth,
)
from inference_models.errors import InvalidEnvVariable


def test_parse_rfdetr_pipeline_depth_uses_default_when_env_missing() -> None:
    assert parse_rfdetr_pipeline_depth(None) == DEFAULT_RFDETR_PIPELINE_DEPTH


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", 1),
        ("2", 2),
        (" 3 ", MAX_RFDETR_PIPELINE_DEPTH),
        ("99", MAX_RFDETR_PIPELINE_DEPTH),
    ],
)
def test_parse_rfdetr_pipeline_depth_accepts_positive_integers(
    value: str,
    expected: int,
) -> None:
    assert parse_rfdetr_pipeline_depth(value) == expected


@pytest.mark.parametrize("value", ["invalid", "1.5", "", "0", "-1"])
def test_parse_rfdetr_pipeline_depth_rejects_invalid_values(value: str) -> None:
    with pytest.raises(InvalidEnvVariable):
        parse_rfdetr_pipeline_depth(value)


def test_get_rfdetr_pipeline_depth_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "3")
    assert get_rfdetr_pipeline_depth() == MAX_RFDETR_PIPELINE_DEPTH


@pytest.mark.parametrize("value", ["0", "-4", "invalid"])
def test_get_rfdetr_pipeline_depth_rejects_invalid_environment(
    monkeypatch,
    value: str,
) -> None:
    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", value)
    with pytest.raises(InvalidEnvVariable):
        get_rfdetr_pipeline_depth()


def test_parse_qwen_image_edit_cpu_offload_uses_default_when_env_missing() -> None:
    assert (
        parse_qwen_image_edit_cpu_offload(None) == DEFAULT_QWEN_IMAGE_EDIT_CPU_OFFLOAD
    )


@pytest.mark.parametrize(
    "value, expected",
    [
        ("model", "model"),
        ("sequential", "sequential"),
        ("none", "none"),
        (" Sequential ", "sequential"),
        ("MODEL", "model"),
    ],
)
def test_parse_qwen_image_edit_cpu_offload_accepts_valid_modes(
    value: str,
    expected: str,
) -> None:
    assert parse_qwen_image_edit_cpu_offload(value) == expected


@pytest.mark.parametrize("value", ["", "invalid", "true", "sequentiall", "0"])
def test_parse_qwen_image_edit_cpu_offload_rejects_invalid_values(value: str) -> None:
    with pytest.raises(InvalidEnvVariable):
        parse_qwen_image_edit_cpu_offload(value)
