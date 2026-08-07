import sys
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.foundation.twelvelabs.v1 import (
    BlockManifest,
    TwelveLabsPegasusBlockV1,
    analyze_video_with_pegasus,
)


def test_twelvelabs_step_validation_when_input_is_valid() -> None:
    # given
    specification = {
        "type": "roboflow_core/twelvelabs_pegasus@v1",
        "name": "step_1",
        "video_url": "$inputs.video_url",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.twelvelabs_api_key",
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.type == "roboflow_core/twelvelabs_pegasus@v1"
    assert result.name == "step_1"
    assert result.video_url == "$inputs.video_url"
    assert result.prompt == "$inputs.prompt"
    assert result.api_key == "$inputs.twelvelabs_api_key"
    # defaults
    assert result.model_version == "pegasus1.5"
    assert result.max_tokens == 2048
    assert result.temperature is None


def test_twelvelabs_step_validation_when_inputs_given_directly() -> None:
    # given
    specification = {
        "type": "roboflow_core/twelvelabs_pegasus@v1",
        "name": "step_1",
        "video_url": "https://example.com/video.mp4",
        "prompt": "Summarize this video.",
        "api_key": "my-secret-key",
        "model_version": "pegasus1.2",
        "max_tokens": 1024,
        "temperature": 0.5,
    }

    # when
    result = BlockManifest.model_validate(specification)

    # then
    assert result.video_url == "https://example.com/video.mp4"
    assert result.prompt == "Summarize this video."
    assert result.model_version == "pegasus1.2"
    assert result.max_tokens == 1024
    assert result.temperature == 0.5


@pytest.mark.parametrize("value", [None, 1, True, [1, 2, 3]])
def test_twelvelabs_step_validation_when_prompt_is_invalid(value: Any) -> None:
    # given
    specification = {
        "type": "roboflow_core/twelvelabs_pegasus@v1",
        "name": "step_1",
        "video_url": "$inputs.video_url",
        "prompt": value,
        "api_key": "$inputs.twelvelabs_api_key",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", [256, 0, -1])
def test_twelvelabs_step_validation_when_max_tokens_below_minimum(value: int) -> None:
    # given - TwelveLabs requires max_tokens >= 512 for Pegasus
    specification = {
        "type": "roboflow_core/twelvelabs_pegasus@v1",
        "name": "step_1",
        "video_url": "$inputs.video_url",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.twelvelabs_api_key",
        "max_tokens": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


@pytest.mark.parametrize("value", [-0.1, 1.1, 2.0])
def test_twelvelabs_step_validation_when_temperature_out_of_range(value: float) -> None:
    # given
    specification = {
        "type": "roboflow_core/twelvelabs_pegasus@v1",
        "name": "step_1",
        "video_url": "$inputs.video_url",
        "prompt": "$inputs.prompt",
        "api_key": "$inputs.twelvelabs_api_key",
        "temperature": value,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(specification)


def test_twelvelabs_block_is_not_air_gapped() -> None:
    # when
    availability = BlockManifest.get_air_gapped_availability()

    # then
    assert availability.available is False


def _install_fake_twelvelabs_sdk(
    monkeypatch, captured: dict, response_data: str
) -> Mock:
    """Inject a fake `twelvelabs` SDK so the block can be exercised without the real package."""
    fake_client = MagicMock()

    def _analyze(**kwargs):
        captured.update(kwargs)
        result = Mock()
        result.data = response_data
        return result

    fake_client.analyze.side_effect = _analyze

    twelvelabs_module = MagicMock()
    twelvelabs_module.TwelveLabs.return_value = fake_client

    video_context_module = MagicMock()

    class _FakeVideoContextUrl:
        def __init__(self, url: str) -> None:
            self.url = url

    video_context_module.VideoContext_Url = _FakeVideoContextUrl

    monkeypatch.setitem(sys.modules, "twelvelabs", twelvelabs_module)
    monkeypatch.setitem(sys.modules, "twelvelabs.types", MagicMock())
    monkeypatch.setitem(
        sys.modules, "twelvelabs.types.video_context", video_context_module
    )
    return twelvelabs_module.TwelveLabs


def test_analyze_video_with_pegasus_wires_request_correctly(monkeypatch) -> None:
    # given
    captured: dict = {}
    twelvelabs_ctor = _install_fake_twelvelabs_sdk(
        monkeypatch, captured, response_data="A dog runs across a field."
    )

    # when
    output = analyze_video_with_pegasus(
        video_url="https://example.com/video.mp4",
        prompt="What happens in this video?",
        api_key="secret-key",
        model_version="pegasus1.5",
        max_tokens=1024,
        temperature=0.3,
    )

    # then
    assert output == "A dog runs across a field."
    twelvelabs_ctor.assert_called_once_with(api_key="secret-key")
    assert captured["model_name"] == "pegasus1.5"
    assert captured["prompt"] == "What happens in this video?"
    assert captured["max_tokens"] == 1024
    assert captured["temperature"] == 0.3
    assert captured["video"].url == "https://example.com/video.mp4"


def test_twelvelabs_block_run_returns_output(monkeypatch) -> None:
    # given
    captured: dict = {}
    _install_fake_twelvelabs_sdk(
        monkeypatch, captured, response_data="A summary of the clip."
    )
    block = TwelveLabsPegasusBlockV1()

    # when
    result = block.run(
        video_url="https://example.com/video.mp4",
        prompt="Summarize this video.",
        api_key="secret-key",
        model_version="pegasus1.5",
        max_tokens=2048,
        temperature=None,
    )

    # then
    assert result == {"output": "A summary of the clip."}


def test_analyze_video_with_pegasus_handles_empty_response(monkeypatch) -> None:
    # given - the SDK may return `data=None`; the block must coerce to a string
    captured: dict = {}
    _install_fake_twelvelabs_sdk(monkeypatch, captured, response_data=None)

    # when
    output = analyze_video_with_pegasus(
        video_url="https://example.com/video.mp4",
        prompt="Describe this.",
        api_key="secret-key",
        model_version="pegasus1.5",
        max_tokens=512,
        temperature=None,
    )

    # then
    assert output == ""
