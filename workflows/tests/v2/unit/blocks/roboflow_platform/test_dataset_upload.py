"""Dataset collection wiring and deferred reservation boundary."""

from dataclasses import fields
from threading import Event

import pytest
import torch
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.roboflow_dataset_upload.v2 import DatasetUploadBlock
from roboflow_workflows.v2.blocks.sinks.runtime import (
    BACKGROUND_MESSAGE,
    FULL_MESSAGE,
    BackgroundEffects,
    EffectStats,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError

NO_EFFECTS = EffectStats(**{field.name: 0 for field in fields(EffectStats)})


def image():
    return ImageData.from_tensor(
        torch.full((3, 8, 12), 90, dtype=torch.uint8), image_id="fixture"
    )


def arguments(**changes):
    args = dict(
        images=Batch.of([image()]),
        predictions=None,
        target_project="project",
        usage_quota_name="shared",
        data_percentage=100,
        minutely_usage_limit=1,
        hourly_usage_limit=10,
        daily_usage_limit=100,
        max_image_size=(10, 10),
        compression_level=95,
        registration_tags=["tag"],
        persist_predictions=True,
        disable_sink=False,
        fire_and_forget=False,
        labeling_batch_prefix="collection",
        labeling_batches_recreation_frequency="never",
        image_name=None,
        metadata={"camera": 2},
    )
    args.update(changes)
    return args


class Port:
    def __init__(self):
        self.calls = []

    def upload(self, request, *, api_key):
        assert isinstance(api_key, Secret)
        self.calls.append(request)
        return False, "Successfully registered image"


def test_row_order_scalar_batch_values_and_snapshots():
    port = Port()
    args = arguments(
        images=Batch.of([image(), image()]),
        image_name=Batch.of(["first", "last"]),
        metadata=Batch.of([{"x": 1}, {"x": 2}]),
        registration_tags=["static", Batch.of(["a", "b"])],
    )
    result = DatasetUploadBlock(dataset_upload_port=port, api_key="invented").run(
        **args
    )
    assert len(result) == 2
    assert [call.image_name for call in port.calls] == ["first", "last"]
    assert [call.registration_tags for call in port.calls] == [
        ["static", "a"],
        ["static", "b"],
    ]
    assert [call.metadata for call in port.calls] == [{"x": 1}, {"x": 2}]
    args["images"][0].tensor_image.zero_()
    assert port.calls[0].image.tensor_image.max().item() == 90


def test_sampling_skip_before_port_or_image_copy(monkeypatch):
    port = Port()
    monkeypatch.setattr(
        "roboflow_workflows.v2.blocks.roboflow_platform.roboflow_dataset_upload.v2.random.random",
        lambda: 0.5,
    )
    result = DatasetUploadBlock(dataset_upload_port=port, api_key="invented").run(
        **arguments(data_percentage=50)
    )
    assert result == [
        {
            "error_status": False,
            "message": "Registration skipped due to sampling settings",
        }
    ]
    assert not port.calls


@pytest.mark.parametrize(
    "policy,step,key", [(True, False, None), (False, True, "invented")]
)
def test_disabled_writes_nothing(policy, step, key):
    port = Port()
    result = DatasetUploadBlock(
        dataset_upload_port=port, api_key=key, disable_sinks=policy
    ).run(**arguments(disable_sink=step))
    assert result[0]["error_status"] is False
    assert "disabled" in result[0]["message"]
    assert not port.calls


def test_key_validation_and_empty_batch():
    with pytest.raises(ValueError, match="API key"):
        DatasetUploadBlock().run(**arguments(disable_sink=True))
    with pytest.raises(SecretUnavailableError):
        DatasetUploadBlock(api_key=Secret.unavailable("RF_KEY")).run(**arguments())
    assert (
        DatasetUploadBlock(api_key="invented").run(**arguments(images=Batch.of([])))
        == []
    )


OFFLINE_MESSAGE = (
    "The host provided no 'dataset_upload_port' resource. Pass "
    "resources={'dataset_upload_port': "
    "RoboflowAssetsAdapter(cache=shared_cache)} when creating the session. "
    "Import RoboflowAssetsAdapter from "
    "inference.roboflow_workflows_plugin.v2.platform_assets_adapter; "
    "shared_cache must be a caller-owned BaseCache with quota locks."
)


def test_persist_false_removes_prediction():
    port = Port()
    DatasetUploadBlock(dataset_upload_port=port, api_key="invented").run(
        **arguments(predictions=object(), persist_predictions=False)
    )
    assert port.calls[0].prediction is None


@pytest.mark.parametrize("with_runner", [False, True])
@pytest.mark.parametrize("fire_and_forget", [False, True])
def test_missing_port_raises_before_sampling_or_admission(
    with_runner, fire_and_forget, monkeypatch
):
    sampled = []
    monkeypatch.setattr(
        "roboflow_workflows.v2.blocks.roboflow_platform.roboflow_dataset_upload.v2.random.random",
        lambda: sampled.append(True) or 0.0,
    )
    with BackgroundEffects() as effects:
        block = DatasetUploadBlock(
            api_key="invented", sink_effects=effects if with_runner else None
        )
        with pytest.raises(PlatformUnavailableError) as caught:
            block.run(**arguments(fire_and_forget=fire_and_forget))
        assert effects.drain(5)
    assert str(caught.value) == OFFLINE_MESSAGE
    assert not sampled
    assert effects.stats() == NO_EFFECTS
    assert effects.failures() == ()


def test_missing_port_keeps_disabled_empty_and_validation_outcomes():
    with BackgroundEffects() as effects:
        block = DatasetUploadBlock(api_key="invented", sink_effects=effects)
        disabled = block.run(**arguments(disable_sink=True, fire_and_forget=True))
        empty = block.run(**arguments(images=Batch.of([]), fire_and_forget=True))
        by_policy = DatasetUploadBlock(disable_sinks=True, sink_effects=effects).run(
            **arguments(fire_and_forget=True)
        )
        with pytest.raises(ValueError, match="API key"):
            DatasetUploadBlock(sink_effects=effects).run(**arguments())
        with pytest.raises(ValueError, match="batch length"):
            block.run(**arguments(image_name=Batch.of(["a", "b"])))
    assert disabled[0]["error_status"] is False and "disabled" in disabled[0]["message"]
    assert empty == []
    assert by_policy[0]["error_status"] is False
    assert effects.stats() == NO_EFFECTS


def test_configured_port_unavailability_stays_a_row_or_runner_failure():
    class UnavailablePort:
        def upload(self, request, *, api_key):
            raise PlatformUnavailableError("configured host adapter is down")

    port = UnavailablePort()
    inline = DatasetUploadBlock(dataset_upload_port=port, api_key="invented").run(
        **arguments()
    )
    assert inline[0]["error_status"] is True
    assert "configured host adapter is down" in inline[0]["message"]
    with BackgroundEffects() as effects:
        admitted = DatasetUploadBlock(
            dataset_upload_port=port, api_key="invented", sink_effects=effects
        ).run(**arguments(fire_and_forget=True))
        assert effects.drain(5)
    assert admitted == [{"error_status": False, "message": BACKGROUND_MESSAGE}]
    assert effects.stats().failed == 1


def test_rejected_scheduling_never_calls_upload():
    started, release = Event(), Event()

    def held():
        started.set()
        assert release.wait(5)
        return False, "done"

    port = Port()
    with BackgroundEffects(max_pending=1, when_full="reject") as effects:
        effects.deliver(held, label="held")
        assert started.wait(5)
        try:
            result = DatasetUploadBlock(
                dataset_upload_port=port, api_key="invented", sink_effects=effects
            ).run(**arguments(fire_and_forget=True))
            assert result == [{"error_status": True, "message": FULL_MESSAGE}]
            assert not port.calls
        finally:
            release.set()


def test_background_calls_port_after_admission_and_drains():
    port = Port()
    with BackgroundEffects() as effects:
        result = DatasetUploadBlock(
            dataset_upload_port=port, api_key="invented", sink_effects=effects
        ).run(**arguments(fire_and_forget=True))
        assert result[0]["error_status"] is False
    assert len(port.calls) == 1
    assert effects.stats().completed == 1
