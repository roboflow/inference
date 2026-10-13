"""Dataset collection wiring and deferred reservation boundary."""

from threading import Event

import pytest
import torch
from roboflow_workflows.v2.blocks.roboflow_platform.dataset_upload import (
    DatasetUploadBlock,
)
from roboflow_workflows.v2.blocks.sinks.runtime import FULL_MESSAGE, BackgroundEffects
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError


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
        "roboflow_workflows.v2.blocks.roboflow_platform.dataset_upload.random.random",
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


def test_persist_false_removes_prediction_and_offline_is_actionable():
    port = Port()
    DatasetUploadBlock(dataset_upload_port=port, api_key="invented").run(
        **arguments(predictions=object(), persist_predictions=False)
    )
    assert port.calls[0].prediction is None
    result = DatasetUploadBlock(api_key="invented").run(**arguments())
    assert result[0]["error_status"] is True
    assert result[0]["message"] == (
        "PlatformUnavailableError: "
        "The host provided no 'dataset_upload_port' resource. Pass "
        "resources={'dataset_upload_port': "
        "RoboflowAssetsAdapter(cache=shared_cache)} when creating the session. "
        "Import RoboflowAssetsAdapter from "
        "inference.roboflow_workflows_plugin.v2.platform_assets_adapter; "
        "shared_cache must be a caller-owned BaseCache with quota locks."
    )


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
