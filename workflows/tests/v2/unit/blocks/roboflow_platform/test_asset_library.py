"""Asset row broadcast, duplicate merge and no-write branches."""

import copy

import pytest
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.asset_library_attributes.v1 import EMPTY_MESSAGE, MAX_BATCH_UPDATES, SUCCESS_MESSAGE, AssetLibraryAttributesBlock
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.secrets import Secret, SecretUnavailableError


class Port:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def get_workspace(self, *, api_key):
        assert isinstance(api_key, Secret)
        self.calls.append("workspace")
        return "shop"

    def update_attributes(self, **arguments):
        self.calls.append(arguments)
        if self.error:
            raise self.error


def test_duplicate_merge_broadcast_row_order_and_no_input_mutation():
    port = Port()
    block = AssetLibraryAttributesBlock(asset_library_port=port, api_key="invented")
    metadata = {"color": Batch.of(["red", "blue", "green"]), "camera": "north"}
    tags = ["shared", Batch.of(["red", "shared", "green"])]
    before = copy.deepcopy((metadata, tags))
    result = block.run(
        source_id=Batch.of(["a", "b", "a"]),
        metadata=metadata,
        tags=tags,
        disable_sink=False,
    )
    assert result == [{"error_status": False, "message": SUCCESS_MESSAGE}] * 3
    assert port.calls[1]["updates"] == [
        {
            "imageId": "a",
            "metadata": {"color": "green", "camera": "north"},
            "addTags": ["red", "shared", "green"],
        },
        {
            "imageId": "b",
            "metadata": {"color": "blue", "camera": "north"},
            "addTags": ["shared"],
        },
    ]
    assert (metadata, tags) == before


def test_empty_duplicate_rows_stay_skipped():
    port = Port()
    block = AssetLibraryAttributesBlock(asset_library_port=port, api_key="invented")
    result = block.run(
        source_id=Batch.of(["a", "a", "b"]),
        metadata=Batch.of([{}, {"x": 1}, None]),
        tags=None,
        disable_sink=False,
    )
    assert [item["message"] for item in result] == [
        EMPTY_MESSAGE,
        SUCCESS_MESSAGE,
        EMPTY_MESSAGE,
    ]
    assert port.calls[1]["updates"] == [{"imageId": "a", "metadata": {"x": 1}}]


@pytest.mark.parametrize(
    "by_policy,disable,key,expected",
    [
        (True, False, None, "execution policy"),
        (False, True, "invented", "disable_sink"),
    ],
)
def test_disabled_calls_nothing(by_policy, disable, key, expected):
    port = Port()
    result = AssetLibraryAttributesBlock(
        asset_library_port=port, api_key=key, disable_sinks=by_policy
    ).run(source_id="a", metadata={}, tags=None, disable_sink=disable)
    assert expected in result["message"]
    assert result["error_status"] is False
    assert not port.calls


def test_missing_key_precedes_parameter_disable_and_replay_reveals_nothing():
    with pytest.raises(ValueError, match="API key"):
        AssetLibraryAttributesBlock().run(
            source_id="a", metadata={}, tags=None, disable_sink=True
        )
    with pytest.raises(SecretUnavailableError):
        AssetLibraryAttributesBlock(api_key=Secret.unavailable("RF_KEY")).run(
            source_id="a", metadata={}, tags=None, disable_sink=False
        )


def test_empty_updates_resolve_workspace_but_never_dispatch():
    port = Port()
    block = AssetLibraryAttributesBlock(asset_library_port=port, api_key="invented")
    assert (
        block.run(source_id="a", metadata={}, tags=[], disable_sink=False)["message"]
        == EMPTY_MESSAGE
    )
    assert port.calls == ["workspace"]
    assert (
        block.run(source_id=Batch.of([]), metadata={}, tags=[], disable_sink=False)
        == []
    )


def test_maximum_and_misaligned_batches_fail_without_update():
    port = Port()
    block = AssetLibraryAttributesBlock(asset_library_port=port, api_key="invented")
    with pytest.raises(ValueError, match="at most"):
        block.run(
            source_id=Batch.of(["x"] * (MAX_BATCH_UPDATES + 1)),
            metadata={},
            tags=None,
            disable_sink=False,
        )
    assert not port.calls
    with pytest.raises(ValueError, match="length"):
        block.run(
            source_id=Batch.of(["x", "y"]),
            metadata=Batch.of([{}]),
            tags=None,
            disable_sink=False,
        )


@pytest.mark.parametrize("sources", [["a"], ["a", "b"]])
def test_submission_error_applies_to_each_effective_row(sources):
    port = Port(ValueError("safe fake failure"))
    block = AssetLibraryAttributesBlock(asset_library_port=port, api_key="invented")
    results = block.run(
        source_id=Batch.of(sources), metadata={"x": 1}, tags=[], disable_sink=False
    )
    assert all(
        item["error_status"] and "safe fake failure" in item["message"]
        for item in results
    )


def test_offline_fails_before_reveal():
    with pytest.raises(PlatformUnavailableError) as error:
        AssetLibraryAttributesBlock(api_key="invented").run(
            source_id="x", metadata={"x": 1}, tags=None, disable_sink=False
        )
    assert str(error.value) == (
        "The host provided no 'asset_library_port' resource. Pass "
        "resources={'asset_library_port': "
        "RoboflowAssetsAdapter(cache=shared_cache)} when creating the session. "
        "Import RoboflowAssetsAdapter from "
        "inference.roboflow_workflows_plugin.v2.platform_assets_adapter; "
        "shared_cache must be a caller-owned BaseCache with quota locks."
    )
