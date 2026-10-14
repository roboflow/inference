"""Legacy behaviour kept for the Roboflow-platform blocks after they became
core blocks of roboflow_workflows: the 1.6.1-1.7.2 plugin path, the API
fallback for engines built without a platform binding, and the dataset-upload
error message."""

import importlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

# isort: off
import inference.core  # noqa: F401

# isort: on
from roboflow_workflows.core_steps.integrations.roboflow.visual_search.v1 import (
    RoboflowVisualSearchBlockV1,
)
from roboflow_workflows.core_steps.sinks.roboflow.custom_metadata import (
    v1 as custom_metadata_v1,
)
from roboflow_workflows.core_steps.sinks.roboflow.dataset_upload import (
    v1 as dataset_upload_v1,
)
from roboflow_workflows.core_steps.sinks.roboflow.dataset_upload import (
    v1_tensor as dataset_upload_v1_tensor,
)
from roboflow_workflows.prototypes.platform_client import (
    OFFLINE_PLATFORM_CLIENT,
    RoboflowPlatformClient,
    resolve_platform_blocks_client,
)

from inference._workflows_compat_inventory import (
    _PLATFORM_BLOCKS_PLUGIN_INVENTORY_LEGACY,
    _PLATFORM_BLOCKS_PLUGIN_INVENTORY_PACKAGES,
)
from inference.core import roboflow_api
from inference.core.cache import MemoryCache
from inference.core.env import ROBOFLOW_BLOCKS_PLUGIN
from inference.core.interfaces.roboflow_platform_client import SERVER_PLATFORM_CLIENT
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_blocks_from_plugin,
)

PLUGIN_PREFIX = "inference.roboflow_workflows_plugin."
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
PLATFORM_CLIENT_BLOCKS = [
    ("sinks.roboflow.dataset_upload.v1", "RoboflowDatasetUploadBlockV1"),
    ("sinks.roboflow.dataset_upload.v1_tensor", "RoboflowDatasetUploadBlockV1"),
    ("sinks.roboflow.dataset_upload.v2", "RoboflowDatasetUploadBlockV2"),
    ("sinks.roboflow.dataset_upload.v2_tensor", "RoboflowDatasetUploadBlockV2"),
    ("sinks.roboflow.custom_metadata.v1", "RoboflowCustomMetadataBlockV1"),
    ("sinks.roboflow.custom_metadata.v1_tensor", "RoboflowCustomMetadataBlockV1"),
    (
        "sinks.roboflow.model_monitoring_inference_aggregator.v1",
        "ModelMonitoringInferenceAggregatorBlockV1",
    ),
    (
        "sinks.roboflow.model_monitoring_inference_aggregator.v1_tensor",
        "ModelMonitoringInferenceAggregatorBlockV1",
    ),
    ("sinks.roboflow.vision_events.v1", "RoboflowVisionEventsBlockV1"),
    ("sinks.roboflow.vision_events.v1_tensor", "RoboflowVisionEventsBlockV1"),
    (
        "sinks.roboflow.asset_library_attributes.v1",
        "RoboflowAssetLibraryAttributesBlockV1",
    ),
    ("integrations.roboflow.visual_search.v1", "RoboflowVisualSearchBlockV1"),
    (
        "integrations.roboflow.visual_search_classifier.v1",
        "RoboflowVisualSearchClassifierBlockV1",
    ),
    (
        "integrations.roboflow.visual_search_classifier.v1_tensor",
        "RoboflowVisualSearchClassifierBlockV1",
    ),
]


def _canonical_name(legacy_name: str) -> str:
    suffix = legacy_name[len(PLUGIN_PREFIX) :]
    group, _, rest = suffix.partition(".")
    canonical = f"roboflow_workflows.core_steps.{group}.roboflow"
    if rest:
        canonical = f"{canonical}.{rest}"
    return canonical


def test_the_legacy_plugin_loader_is_kept_and_loads_nothing() -> None:
    # when
    blocks = load_blocks_from_plugin(plugin_name=ROBOFLOW_BLOCKS_PLUGIN)

    # then
    assert ROBOFLOW_BLOCKS_PLUGIN == "inference.roboflow_workflows_plugin.loader"
    assert blocks == []


def test_every_module_of_the_legacy_plugin_is_inventoried() -> None:
    assert len(_PLATFORM_BLOCKS_PLUGIN_INVENTORY_LEGACY) == 28
    assert _PLATFORM_BLOCKS_PLUGIN_INVENTORY_PACKAGES <= (
        _PLATFORM_BLOCKS_PLUGIN_INVENTORY_LEGACY
    )


@pytest.mark.parametrize(
    "legacy_name", sorted(_PLATFORM_BLOCKS_PLUGIN_INVENTORY_LEGACY)
)
def test_legacy_plugin_submodule_import_returns_the_canonical_module(
    legacy_name: str,
) -> None:
    # when
    legacy_module = importlib.import_module(legacy_name)

    # then
    canonical_module = importlib.import_module(_canonical_name(legacy_name))
    assert legacy_module is canonical_module


def test_legacy_plugin_from_import_returns_the_canonical_class() -> None:
    from roboflow_workflows.core_steps.sinks.roboflow.dataset_upload.v2 import (
        RoboflowDatasetUploadBlockV2 as canonical_block,
    )

    from inference.roboflow_workflows_plugin.sinks.dataset_upload.v2 import (
        RoboflowDatasetUploadBlockV2 as legacy_block,
    )

    assert legacy_block is canonical_block


def test_a_pickle_written_against_the_legacy_plugin_path_still_loads() -> None:
    # given
    payload = (
        b"cinference.roboflow_workflows_plugin.sinks.dataset_upload.v1\n"
        b"RoboflowDatasetUploadBlockV1\np0\n."
    )

    # when
    loaded = pickle.loads(payload)

    # then
    assert loaded is dataset_upload_v1.RoboflowDatasetUploadBlockV1


def test_a_module_the_legacy_plugin_never_had_is_not_aliased() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            "inference.roboflow_workflows_plugin.sinks.dataset_upload.active_learning"
        )


VISUAL_SEARCH_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/visual_search@v1",
            "name": "search",
            "image": "$inputs.image",
            "workspace": "my-workspace",
            "target_project": "my-project",
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "message", "selector": "$steps.search.message"},
        {
            "type": "JsonField",
            "name": "error_status",
            "selector": "$steps.search.error_status",
        },
    ],
}


class _MainStyleClient:
    """A custom client written against the protocol main exposed."""

    def post(
        self, endpoint, api_key, payload=None, params=None, http_errors_handlers=None
    ):
        raise AssertionError("not used by the Roboflow-platform blocks")

    def build_api_headers(self, explicit_headers=None):
        return dict(explicit_headers or {})

    def build_weights_provider_headers(self, countinference=None, service_secret=None):
        return None

    def wrap_url(self, url: str) -> str:
        return url


class _MainStyleProtocolSubclass(RoboflowPlatformClient):
    """A custom client subclassing the protocol, written against main's methods."""

    def post(
        self, endpoint, api_key, payload=None, params=None, http_errors_handlers=None
    ):
        raise AssertionError("not used by the Roboflow-platform blocks")

    def build_api_headers(self, explicit_headers=None):
        return dict(explicit_headers or {})

    def build_weights_provider_headers(self, countinference=None, service_secret=None):
        return None

    def wrap_url(self, url: str) -> str:
        return url


def _block_class(module_suffix: str, class_name: str) -> type:
    module = importlib.import_module(f"roboflow_workflows.core_steps.{module_suffix}")
    return getattr(module, class_name)


def _construct_block(block_class: type, platform_client=OFFLINE_PLATFORM_CLIENT):
    init_parameters = {
        name: None
        for name in block_class.get_init_parameters()
        if name != "platform_client"
    }
    init_parameters["api_key"] = "my-key"
    return block_class(**init_parameters, platform_client=platform_client)


def _run_visual_search(init_parameters: dict) -> dict:
    execution_engine = ExecutionEngine.init(
        workflow_definition=VISUAL_SEARCH_WORKFLOW,
        init_parameters=init_parameters,
        max_concurrent_steps=1,
    )
    result = execution_engine.run(
        runtime_parameters={"image": np.zeros((8, 12, 3), dtype=np.uint8)}
    )
    return result[0]


@pytest.mark.parametrize(
    "init_parameters",
    [
        {"workflows_core.api_key": "my-key"},
        {
            "workflows_core.api_key": "my-key",
            "workflows_core.platform_client": _MainStyleClient(),
        },
        {
            "workflows_core.api_key": "my-key",
            "workflows_core.platform_client": _MainStyleProtocolSubclass(),
        },
    ],
    ids=[
        "no-platform-binding",
        "main-style-custom-client",
        "main-style-protocol-subclass",
    ],
)
def test_engine_built_directly_still_reaches_the_roboflow_api(
    init_parameters: dict,
) -> None:
    # given
    with mock.patch.object(
        roboflow_api,
        "search_project_images_at_roboflow",
        return_value={"results": []},
    ) as search_mock:
        # when
        result = _run_visual_search(init_parameters=init_parameters)

    # then
    assert result == {
        "message": "No visually similar images found.",
        "error_status": False,
    }
    search_mock.assert_called_once()
    assert search_mock.call_args.kwargs["api_key"] == "my-key"
    assert search_mock.call_args.kwargs["workspace"] == "my-workspace"


def test_block_constructed_directly_uses_the_server_client() -> None:
    # when
    block = RoboflowVisualSearchBlockV1(api_key="my-key")

    # then
    assert block._platform_client is SERVER_PLATFORM_CLIENT


def test_a_client_with_every_platform_operation_is_kept() -> None:
    # given
    platform_client = mock.MagicMock()

    # when
    resolved = resolve_platform_blocks_client(platform_client)

    # then
    assert resolved is platform_client
    assert resolve_platform_blocks_client(OFFLINE_PLATFORM_CLIENT) is (
        SERVER_PLATFORM_CLIENT
    )
    assert resolve_platform_blocks_client(_MainStyleClient()) is SERVER_PLATFORM_CLIENT


@pytest.mark.parametrize("module_suffix,class_name", PLATFORM_CLIENT_BLOCKS)
def test_a_protocol_subclass_with_only_main_operations_gets_the_server_client(
    module_suffix: str, class_name: str
) -> None:
    # given
    block_class = _block_class(module_suffix=module_suffix, class_name=class_name)

    # when
    block = _construct_block(
        block_class=block_class, platform_client=_MainStyleProtocolSubclass()
    )

    # then
    assert block._platform_client is SERVER_PLATFORM_CLIENT


def _main_era_pickle(block_class: type) -> bytes:
    block = _construct_block(block_class=block_class)
    state = dict(block.__dict__)
    del state["_platform_client"]
    if "_offloader" in state:
        module = sys.modules[block_class.__module__]
        state["_offloader"] = module.call_asset_library_attributes_endpoint
    main_era_block = block_class.__new__(block_class)
    main_era_block.__dict__.update(state)
    return pickle.dumps(main_era_block)


@pytest.mark.parametrize("module_suffix,class_name", PLATFORM_CLIENT_BLOCKS)
def test_a_block_pickled_with_main_era_state_gets_the_server_client(
    module_suffix: str, class_name: str
) -> None:
    # given
    block_class = _block_class(module_suffix=module_suffix, class_name=class_name)
    payload = _main_era_pickle(block_class=block_class)

    # when
    restored = pickle.loads(payload)

    # then
    assert restored._platform_client is SERVER_PLATFORM_CLIENT
    assert restored._api_key == "my-key"


@pytest.mark.parametrize("module_suffix,class_name", PLATFORM_CLIENT_BLOCKS)
def test_restoring_saved_client_state_keeps_it_without_calling_the_fallback(
    module_suffix: str, class_name: str
) -> None:
    # given
    block_class = _block_class(module_suffix=module_suffix, class_name=class_name)
    saved_client = object()
    restored = block_class.__new__(block_class)

    # when
    with mock.patch.object(
        sys.modules[block_class.__module__],
        "resolve_platform_blocks_client",
        side_effect=RuntimeError("host unavailable"),
    ) as fallback_mock:
        restored.__setstate__({"_api_key": "my-key", "_platform_client": saved_client})

    # then
    assert restored._platform_client is saved_client
    fallback_mock.assert_not_called()


def test_a_visual_search_block_pickled_with_main_era_state_still_searches() -> None:
    # given
    main_era_block = RoboflowVisualSearchBlockV1.__new__(RoboflowVisualSearchBlockV1)
    main_era_block.__dict__.update({"_api_key": "my-key"})
    restored = pickle.loads(pickle.dumps(main_era_block))
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((8, 12, 3), dtype=np.uint8),
    )

    # when
    with mock.patch.object(
        roboflow_api,
        "search_project_images_at_roboflow",
        return_value={"results": []},
    ) as search_mock:
        result = restored.run(
            image=image, workspace="my-workspace", target_project="my-project"
        )

    # then
    assert result["message"] == "No visually similar images found."
    assert result["error_status"] is False
    search_mock.assert_called_once()
    assert search_mock.call_args.kwargs["api_key"] == "my-key"


CANONICAL_ONLY_CHILD_PROBE = """
import json
import os

from roboflow_workflows.core_steps.integrations.roboflow.visual_search.v1 import (
    RoboflowVisualSearchBlockV1,
)
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)

load_workflow_blocks()
block = RoboflowVisualSearchBlockV1(api_key="my-key")
print(
    json.dumps(
        {
            "plugins": os.environ.get("WORKFLOWS_PLUGINS"),
            "client": type(block._platform_client).__name__,
        }
    )
)
"""


def test_a_canonical_only_child_process_inherits_the_host_client() -> None:
    # given
    env = {
        **os.environ,
        "DISABLE_VERSION_CHECK": "True",
        "PYTHONPATH": os.pathsep.join(
            [
                str(REPO_ROOT),
                str(REPO_ROOT / "workflows"),
                str(REPO_ROOT / "inference_models"),
                str(REPO_ROOT / "stream_vision"),
            ]
        ),
        "ENABLE_TENSOR_DATA_REPRESENTATION": "False",
    }
    env.pop("MODEL_CACHE_DIR", None)

    # when
    result = subprocess.run(
        [sys.executable, "-c", CANONICAL_ONLY_CHILD_PROBE],
        env=env,
        capture_output=True,
        text=True,
    )

    # then
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["plugins"].split(",")[0] == ROBOFLOW_BLOCKS_PLUGIN
    assert report["client"] == "ServerRoboflowPlatformClient"


def test_legacy_helper_signature_without_platform_client_still_works() -> None:
    # given
    cache = MemoryCache()

    # when
    with mock.patch.object(
        roboflow_api, "get_roboflow_workspace", return_value="my-workspace"
    ) as workspace_mock, mock.patch.object(
        roboflow_api, "add_custom_metadata"
    ) as add_custom_metadata_mock:
        workspace = custom_metadata_v1.get_workspace_name("my-key", cache)
        result = custom_metadata_v1.add_custom_metadata_request(
            cache, "my-key", ["inference-id"], "location", "toronto"
        )

    # then
    assert workspace == "my-workspace"
    workspace_mock.assert_called_once_with(api_key="my-key")
    assert result == (False, "Custom metadata upload was successful")
    add_custom_metadata_mock.assert_called_once_with(
        api_key="my-key",
        workspace_id="my-workspace",
        inference_ids=["inference-id"],
        field_name="location",
        field_value="toronto",
    )


ZERO_SIZE_IMAGE_MESSAGE = (
    "Error while registration. Error type: "
    "<class 'inference.core.exceptions.InvalidNumpyInput'>. Details: Could not "
    "prepare image for registration because its dimensions are invalid: "
    "width=10, height=0."
)


@pytest.mark.parametrize(
    "module", [dataset_upload_v1, dataset_upload_v1_tensor], ids=["numpy", "tensor"]
)
def test_zero_size_image_message_names_the_legacy_error(module) -> None:
    # given
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((0, 10, 3), dtype=np.uint8),
    )

    # when
    with mock.patch.object(
        roboflow_api, "get_roboflow_workspace", return_value="my-workspace"
    ), mock.patch.object(roboflow_api, "register_image_at_roboflow") as register_mock:
        result = module.execute_registration(
            image=image,
            prediction=None,
            target_project="my-project",
            persist_predictions=False,
            usage_quota_name="my-quota",
            minutely_usage_limit=10,
            hourly_usage_limit=100,
            daily_usage_limit=1000,
            max_image_size=(128, 128),
            compression_level=75,
            registration_tags=[],
            labeling_batch_prefix="my-batch",
            new_labeling_batch_frequency="never",
            cache=MemoryCache(),
            api_key="my-key",
        )

    # then
    assert result == (True, ZERO_SIZE_IMAGE_MESSAGE)
    register_mock.assert_not_called()


def _usage_after(stored_value: int, *, consume: bool) -> int:
    from roboflow_workflows.core_steps.sinks.roboflow.dataset_upload import (
        active_learning,
    )

    cache = MemoryCache()
    for limit_type in active_learning.StrategyLimitType:
        cache.set(
            key=active_learning.generate_cache_key_for_active_learning_usage(
                limit_type=limit_type,
                workspace="ws",
                project="proj",
                strategy_name="quota",
            ),
            value={active_learning.USAGE_KEY: stored_value},
        )
    if consume:
        active_learning.use_credit_of_matching_strategy(
            cache=cache,
            workspace="ws",
            project="proj",
            matching_strategies_limits={"quota": []},
        )
    else:
        active_learning.return_strategy_credit(
            cache=cache, workspace="ws", project="proj", strategy_name="quota"
        )
    usage = active_learning.get_current_strategy_limit_usage(
        cache=cache,
        workspace="ws",
        project="proj",
        strategy_name="quota",
        limit_type=active_learning.StrategyLimitType.DAILY,
    )
    return usage


@pytest.mark.parametrize(
    "stored_value,consume,expected",
    [(-5, True, -4), (0, True, 1), (0, False, 0), (3, False, 2)],
)
def test_quota_arithmetic_matches_the_legacy_server(
    stored_value: int, consume: bool, expected: int
) -> None:
    assert _usage_after(stored_value, consume=consume) == expected
