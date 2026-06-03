import hashlib
import logging
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Protocol, Type, Union

from pydantic import ConfigDict, Field

from inference.core.cache.base import BaseCache
from inference.core.roboflow_api import (
    batch_update_image_metadata_at_roboflow,
    get_roboflow_workspace,
    update_image_metadata_at_roboflow,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Update attributes and tags for Asset Library images."

LONG_DESCRIPTION = """
Submit attribute and tag updates for existing Asset Library images, enabling enrichment workflows where model outputs become filterable image fields.

## How This Block Works

This block submits key-value attributes and tags for existing Asset Library images in your Roboflow workspace. Attribute values are stored as image metadata. The block:

1. Receives Asset Library source image IDs, optional attributes, and optional tags
2. Resolves the target workspace from the configured Roboflow API key
3. Skips rows where both attributes and tags are empty
4. Merges duplicate source IDs using sequential semantics: later attribute values win, and tags are added as a de-duplicated set
5. Submits one batch update request and returns one submission status per input source ID, in input order

Re-running the same workflow against the same source IDs is safe: attribute keys are upserted (last write wins) and tags are unioned. There is no destructive write.

The block does not send image bytes and does not create new images. It only updates existing Asset Library source images. Removing attribute keys, removing tags, writing annotations, and creating images are intentionally out of scope for this workflow block.

## Requirements

This block requires a valid Roboflow API key. The API key determines the workspace whose Asset Library images can be updated.
"""

MAX_BATCH_UPDATES = 1000
WORKSPACE_NAME_CACHE_EXPIRE = 900  # 15 min
SKIPPED_EMPTY_UPDATE_MESSAGE = "Skipped because no attributes or tags were provided"
UPDATE_SUCCESS_MESSAGE = "Image attributes update submitted"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Roboflow Asset Library Attributes",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "data_storage",
                "icon": "fal fa-tags",
                "blockPriority": 1,
                "requires_rf_key": True,
            },
        }
    )
    type: Literal["roboflow_core/asset_library_attributes@v1"]
    source_id: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Asset Library source image ID to update. For batch workflows, provide one source ID per image.",
        examples=["$inputs.source_id", "source_abc123"],
    )
    metadata: Union[
        Dict[str, Union[str, int, float, bool, Selector()]],
        Selector(kind=[DICTIONARY_KIND]),
    ] = Field(
        default_factory=dict,
        description=(
            "Optional key-value attributes to set on the Asset Library image. "
            "Attributes are stored as image metadata. Either an inline dict whose "
            "values may be static or selector references (e.g. `$inputs.camera_id`), "
            "or a whole-field selector to a per-row dict produced by an upstream step."
        ),
        examples=[
            {"color": "red", "score": 0.98},
            {"camera": "$inputs.camera_id", "source": "edge"},
            "$steps.classifier.top_label_dict",
        ],
    )
    tags: Optional[
        Union[
            List[Union[Selector(kind=[STRING_KIND]), str]],
            Selector(kind=[LIST_OF_VALUES_KIND]),
        ]
    ] = Field(
        default=None,
        description=(
            "Optional tags to add to the Asset Library image. Each entry may be a static "
            "string or a reference to a workflow input/step (e.g. `$inputs.label`)."
        ),
        examples=[
            ["auto-labeled", "red"],
            ["static-tag", "$inputs.label"],
            "$inputs.tags",
        ],
    )
    disable_sink: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="If True, the block execution is disabled and no Asset Library attribute writes occur.",
        examples=[False, "$inputs.disable_sink"],
    )

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["source_id"]

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return ["metadata", "tags"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.10.0,<2.0.0"


class UpdateAssetLibraryAttributesOffloader(Protocol):
    """Callable contract for offloading the Roboflow image-attributes request.

    Implementations decide how to actually deliver the updates — call the
    Roboflow API inline, enqueue to a background worker, write to a log, etc.
    """

    def __call__(
        self,
        workspace_id: str,
        updates: List[Dict[str, Any]],
        api_key: str,
    ) -> Dict[str, Any]: ...


def call_asset_library_attributes_endpoint(
    workspace_id: str,
    updates: List[Dict[str, Any]],
    api_key: str,
) -> Dict[str, Any]:
    if len(updates) == 1:
        return call_single_image_endpoint(
            workspace_id=workspace_id, update=updates[0], api_key=api_key
        )
    return call_batch_endpoint(
        workspace_id=workspace_id, updates=updates, api_key=api_key
    )


class RoboflowAssetLibraryAttributesBlockV1(WorkflowBlock):

    def __init__(
        self,
        api_key: Optional[str],
        cache: BaseCache,
        update_attributes_offloader: Optional[
            UpdateAssetLibraryAttributesOffloader
        ] = None,
    ):
        self._api_key = api_key
        self._cache = cache
        self._offloader: UpdateAssetLibraryAttributesOffloader = (
            update_attributes_offloader or call_asset_library_attributes_endpoint
        )

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache", "update_attributes_offloader"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        source_id: Batch[str],
        metadata: Optional[
            Union[Dict[str, Any], Batch[Optional[Dict[str, Any]]]]
        ] = None,
        tags: Optional[Union[List[str], Batch[Optional[List[str]]]]] = None,
        disable_sink: bool = False,
    ) -> BlockResult:
        if self._api_key is None:
            raise ValueError(
                "Roboflow Asset Library Attributes block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                "retrieve one."
            )
        if disable_sink:
            return [
                {
                    "error_status": False,
                    "message": "Sink was disabled by parameter `disable_sink`",
                }
                for _ in range(len(source_id))
            ]

        if len(source_id) > MAX_BATCH_UPDATES:
            raise ValueError(
                f"Roboflow Asset Library Attributes block supports at most {MAX_BATCH_UPDATES} updates per run(). "
                "Reduce the workflow batch size."
            )

        workspace_id = get_workspace_name(api_key=self._api_key, cache=self._cache)
        effective = build_effective_updates(
            source_ids=source_id,
            metadata=metadata,
            tags=tags,
        )

        if not effective.updates:
            return effective.results

        submitted_result = self._offloader(
            workspace_id=workspace_id,
            updates=effective.updates,
            api_key=self._api_key,
        )

        for update in effective.updates:
            for result_index in effective.result_indices_by_id[update["imageId"]]:
                effective.results[result_index] = submitted_result
        return effective.results


def get_workspace_name(api_key: str, cache: BaseCache) -> str:
    api_key_hash = hashlib.md5(
        api_key.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cached_workspace_name = cache.get(cache_key)
    if cached_workspace_name:
        return cached_workspace_name
    workspace_name_from_api = get_roboflow_workspace(api_key=api_key)
    cache.set(
        key=cache_key, value=workspace_name_from_api, expire=WORKSPACE_NAME_CACHE_EXPIRE
    )
    return workspace_name_from_api


class EffectiveUpdates(NamedTuple):
    updates: List[Dict[str, Any]]
    result_indices_by_id: Dict[str, List[int]]
    results: List[Optional[Dict[str, Any]]]


def _resolve_element(v: Any, index: int) -> Any:
    return v[index] if isinstance(v, Batch) else v


def _normalize_to_per_row(value: Any, n: int) -> List[Any]:
    if isinstance(value, Batch):
        return list(value)
    if value is None:
        return [None] * n
    if isinstance(value, dict):
        if not any(isinstance(v, Batch) for v in value.values()):
            return [value] * n
        return [{k: _resolve_element(v, i) for k, v in value.items()} for i in range(n)]
    if isinstance(value, list):
        if not any(isinstance(v, Batch) for v in value):
            return [value] * n
        return [[_resolve_element(v, i) for v in value] for i in range(n)]
    return [value] * n


def build_effective_updates(
    source_ids: Batch[str],
    metadata: Optional[Union[Dict[str, Any], Batch[Optional[Dict[str, Any]]]]],
    tags: Optional[Union[List[str], Batch[Optional[List[str]]]]],
) -> EffectiveUpdates:
    n = len(source_ids)
    metadata_values = _normalize_to_per_row(metadata, n)
    tag_values = _normalize_to_per_row(tags, n)
    results: List[Optional[Dict[str, Any]]] = [None] * n
    updates_by_source_id: Dict[str, Dict[str, Any]] = {}
    result_indices_by_id: Dict[str, List[int]] = {}

    for result_index, (source, image_metadata, image_tags) in enumerate(
        zip(source_ids, metadata_values, tag_values)
    ):
        if not image_metadata and not image_tags:
            results[result_index] = {
                "error_status": False,
                "message": SKIPPED_EMPTY_UPDATE_MESSAGE,
            }
            continue

        merged = updates_by_source_id.setdefault(source, {"imageId": source})
        if image_metadata:
            merged.setdefault("metadata", {}).update(image_metadata)
        if image_tags:
            combined = [*merged.get("addTags", []), *image_tags]
            merged["addTags"] = list(dict.fromkeys(reversed(combined)))[::-1]
        result_indices_by_id.setdefault(source, []).append(result_index)

    return EffectiveUpdates(
        updates=list(updates_by_source_id.values()),
        result_indices_by_id=result_indices_by_id,
        results=results,
    )


def _extract_response_body(error: Exception) -> str:
    for exc in (error, getattr(error, "__cause__", None)):
        response = getattr(exc, "response", None)
        if response is None:
            continue
        try:
            return str(response.json())
        except Exception:
            pass
        try:
            return response.text[:500]
        except Exception:
            pass
    return ""


def _format_api_error(prefix: str, error: Exception) -> str:
    body = _extract_response_body(error)
    detail = f"Error type: {type(error).__name__}. Details: {error}"
    if body:
        detail += f". Response body: {body}"
    return f"{prefix}. {detail}"


def call_single_image_endpoint(
    workspace_id: str,
    update: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    try:
        update_image_metadata_at_roboflow(
            api_key=api_key,
            workspace_id=workspace_id,
            image_id=update["imageId"],
            metadata=update.get("metadata"),
            add_tags=update.get("addTags"),
        )
    except Exception as error:
        message = _format_api_error(
            "Error while updating Asset Library attributes", error
        )
        logging.warning(
            "Could not update Asset Library attributes for image %s: %s",
            update["imageId"],
            message,
        )
        return {"error_status": True, "message": message}
    return {"error_status": False, "message": UPDATE_SUCCESS_MESSAGE}


def call_batch_endpoint(
    workspace_id: str,
    updates: List[Dict[str, Any]],
    api_key: str,
) -> Dict[str, Any]:
    try:
        response = batch_update_image_metadata_at_roboflow(
            api_key=api_key,
            workspace_id=workspace_id,
            updates=updates,
        )
        if not response.get("taskId"):
            raise ValueError("Malformed image metadata batch response: missing taskId")
    except Exception as error:
        message = _format_api_error(
            "Error while submitting Asset Library attributes update", error
        )
        logging.warning(
            "Could not submit Asset Library attributes batch update: %s", message
        )
        return {"error_status": True, "message": message}
    logging.info(
        "Submitted Asset Library attributes batch update: updates=%d",
        len(updates),
    )
    return {"error_status": False, "message": UPDATE_SUCCESS_MESSAGE}
