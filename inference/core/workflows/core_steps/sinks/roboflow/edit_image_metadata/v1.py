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

SHORT_DESCRIPTION = "Write metadata and tags to existing Roboflow images."

LONG_DESCRIPTION = """
Update existing Roboflow source images with workflow-derived metadata and tags, enabling Asset Library enrichment workflows where model outputs become filterable image fields.

## How This Block Works

This block writes metadata key-value pairs and tags back to existing images in your Roboflow workspace. The block:

1. Receives Roboflow source image IDs, optional metadata, and optional tags
2. Resolves the target workspace from the configured Roboflow API key
3. Skips rows where both metadata and tags are empty
4. Merges duplicate source IDs using sequential semantics: later metadata values win, and tags are added as a de-duplicated set
5. Returns one status result per input source ID, in input order

Re-running the same workflow against the same source IDs is safe: metadata keys are upserted (last write wins) and tags are unioned. There is no destructive write.

The block does not send image bytes and does not create new images. It only updates existing source images. Removing metadata keys, removing tags, writing annotations, and creating images are intentionally out of scope for this workflow block.

## Requirements

This block requires a valid Roboflow API key. The API key determines the workspace whose source images can be updated.
"""

MAX_BATCH_UPDATES = 1000
WORKSPACE_NAME_CACHE_EXPIRE = 900  # 15 min
SKIPPED_EMPTY_UPDATE_MESSAGE = "Skipped because no metadata or tags were provided"
UPDATE_SUCCESS_MESSAGE = "Metadata updated"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Edit Image Metadata",
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
    type: Literal["roboflow_core/edit_image_metadata@v1"]
    source_id: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Roboflow source image ID to update. For batch workflows, provide one source ID per image.",
        examples=["$inputs.source_id", "source_abc123"],
    )
    metadata: Union[
        Dict[str, Union[str, int, float, bool, Selector()]],
        Selector(kind=[DICTIONARY_KIND]),
    ] = Field(
        default_factory=dict,
        description=(
            "Optional key-value metadata to set on the image. Either an inline "
            "dict whose values may be static or selector references (e.g. "
            "`$inputs.camera_id`), or a whole-field selector to a per-row dict "
            "produced by an upstream step."
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
            "Optional tags to add to the image. Each entry may be a static "
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
        description="If True, the block execution is disabled and no metadata writes occur.",
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


class UpdateMetadataOffloader(Protocol):
    """Callable contract for offloading the Roboflow metadata-update request.

    Implementations decide how to actually deliver the updates — call the
    Roboflow API inline, enqueue to a background worker, write to a log, etc.
    """

    def __call__(
        self,
        workspace_id: str,
        updates: List[Dict[str, Any]],
        api_key: str,
    ) -> Dict[str, Any]: ...


def call_image_metadata_endpoint(
    workspace_id: str,
    updates: List[Dict[str, Any]],
    api_key: str,
) -> Dict[str, Any]:
    if len(updates) == 1:
        return call_single_endpoint(
            workspace_id=workspace_id, update=updates[0], api_key=api_key
        )
    return call_batch_endpoint(
        workspace_id=workspace_id, updates=updates, api_key=api_key
    )


class EditImageMetadataBlockV1(WorkflowBlock):

    def __init__(
        self,
        api_key: Optional[str],
        cache: BaseCache,
        update_metadata_offloader: Optional[UpdateMetadataOffloader] = None,
    ):
        self._api_key = api_key
        self._cache = cache
        self._offloader: UpdateMetadataOffloader = (
            update_metadata_offloader or call_image_metadata_endpoint
        )

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache", "update_metadata_offloader"]

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
                "EditImageMetadata block cannot run without Roboflow API key. "
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
                f"EditImageMetadata block supports at most {MAX_BATCH_UPDATES} updates per run(). "
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


def build_effective_updates(
    source_ids: Batch[str],
    metadata: Optional[Union[Dict[str, Any], Batch[Optional[Dict[str, Any]]]]],
    tags: Optional[Union[List[str], Batch[Optional[List[str]]]]],
) -> EffectiveUpdates:
    n = len(source_ids)
    metadata_values: List[Optional[Dict[str, Any]]] = (
        list(metadata) if isinstance(metadata, Batch) else [metadata] * n
    )
    tag_values: List[Optional[List[str]]] = (
        list(tags) if isinstance(tags, Batch) else [tags] * n
    )
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


def call_single_endpoint(
    workspace_id: str,
    update: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    image_id = update["imageId"]
    try:
        update_image_metadata_at_roboflow(
            api_key=api_key,
            workspace_id=workspace_id,
            image_id=image_id,
            metadata=update.get("metadata"),
            add_tags=update.get("addTags"),
        )
        logging.info(
            "Updated image metadata: workspace_id=%s image_id=%s",
            workspace_id,
            image_id,
        )
        return {"error_status": False, "message": UPDATE_SUCCESS_MESSAGE}
    except Exception as error:
        logging.warning(
            "Could not update metadata: workspace_id=%s image_id=%s reason=%s",
            workspace_id,
            image_id,
            error,
        )
        return {
            "error_status": True,
            "message": f"Error while updating image metadata. Error type: {type(error)}. Details: {error}",
        }


def call_batch_endpoint(
    workspace_id: str,
    updates: List[Dict[str, Any]],
    api_key: str,
) -> Dict[str, Any]:
    response = batch_update_image_metadata_at_roboflow(
        api_key=api_key,
        workspace_id=workspace_id,
        updates=updates,
    )
    if not response.get("taskId"):
        raise ValueError("Malformed image metadata batch response: missing taskId")
    logging.info(
        "Submitted image metadata batch update: workspace_id=%s updates=%d",
        workspace_id,
        len(updates),
    )
    return {"error_status": False, "message": UPDATE_SUCCESS_MESSAGE}
