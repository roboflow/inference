import hashlib
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

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
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    IMAGE_KIND,
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

1. Receives workflow images, Roboflow source image IDs, optional metadata, and optional tags
2. Resolves the target workspace from the configured Roboflow API key
3. Skips rows where both metadata and tags are empty
4. Merges duplicate source IDs using sequential semantics: later metadata values win, and tags are added as a de-duplicated set
5. Uses the synchronous single-image metadata endpoint for one effective update
6. Uses the asynchronous batch metadata endpoint for multiple effective updates
7. Returns one status result per input image, in input order

The block does not send image bytes and does not create new images. It only updates existing source images. Removing metadata keys, removing tags, writing annotations, and creating images are intentionally out of scope for this workflow block.

## Requirements

This block requires a valid Roboflow API key. The API key determines the workspace whose source images can be updated.
"""

MAX_BATCH_UPDATES = 1000
WORKSPACE_NAME_CACHE_EXPIRE = 900  # 15 min
SKIPPED_EMPTY_UPDATE_MESSAGE = "Skipped because no metadata or tags were provided"
SINGLE_UPDATE_SUCCESS_MESSAGE = "Metadata updated"


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
    type: Literal[
        "roboflow_core/edit_image_metadata@v1",
        "roboflow_core/roboflow_edit_image_metadata@v1",
        "EditImageMetadata",
    ]
    images: Selector(kind=[IMAGE_KIND]) = Field(
        description="Image(s) being processed. Used to align this sink with workflow batch dimensionality; image bytes are not sent to the metadata API.",
        examples=["$inputs.image"],
    )
    source_id: Union[str, Selector(kind=[STRING_KIND])] = Field(
        description="Roboflow source image ID to update. For batch workflows, provide one source ID per image.",
        examples=["$inputs.source_id", "source_abc123"],
    )
    metadata: Optional[Union[Dict[str, Any], Selector(kind=[DICTIONARY_KIND])]] = Field(
        default=None,
        description="Optional key-value metadata to set on the image. Metadata is stored as user_metadata and can be used for filtering in Roboflow.",
        examples=[{"color": "red", "score": 0.98}, "$inputs.metadata"],
    )
    tags: Optional[Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])]] = Field(
        default=None,
        description="Optional tags to add to the image.",
        examples=[["auto-labeled", "red"], "$inputs.tags"],
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
        return ["images", "source_id", "metadata", "tags"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class EditImageMetadataBlockV1(WorkflowBlock):

    def __init__(self, api_key: Optional[str], cache: BaseCache):
        self._api_key = api_key
        self._cache = cache

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key", "cache"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        source_id: Batch[str],
        metadata: Optional[Batch[Optional[Dict[str, Any]]]] = None,
        tags: Optional[Batch[Optional[List[str]]]] = None,
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
                for _ in range(len(images))
            ]

        workspace_id = get_workspace_name(api_key=self._api_key, cache=self._cache)
        updates, source_id_to_result_indices, results = build_effective_updates(
            images_count=len(images),
            source_ids=source_id,
            metadata=metadata,
            tags=tags,
        )

        if not updates:
            return results
        if len(updates) > MAX_BATCH_UPDATES:
            raise ValueError(
                f"EditImageMetadata block supports at most {MAX_BATCH_UPDATES} updates per run(). "
                "Reduce the workflow batch size."
            )

        if len(updates) == 1:
            submitted_result = call_single_endpoint(
                workspace_id=workspace_id,
                update=updates[0],
                api_key=self._api_key,
            )
        else:
            submitted_result = call_batch_endpoint(
                workspace_id=workspace_id,
                updates=updates,
                api_key=self._api_key,
            )

        for update in updates:
            for result_index in source_id_to_result_indices[update["imageId"]]:
                results[result_index] = submitted_result
        return results


def get_workspace_name(api_key: str, cache: BaseCache) -> str:
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cached_workspace_name = cache.get(cache_key)
    if cached_workspace_name:
        return cached_workspace_name
    workspace_name_from_api = get_roboflow_workspace(api_key=api_key)
    cache.set(
        key=cache_key, value=workspace_name_from_api, expire=WORKSPACE_NAME_CACHE_EXPIRE
    )
    return workspace_name_from_api


def build_effective_updates(
    images_count: int,
    source_ids: Batch[str],
    metadata: Optional[Batch[Optional[Dict[str, Any]]]],
    tags: Optional[Batch[Optional[List[str]]]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]], List[Dict[str, Any]]]:
    metadata_values = metadata or [None] * images_count
    tag_values = tags or [None] * images_count
    results: List[Optional[Dict[str, Any]]] = [None] * images_count
    updates_by_source_id: Dict[str, Dict[str, Any]] = {}
    source_id_to_result_indices: Dict[str, List[int]] = {}

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
            merged.setdefault("addTags", [])
            for tag in image_tags:
                if tag in merged["addTags"]:
                    merged["addTags"].remove(tag)
                merged["addTags"].append(tag)
        source_id_to_result_indices.setdefault(source, []).append(result_index)

    return list(updates_by_source_id.values()), source_id_to_result_indices, results  # type: ignore


def call_single_endpoint(
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
        return {"error_status": False, "message": SINGLE_UPDATE_SUCCESS_MESSAGE}
    except Exception as error:
        logging.warning(
            f"Could not update metadata for image ID: {update['imageId']}. Reason: {error}"
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
    task_id = response.get("taskId")
    if not task_id:
        raise ValueError(f"Malformed image metadata batch response: {response}")
    return {
        "error_status": False,
        "message": f"Submitted as async task {task_id}",
    }
