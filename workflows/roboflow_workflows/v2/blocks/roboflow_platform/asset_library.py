"""Update existing Asset Library images, merging duplicate IDs as in V1."""

from typing import Any, Dict, List, Optional, Protocol, Sequence, Union

from pydantic import Field, StrictBool
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.blocks.roboflow_platform.ports import (
    WorkspacePort,
    api_key_for_dispatch,
)
from roboflow_workflows.v2.blocks.sinks.runtime import disabled_sink_message
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.secrets import Secret
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref

BATCH_ALWAYS = "always"
BATCH_IF_VARYING = "if_varying"


MAX_BATCH_UPDATES = 1000
SUCCESS_MESSAGE = "Image attributes update submitted"
EMPTY_MESSAGE = "Skipped because no attributes or tags were provided"


class AssetLibraryPort(WorkspacePort, Protocol):
    """Host operations for existing image attributes and tags."""

    def update_attributes(
        self, *, api_key: Secret, workspace_id: str, updates: Sequence[Dict[str, Any]]
    ) -> None:
        """Submit merged updates.

        Args:
            api_key: Unrevealed credential.
            workspace_id: Workspace owning the images.
            updates: Detached imageId/metadata/addTags payloads, in first-ID order.
        """


_OFFLINE_MESSAGE = (
    "The host provided no 'asset_library_port' resource. Pass "
    "resources={'asset_library_port': "
    "RoboflowAssetsAdapter(cache=shared_cache)} when creating the session. "
    "Import RoboflowAssetsAdapter from "
    "inference.roboflow_workflows_plugin.v2.platform_assets_adapter; "
    "shared_cache must be a caller-owned BaseCache with quota locks."
)


class _OfflineAssetLibraryPort:
    def get_workspace(self, *, api_key: Secret) -> str:
        raise PlatformUnavailableError(_OFFLINE_MESSAGE)

    def update_attributes(self, **arguments: Any) -> None:
        raise PlatformUnavailableError(_OFFLINE_MESSAGE)


def _per_row(value: Any, count: int) -> List[Any]:
    if isinstance(value, Batch):
        if len(value) != count:
            raise ValueError("Per-row values must match the source batch length")
        return list(value)

    if isinstance(value, dict):
        columns = {key: _per_row(item, count) for key, item in value.items()}
        return [
            {key: items[index] for key, items in columns.items()}
            for index in range(count)
        ]

    if isinstance(value, list):
        columns = [_per_row(item, count) for item in value]
        return [[items[index] for items in columns] for index in range(count)]

    return [value] * count


class AssetLibraryAttributesBlock(Block):
    """Upsert metadata and union tags; later duplicate values win.

    Args:
        asset_library_port: Caller-provided platform adapter; offline by default.
        api_key: String or Secret supplied by the host.
        disable_sinks: Execution policy disabling all writes.
    """

    type = "v2/asset_library_attributes@v1"
    aliases = ("RoboflowAssetLibraryAttributes",)
    metadata = {
        "name": "Roboflow Asset Library Attributes",
        "port_of": "roboflow_core/asset_library_attributes@v1",
    }
    outputs = {
        "error_status": Output(
            BOOLEAN_KIND,
            description="False on submission or skip; True on request failure.",
        ),
        "message": Output(STRING_KIND, description="Per-input submission outcome."),
    }

    class Params(BlockParams):
        source_id: str | Ref(STRING_KIND, batch=BATCH_ALWAYS) = Field(
            description="Existing Asset Library image ID, one per input row."
        )
        metadata: Dict[
            str,
            Union[str, int, float, bool, Ref(batch=BATCH_IF_VARYING)],
        ] | Ref(DICTIONARY_KIND, batch=BATCH_IF_VARYING) = Field(
            default_factory=dict,
            description="Attributes to upsert; selected values may vary by row.",
        )
        tags: Optional[List[str | Ref(STRING_KIND, batch=BATCH_IF_VARYING)]] | Ref(
            LIST_OF_VALUES_KIND,
            batch=BATCH_IF_VARYING,
        ) = Field(default=None, description="Tags to union with existing tags.")
        disable_sink: StrictBool | Ref(BOOLEAN_KIND) = Field(
            default=False, description="Skip updates when true."
        )

    def __init__(
        self,
        *,
        asset_library_port: AssetLibraryPort = _OfflineAssetLibraryPort(),
        api_key: Optional[Union[str, Secret]] = None,
        disable_sinks: bool = False,
    ) -> None:
        self._port = asset_library_port
        self._api_key = api_key
        self._disable_sinks = disable_sinks

    def run(
        self,
        *,
        source_id: Union[str, Batch[str]],
        metadata: Any,
        tags: Any,
        disable_sink: bool,
    ) -> Any:
        """Submit one merged update per nonempty ID and preserve input row order.

        Args:
            source_id: Scalar ID or engine batch.
            metadata: Shared dict, per-row Batch, or dict with Batch values.
            tags: Shared tags, per-row Batch, or list with Batch values.
            disable_sink: Step policy; API-key validation precedes this check as in V1.

        Returns:
            One outcome dict for a scalar ID, otherwise a list of outcome dicts.

        Raises:
            ValueError: Missing API key, too many rows, or mismatched batches.
            PlatformUnavailableError: No host adapter was supplied.
        """
        batched = isinstance(source_id, Batch)
        sources = list(source_id) if batched else [source_id]
        count = len(sources)
        if self._disable_sinks:
            results = [
                {
                    "error_status": False,
                    "message": disabled_sink_message(by_policy=True),
                }
                for _ in sources
            ]
        else:
            key = api_key_for_dispatch(
                self._api_key, block_name="Roboflow Asset Library Attributes"
            )
            if disable_sink:
                results = [
                    {
                        "error_status": False,
                        "message": disabled_sink_message(by_policy=False),
                    }
                    for _ in sources
                ]
            else:
                if count > MAX_BATCH_UPDATES:
                    raise ValueError(
                        f"Roboflow Asset Library Attributes supports at most {MAX_BATCH_UPDATES} updates per run(). Reduce the workflow batch size."
                    )

                workspace = self._port.get_workspace(api_key=key)
                results = self._submit(
                    sources, metadata=metadata, tags=tags, key=key, workspace=workspace
                )

        return results if batched else results[0]

    def _submit(
        self,
        sources: List[str],
        *,
        metadata: Any,
        tags: Any,
        key: Secret,
        workspace: str,
    ) -> List[Dict[str, Any]]:
        metadata_rows = _per_row(metadata, len(sources))
        tag_rows = _per_row(tags, len(sources))
        results = [{"error_status": False, "message": EMPTY_MESSAGE} for _ in sources]
        updates: Dict[str, Dict[str, Any]] = {}
        indices: Dict[str, List[int]] = {}
        for index, (source, attributes, image_tags) in enumerate(
            zip(sources, metadata_rows, tag_rows)
        ):
            if not attributes and not image_tags:
                continue

            update = updates.setdefault(source, {"imageId": source})
            if attributes:
                update.setdefault("metadata", {}).update(attributes)
            if image_tags:
                combined = [*update.get("addTags", []), *image_tags]
                update["addTags"] = list(dict.fromkeys(reversed(combined)))[::-1]
            indices.setdefault(source, []).append(index)

        if not updates:
            return results

        try:
            self._port.update_attributes(
                api_key=key, workspace_id=workspace, updates=list(updates.values())
            )
        except PlatformUnavailableError:
            raise
        except Exception as error:
            prefix = (
                "Error while updating Asset Library attributes"
                if len(updates) == 1
                else "Error while submitting Asset Library attributes update"
            )
            outcome = {
                "error_status": True,
                "message": f"{prefix}. Error type: {type(error).__name__}. Details: {error}",
            }
        else:
            outcome = {"error_status": False, "message": SUCCESS_MESSAGE}
        for source_indices in indices.values():
            for index in source_indices:
                results[index] = dict(outcome)

        return results
