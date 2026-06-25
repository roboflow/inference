from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from inference.core.roboflow_api import search_project_images_at_roboflow
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    DICTIONARY_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Find visually similar images in a Roboflow project."

LONG_DESCRIPTION = """
Search a Roboflow project for images that look similar to an input image and return the best match plus metadata.

## How This Block Works

This block uses the existing Roboflow project image search API:

1. Receives an input image from the workflow
2. Sends the image to Roboflow project search as `image_base64`
3. Requests useful fields such as image URL, tags, and user metadata
4. Returns the best match and the top matches list

The target project should already contain uploaded images. Roboflow indexes those images for visual search using the platform's existing image indexing pipeline. This block does not create images, update metadata, or manage the index.
"""

TopK = Annotated[int, Field(ge=1, le=50)]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Roboflow Visual Search",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "advanced",
                "icon": "far fa-search",
                "blockPriority": 1,
                "requires_rf_key": True,
            },
        }
    )
    type: Literal["roboflow_core/visual_search@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Image to use as the visual search query.",
        examples=["$inputs.image", "$steps.crop.crops"],
    )
    workspace: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Roboflow workspace URL slug that owns the target project.",
        examples=["my-workspace", "$inputs.workspace"],
    )
    target_project: Union[Selector(kind=[ROBOFLOW_PROJECT_KIND]), str] = Field(
        description="Roboflow project URL slug to search.",
        examples=["reference-images", "$inputs.target_project"],
    )
    top_k: Union[TopK, Selector(kind=[INTEGER_KIND])] = Field(
        default=1,
        description="Number of visually similar images to return. Use 1 when you only need the best match.",
        examples=[1, 5, "$inputs.top_k"],
    )

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["image"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="matched", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="match", kind=[DICTIONARY_KIND]),
            OutputDefinition(name="matches", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class RoboflowVisualSearchBlockV1(WorkflowBlock):
    def __init__(self, api_key: Optional[str]):
        self._api_key = api_key

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["api_key"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: Union[WorkflowImageData, Batch[WorkflowImageData]],
        workspace: str,
        target_project: str,
        top_k: int = 1,
    ) -> BlockResult:
        if self._api_key is None:
            raise ValueError(
                "Roboflow Visual Search block cannot run without a Roboflow API key. "
                "Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve one."
            )

        if isinstance(image, Batch):
            return [
                self._search_single_image(
                    image=single_image,
                    workspace=workspace,
                    target_project=target_project,
                    top_k=top_k,
                )
                for single_image in image
            ]

        return self._search_single_image(
            image=image,
            workspace=workspace,
            target_project=target_project,
            top_k=top_k,
        )

    def _search_single_image(
        self,
        image: WorkflowImageData,
        workspace: str,
        target_project: str,
        top_k: int,
    ) -> Dict[str, Any]:
        try:
            response = search_project_images_at_roboflow(
                api_key=self._api_key,
                workspace=workspace,
                project=target_project,
                image_base64=image.base64_image,
                limit=top_k,
            )
            matches = [_format_match(match) for match in response.get("results", [])]
            if not matches:
                return {
                    "matched": False,
                    "match": {},
                    "matches": [],
                    "error_status": False,
                    "message": "No visually similar images found.",
                }

            return {
                "matched": True,
                "match": matches[0],
                "matches": matches,
                "error_status": False,
                "message": "Visual search completed.",
            }
        except Exception as error:
            return {
                "matched": False,
                "match": {},
                "matches": [],
                "error_status": True,
                "message": f"Visual search failed: {error}",
            }


def _format_match(match: Dict[str, Any]) -> Dict[str, Any]:
    filename = match.get("filename") or match.get("name")
    return {
        "image_id": match.get("id"),
        "score": match.get("score"),
        "image_url": match.get("url"),
        "name": match.get("name") or filename,
        "filename": filename,
        "metadata": match.get("user_metadata") or {},
        "tags": match.get("tags") or [],
        "width": match.get("width"),
        "height": match.get("height"),
        "aspect_ratio": match.get("aspectRatio"),
    }
