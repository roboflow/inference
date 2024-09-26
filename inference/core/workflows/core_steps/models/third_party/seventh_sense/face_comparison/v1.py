from typing import List, Literal, Optional, Type, Union
from pydantic import ConfigDict, Field

from opencv.fr import FR
from opencv.fr.compare.schemas import CompareRequest
from opencv.fr.search.schemas import SearchMode

from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    STRING_KIND,
    StepOutputImageSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The OpenCV Face Comparison API accepts two images and returns a similarity score between 0 and 1
based on the similarity of the faces in the images. A score above 0.7 signifies a high likelihood
that the photos are of the same person.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Face Comparison",
            "version": "v1",
            "short_description": "Determine if two faces are of the same person.",
            "long_description": LONG_DESCRIPTION,
            "license": "MIT",
            "block_type": "model",
            "search_keywords": ["facial", "identity", "seventh sense", "opencv"],
        }
    )
    type: Literal["roboflow_core/seventh_sense/face_comparison@1"]
    image_1: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image 1",
        description="The first image to compare against the second image",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    image_2: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        title="Image 2",
        description="The second image to compare against the first image",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    search_mode: Union[
        Literal[
            "FAST",
            "ACCURATE",
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        description="Search mode for the face comparison",
        default="FAST",
        examples=["FAST", "ACCURATE", "$inputs.search_mode"],
    )
    backend_url: Union[
        Literal[
            "https://sg.opencv.fr",
            "https://us.opencv.fr",
            "https://eu.opencv.fr",
        ],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        description="Region of your Seventh Sense account.",
        default="https://us.opencv.fr",
        examples=["https://sg.opencv.fr", "$inputs.seventh_sense_backend_url"],
    )

    api_key: Union[WorkflowParameterSelector(kind=[STRING_KIND]), str] = Field(
        description="Your Seventh Sense API key",
        examples=["xxxxxx", "$inputs.seventh_sense_api_key"],
        private=True,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="score",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
                description="Similarity score between 0 and 1",
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class FaceComparisonBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdkCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"

    def getSdk(self, backend_url, api_key):
        key = f"{backend_url}_{api_key}"

        if key not in self.sdkCache:
            self.sdkCache[key] = FR(backend_url, api_key)

        return self.sdkCache[key]

    def run(
        self,
        image_1: WorkflowImageData,
        image_2: WorkflowImageData,
        search_mode: str,
        backend_url: str,
        api_key: str,
    ) -> BlockResult:
        sdk = self.getSdk(backend_url, api_key)

        compare_request = CompareRequest(
            [image_1.numpy_image],
            [image_2.numpy_image],
            search_mode=(
                SearchMode.FAST if search_mode == "FAST" else SearchMode.ACCURATE
            ),
        )
        score = sdk.compare.compare_image_sets(compare_request)

        return {"score": score}
