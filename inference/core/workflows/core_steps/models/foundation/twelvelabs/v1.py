from typing import List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Analyze a whole video with [TwelveLabs](https://twelvelabs.io) Pegasus, a video-language
model that understands what happens across an entire clip - not just a single frame.

Unlike image-based VLM blocks, this block reasons over the temporal content of a video. Pass a
publicly accessible video URL together with a natural-language prompt and the model returns a free-form
text answer. Typical uses include summarising footage, answering questions about events in a video,
generating chapter descriptions, or extracting structured information from recordings.

You need a TwelveLabs API key to use this block. You can grab a free API key at
[twelvelabs.io](https://twelvelabs.io) - there is a generous free tier.

**WARNING!**

The video referenced by `video_url` is fetched and processed by TwelveLabs' servers, so the URL must be
publicly reachable. The model has resolution and duration requirements (resolution between 360p and 2160p);
very short or low-resolution clips may be rejected by the API.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "TwelveLabs Pegasus",
            "version": "v1",
            "short_description": "Understand and answer questions about a whole video with TwelveLabs Pegasus.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "LMM",
                "VLM",
                "video",
                "TwelveLabs",
                "Pegasus",
                "video understanding",
            ],
            "beta": True,
            "is_vlm_block": True,
            "ui_manifest": {
                "section": "model",
                "icon": "fa-solid fa-film",
                "blockPriority": 6,
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/twelvelabs_pegasus@v1"]
    video_url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Publicly accessible URL of the video to analyze.",
        examples=["https://example.com/video.mp4", "$inputs.video_url"],
    )
    prompt: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Natural-language prompt describing what TwelveLabs Pegasus should do with the video.",
        examples=["Summarize what happens in this video.", "$inputs.prompt"],
        json_schema_extra={"multiline": True},
    )
    api_key: Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str] = Field(
        description="Your TwelveLabs API key.",
        examples=["xxx-xxx", "$inputs.twelvelabs_api_key"],
        private=True,
    )
    model_version: Union[
        Selector(kind=[STRING_KIND]),
        Literal["pegasus1.5", "pegasus1.2"],
    ] = Field(
        default="pegasus1.5",
        description="TwelveLabs Pegasus model to be used.",
        examples=["pegasus1.5", "$inputs.twelvelabs_model"],
    )
    max_tokens: int = Field(
        default=2048,
        ge=512,
        description="Maximum number of tokens the model can generate in its response. "
        "TwelveLabs requires this to be at least 512 for Pegasus.",
    )
    temperature: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        description="Temperature to sample from the model - value in range 0.0-1.0, the higher - the more "
        'random / "creative" the generations are.',
        ge=0.0,
        le=1.0,
    )

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class TwelveLabsPegasusBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def run(
        self,
        video_url: str,
        prompt: str,
        api_key: str,
        model_version: str,
        max_tokens: int,
        temperature: Optional[float],
    ) -> BlockResult:
        output = analyze_video_with_pegasus(
            video_url=video_url,
            prompt=prompt,
            api_key=api_key,
            model_version=model_version,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {"output": output}


def analyze_video_with_pegasus(
    video_url: str,
    prompt: str,
    api_key: str,
    model_version: str,
    max_tokens: int,
    temperature: Optional[float],
) -> str:
    # `twelvelabs` is an optional dependency - import lazily so this module can always be loaded.
    try:
        from twelvelabs import TwelveLabs
        from twelvelabs.types.video_context import VideoContext_Url
    except ImportError as error:
        raise ImportError(
            "The `twelvelabs` package is required to use the TwelveLabs Pegasus block. "
            "Install it with `pip install twelvelabs`."
        ) from error
    client = TwelveLabs(api_key=api_key)
    response = client.analyze(
        model_name=model_version,
        video=VideoContext_Url(url=video_url),
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.data or ""
