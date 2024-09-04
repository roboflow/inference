from typing import Any, List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import StepOutputSelector, BATCH_OF_STRING_KIND, \
    BATCH_OF_BOOLEAN_KIND
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
The block expects string input that would be produced by blocks exposing Large Language Models (LLMs) and 
Visual Language Models (VLMs) into JSON.

Accepted formats:
- valid JSON strings
- JSON documents wrapped with Markdown tags (very common for GPT responses)
```
```json
{"my": "json"}
```
```
"""

SHORT_DESCRIPTION = "Parses raw string into JSON."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "JSON Parser",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
        }
    )
    type: Literal[
        "roboflow_core/json_parser@v1",
    ]
    raw_json: StepOutputSelector(kind=[BATCH_OF_STRING_KIND])
    expected_fields: List[str] = Field(
        description="List of expected JSON fields",
        examples=[["field_a", "field_b"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BATCH_OF_BOOLEAN_KIND]),
            OutputDefinition(name="*")
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=field_name)
            for field_name in self.expected_fields
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class FirstNonEmptyOrDefaultBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        raw_json: str,
        expected_fields: List[str],
    ) -> BlockResult:
        pass
