import json
import logging
import re
from typing import List, Literal, Optional, Tuple, Type

from pydantic import AfterValidator, ConfigDict, Field
from typing_extensions import Annotated

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json([\s\S]*?)```", flags=re.IGNORECASE)

LONG_DESCRIPTION = """
The block expects string input that would be produced by blocks exposing Large Language Models (LLMs) and 
Visual Language Models (VLMs). Input is parsed to JSON, and its keys are exposed as block outputs.

Accepted formats:
- valid JSON strings
- JSON documents wrapped with Markdown tags (very common for GPT responses)
```
{"my": "json"}
```

**Details regarding block behavior:**

- `error_status` is set `True` whenever at least one of `expected_fields` cannot be retrieved from input

- in case of multiple markdown blocks with raw JSON content - only first will be parsed and returned, while
`error_status` will remain `False`
"""

SHORT_DESCRIPTION = "Parse raw string into JSON."


def validate_reserved_fields(expected_fields: List[str]) -> List[str]:
    if "error_status" in expected_fields:
        raise ValueError(
            "`error_status` is reserved field name and cannot be "
            "used in `expected_fields` of `roboflow_core/json_parser@v1` block."
        )
    return expected_fields


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "JSON Parser",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-table-tree",
                "blockPriority": 5,
            },
        }
    )
    type: Literal["roboflow_core/json_parser@v1"]
    raw_json: Selector(kind=[LANGUAGE_MODEL_OUTPUT_KIND]) = Field(
        description="The string with raw JSON to parse.",
        examples=[["$steps.lmm.output"]],
    )
    expected_fields: Annotated[List[str], AfterValidator(validate_reserved_fields)] = (
        Field(
            description="List of expected JSON fields. `error_status` field name is reserved and cannot be used.",
            examples=[["field_a", "field_b"]],
        )
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="*"),
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        result = [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
        ]
        for field_name in self.expected_fields:
            result.append(OutputDefinition(name=field_name))
        return result

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class JSONParserBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        raw_json: str,
        expected_fields: List[str],
    ) -> BlockResult:
        error_status, parsed_data = string2json(
            raw_json=raw_json,
            expected_fields=expected_fields,
        )
        parsed_data["error_status"] = error_status
        return parsed_data


def string2json(
    raw_json: str,
    expected_fields: List[str],
) -> Tuple[bool, dict]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(raw_json)
    if len(json_blocks_found) == 0:
        return try_parse_json(raw_json, expected_fields=expected_fields)
    first_block = json_blocks_found[0]
    return try_parse_json(first_block, expected_fields=expected_fields)


def try_parse_json(content: str, expected_fields: List[str]) -> Tuple[bool, dict]:
    try:
        parsed_data = json.loads(content)
        result = {}
        all_fields_find = True
        for field in expected_fields:
            if field not in parsed_data:
                all_fields_find = False
            result[field] = parsed_data.get(field)
        return not all_fields_find, result
    except Exception as error:
        logging.warning(
            f"Could not parse JSON in `roboflow_core/json_parser@v1` block. "
            f"Error type: {error.__class__.__name__}. Details: {error}"
        )
        return True, {field: None for field in expected_fields}
