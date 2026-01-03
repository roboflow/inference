import json
import logging
import re
from typing import List, Literal, Optional, Tuple, Type, Union

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
Parse JSON strings (raw JSON or JSON wrapped in Markdown code blocks) into structured data by extracting specified fields and exposing them as individual outputs, enabling JSON parsing, LLM/VLM output processing, structured data extraction, and configuration parsing workflows where JSON strings need to be converted into usable workflow data.

## How This Block Works

This block parses JSON strings and extracts specified fields as individual outputs. The block:

1. Receives a JSON string input (typically from LLM/VLM blocks or workflow inputs)
2. Detects and extracts JSON content:

   **Handles Markdown-wrapped JSON:**
   - Searches for JSON wrapped in Markdown code blocks (```json ... ```)
   - This format is very common in LLM/VLM responses (e.g., GPT responses)
   - If multiple markdown JSON blocks are found, only the first block is parsed
   - Extracts the JSON content from within the markdown tags

   **Handles raw JSON strings:**
   - If no markdown blocks are found, attempts to parse the entire string as JSON
   - Supports standard JSON format strings
3. Parses JSON content:
   - Uses Python's JSON parser to convert the string into a JSON object/dictionary
   - Handles parsing errors gracefully (returns None for all fields if parsing fails)
4. Extracts expected fields:
   - Retrieves values for each field specified in `expected_fields` parameter
   - For each expected field, looks up the corresponding key in the parsed JSON
   - Returns the field value (or None if the field is missing)
5. Sets error status:
   - `error_status` is set to `True` if at least one expected field cannot be retrieved from the parsed JSON
   - `error_status` is set to `False` if all expected fields are found (even if multiple markdown blocks exist, only first is parsed)
   - Error status is always included as an output, allowing downstream blocks to check parsing success
6. Exposes fields as outputs:
   - Each field in `expected_fields` becomes a separate output with the field name
   - Field values are extracted from the parsed JSON and made available as outputs
   - Missing fields are set to None
   - All outputs can be referenced using `$steps.block_name.field_name` syntax
7. Returns parsed data:
   - Outputs include: `error_status` (boolean) and all expected fields
   - Fields contain the extracted values from the JSON (or None if missing)
   - Outputs can be used in subsequent workflow steps

The block is particularly useful for processing LLM/VLM outputs that return JSON, extracting structured configuration from JSON strings, and parsing JSON responses into workflow-usable data. It handles the common case where LLMs wrap JSON in markdown code blocks.

## Common Use Cases

- **LLM/VLM Output Processing**: Parse JSON outputs from Large Language Models and Visual Language Models (e.g., parse GPT JSON responses, extract structured data from LLM outputs, process VLM JSON responses), enabling LLM/VLM output processing workflows
- **Structured Data Extraction**: Extract structured data from JSON strings for use in workflows (e.g., extract configuration parameters, parse JSON responses, extract structured fields), enabling structured data extraction workflows
- **Configuration Parsing**: Parse JSON configuration strings into workflow parameters (e.g., parse model configuration, extract workflow parameters, parse JSON configs), enabling configuration parsing workflows
- **JSON Response Processing**: Process JSON responses from APIs or models (e.g., parse API JSON responses, extract fields from JSON, process JSON data), enabling JSON response processing workflows
- **Dynamic Parameter Extraction**: Extract dynamic parameters from JSON strings for use in workflow steps (e.g., extract model IDs from JSON, parse dynamic configs, extract parameters dynamically), enabling dynamic parameter workflows
- **Data Format Conversion**: Convert JSON strings into structured workflow data (e.g., convert JSON to workflow inputs, parse JSON for workflow use, extract JSON fields), enabling data format conversion workflows

## Connecting to Other Blocks

This block receives JSON strings and produces parsed field outputs:

- **After LLM/VLM blocks** to parse JSON outputs into structured data (e.g., parse LLM JSON outputs, extract VLM JSON fields, process model JSON responses), enabling LLM/VLM-to-parser workflows
- **After workflow inputs** to parse JSON input parameters (e.g., parse JSON config inputs, extract JSON parameters, process JSON workflow inputs), enabling input-parser workflows
- **Before model blocks** to use parsed fields as model parameters (e.g., use parsed model_id for models, use parsed configs for model setup, provide parsed parameters to models), enabling parser-to-model workflows
- **Before logic blocks** to use parsed fields in conditions (e.g., use parsed values in Continue If, filter based on parsed fields, make decisions using parsed data), enabling parser-to-logic workflows
- **Before data storage blocks** to store parsed field values (e.g., store parsed JSON fields, log parsed values, save parsed data), enabling parser-to-storage workflows
- **In workflow outputs** to provide parsed fields as final output (e.g., JSON parsing outputs, structured data outputs, parsed field outputs), enabling parser-to-output workflows

## Requirements

This block requires a JSON string input (raw JSON or JSON wrapped in Markdown code blocks). The `expected_fields` parameter specifies which JSON fields to extract as outputs (field names must be valid JSON keys). The `error_status` field name is reserved and cannot be used in `expected_fields`. The block supports both raw JSON strings and JSON wrapped in markdown code blocks (```json ... ```). If multiple markdown blocks are found, only the first is parsed. If parsing fails or expected fields are missing, fields are set to None and `error_status` is set to True. All expected fields become separate outputs that can be referenced in subsequent workflow steps.
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
    raw_json: Union[str, Selector(kind=[LANGUAGE_MODEL_OUTPUT_KIND])] = Field(
        description="JSON string to parse. Can be raw JSON string (e.g., '{\"key\": \"value\"}') or JSON wrapped in Markdown code blocks (e.g., ```json {\"key\": \"value\"} ```). Markdown-wrapped JSON is common in LLM/VLM responses. If multiple markdown JSON blocks are present, only the first block is parsed. The string is parsed using Python's JSON parser, and specified fields are extracted as outputs.",
        examples=["$steps.lmm.output", "$inputs.config", '{"key": "value"}', '{"model_id": "yolov8n-640"}'],
    )
    expected_fields: Annotated[List[str], AfterValidator(validate_reserved_fields)] = (
        Field(
            description="List of JSON field names to extract from the parsed JSON. Each field becomes a separate output that can be referenced in subsequent workflow steps (e.g., $steps.block_name.field_name). Fields that exist in the JSON are extracted with their values; missing fields are set to None. The 'error_status' field name is reserved (always included as output) and cannot be used in this list. Field names must match JSON keys exactly.",
            examples=[["field_a", "field_b"], ["model_id", "confidence"], ["class_name", "class_id"]],
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
