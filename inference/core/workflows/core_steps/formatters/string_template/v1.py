from copy import copy
from string import Formatter
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Build a string from a template with named placeholders that are filled in at runtime from
workflow inputs and step outputs, enabling dynamic prompt construction for VLM blocks,
message formatting for notification and sink blocks, and any other place where text must be
assembled from workflow data.

## How This Block Works

This block renders a template string by substituting named placeholders with runtime values. The block:

1. Receives a template string containing placeholders in curly braces, e.g.
   `"This facing contains one of the following products: {sku_list}. Answer with the product name or NONE."`
2. Receives input data as a dictionary of named variables, where values may be literal values,
   workflow inputs (`$inputs.*`) or step outputs (`$steps.*`)
3. Optionally applies data transformations using operations (the same operation system as the
   Property Definition and Expression blocks) to each variable before substitution - for example
   joining a list of values into a single string with `SequenceJoin`, or extracting class names
   from detections with `DetectionsPropertyExtract`
4. Substitutes each `{variable_name}` placeholder with the (transformed) value of the matching
   variable, converting non-string values to strings; standard Python format specifications are
   supported (e.g. `{confidence:.2f}`)
5. Returns the rendered string as output of kind `string`, ready to connect into any block field
   accepting a string - for example the `prompt` field of VLM blocks

Literal curly braces can be produced by doubling them (`{{` renders as `{`, `}}` renders as `}`).

The block fails at runtime with a descriptive error when the template references a variable that
is not defined in data, when positional placeholders (`{}` or `{0}`) are used, or when a
placeholder uses attribute or index access (`{var.attr}`, `{var[0]}`) - use data operations to
extract nested values instead. Failing loudly on template/data mismatches prevents silently
malformed output (e.g. a prompt with a missing product list) from propagating downstream.

## Common Use Cases

- **Constrained VLM prompts**: Assemble prompts that embed runtime context (e.g. an expected
  product list from a planogram, a manifest, or a BOM) into an engineered prompt template, so
  users provide only the data while the tested prompt wording stays inside the workflow,
  enabling expectation-driven inspection workflows
- **Dynamic messages for sinks and notifications**: Format human-readable messages containing
  detection counts, class names or timestamps for email, Slack, Twilio or webhook sink blocks,
  enabling readable alerting workflows
- **Structured text records**: Compose single-line text records from multiple step outputs for
  logging or downstream processing, enabling text serialization workflows

## Connecting to Other Blocks

This block consumes workflow inputs and step outputs and produces a string:

- **Before VLM blocks** (e.g. Qwen-VL, Florence-2, OpenAI, Anthropic Claude, Google Gemini) to
  build the `prompt` from runtime data, enabling dynamic prompting workflows
- **After model and analytics blocks** to render their outputs into readable text, enabling
  results formatting workflows
- **Before sink blocks** (email, webhook, MQTT, OPC) to format message payloads, enabling
  notification workflows

## Requirements

The template must reference only variables defined in the data dictionary, using plain names
without attribute or index access. Data operations are optional and use the same operation
system as the Property Definition block; use them to transform values (e.g. join lists,
extract properties) before substitution.
"""

SHORT_DESCRIPTION = (
    "Build a string from a template and runtime data, e.g. a dynamic VLM prompt."
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "String Template",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-brackets-curly",
                "blockPriority": 2,
            },
        }
    )
    type: Literal["roboflow_core/string_template@v1"]
    template: Union[Selector(kind=[STRING_KIND]), str] = Field(
        description="Template string with named placeholders in curly braces, filled in from "
        "data at runtime. Use double braces for literal braces. Example: 'This facing contains "
        "one of: {sku_list}. Answer with the product name or NONE.'",
        examples=[
            "This facing contains one of: {sku_list}. Answer with the product name or NONE.",
            "$inputs.prompt_template",
        ],
    )
    data: Dict[str, Union[Selector(), str, int, float, bool]] = Field(
        description="Dictionary of named variables used to fill template placeholders. Keys are "
        "variable names referenced in the template, values are selectors referencing workflow "
        "inputs or step outputs, or literal values.",
        default_factory=dict,
        examples=[{"sku_list": "$inputs.expected_skus"}],
    )
    data_operations: Dict[str, List[AllOperationsType]] = Field(
        description="Optional dictionary of operations to transform data variables before "
        "substitution. Keys are variable names from data, values are lists of operations (same "
        "as Property Definition block). Useful for joining lists into text (SequenceJoin) or "
        "extracting properties from predictions before templating.",
        examples=[{"sku_list": [{"type": "SequenceJoin", "separator": ", "}]}],
        default_factory=lambda: {},
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class StringTemplateBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        template: str,
        data: Dict[str, Any],
        data_operations: Dict[str, List[AllOperationsType]],
    ) -> BlockResult:
        if not isinstance(template, str):
            raise ValueError(
                f"String Template block expected template to be a string, "
                f"got value of type {type(template).__name__}."
            )
        variables = copy(data)
        for variable_name, operations in data_operations.items():
            if variable_name not in variables:
                raise ValueError(
                    f"String Template block defines operations for variable "
                    f"`{variable_name}` which is not declared in data. "
                    f"Declared variables: {sorted(variables)}."
                )
            operations_chain = build_operations_chain(operations=operations)
            variables[variable_name] = operations_chain(
                variables[variable_name], global_parameters={}
            )
        rendered = render_template(template=template, variables=variables)
        return {"output": rendered}


def render_template(template: str, variables: Dict[str, Any]) -> str:
    placeholders = extract_placeholders(template=template)
    missing = placeholders.difference(variables)
    if missing:
        raise ValueError(
            f"String Template block template references variables {sorted(missing)} "
            f"which are not declared in data. Declared variables: {sorted(variables)}."
        )
    try:
        return template.format(**variables)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"String Template block could not render template: {error}. "
            f"Check that format specs match the types of the substituted values."
        ) from error


def extract_placeholders(template: str) -> set:
    placeholders = set()
    try:
        parsed_template = list(Formatter().parse(template))
    except ValueError as error:
        raise ValueError(
            f"String Template block could not parse template: {error}. "
            f"Use double braces to produce literal braces in the output."
        ) from error
    for _, field_name, format_spec, _ in parsed_template:
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise ValueError(
                "String Template block does not support positional placeholders - "
                "use named placeholders, e.g. `{variable_name}`."
            )
        if "." in field_name or "[" in field_name:
            raise ValueError(
                f"String Template block does not support attribute or index access in "
                f"placeholders (got `{{{field_name}}}`). Use data operations to extract "
                f"nested values before templating."
            )
        placeholders.add(field_name)
        # format specs may contain nested replacement fields (e.g. `{value:{width}}`)
        # which str.format resolves recursively - they need the same validation
        if format_spec and "{" in format_spec:
            placeholders.update(extract_placeholders(template=format_spec))
    return placeholders
