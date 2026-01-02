from typing import Any, List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import Selector
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Merge alternative execution branches by selecting the first non-empty value from multiple data inputs, or returning a default value if all inputs are empty, enabling conditional execution merging, empty value handling, and structured output construction workflows where data from different branches needs to be combined or fallback values need to be provided for missing data.

## How This Block Works

This block merges data from multiple sources (typically from different conditional execution branches) by selecting the first available non-empty value, ensuring outputs are always present for downstream processing. The block:

1. Receives a list of data references and an optional default value:
   - Takes multiple data inputs as a list of selectors (minimum 1 required)
   - Each selector can reference outputs from different workflow steps or branches
   - Receives a default value to use when all inputs are empty
2. Processes inputs in order:
   - Iterates through the data inputs in the order they are provided
   - Checks each input value to determine if it is non-empty (not None)
   - Stops at the first non-empty value encountered
3. Selects first non-empty value:
   - Returns the first non-empty value from the list if found
   - This allows prioritizing certain data sources over others
   - Order matters: earlier inputs have priority over later ones
4. Falls back to default if all empty:
   - If all inputs in the list are empty (None), returns the configured default value
   - Ensures the output is never None, making it safe for downstream blocks
   - Default value can be any type (string, number, object, null, etc.)
5. Handles empty values:
   - This block accepts empty values (None) from conditional execution
   - Unlike most blocks that skip processing when inputs are None, this block processes them
   - Converts potentially empty inputs into a guaranteed non-empty output
6. Returns merged output:
   - Outputs the selected value (first non-empty or default)
   - Output is always non-None, ensuring compatibility with blocks that don't accept empty values
   - Enables structured output construction even when some execution branches produce no data

This block is essential for merging alternative execution branches in workflows with conditional logic. When different branches of a workflow can produce data (e.g., one branch processes data if condition A is true, another if condition B is true), this block allows you to combine those branches by selecting the first available result, ensuring your workflow always produces a valid output.

## Common Use Cases

- **Merging Conditional Branches**: Merge outputs from alternative conditional execution branches into a single value (e.g., merge results from different if-else branches, combine alternative processing paths, unify conditional branch outputs), enabling conditional execution merging workflows
- **Empty Value Handling**: Handle potentially empty values from filtering or conditional execution by providing fallback defaults (e.g., handle filtered data with defaults, provide fallbacks for conditional branches, ensure non-empty outputs from optional steps), enabling robust empty value handling workflows
- **Structured Output Construction**: Ensure workflow outputs always have values even when some execution paths don't produce data (e.g., construct consistent output structures, guarantee output field presence, build structured responses with defaults), enabling structured output construction workflows
- **Priority-Based Selection**: Select data from multiple sources based on priority order (e.g., prefer primary source over fallback, select best available data source, prioritize certain processing results), enabling priority-based data selection workflows
- **Fallback Values**: Provide fallback values when primary data sources are unavailable (e.g., use default values when data missing, provide fallbacks for empty results, ensure downstream compatibility), enabling fallback value workflows
- **Output Normalization**: Normalize outputs to ensure they're always present and non-empty (e.g., normalize optional outputs, ensure consistent output format, guarantee output availability), enabling output normalization workflows

## Connecting to Other Blocks

This block receives multiple data inputs and produces a single merged output:

- **After conditional execution blocks** (ContinueIf, DetectionsFilter, etc.) to merge alternative branch outputs (e.g., merge if-else branch results, combine conditional paths, unify branch outputs), enabling conditional-to-merge workflows
- **Before blocks that don't accept empty values** to ensure inputs are always present (e.g., ensure non-empty inputs, provide fallbacks for empty data, guarantee input availability), enabling merge-to-processing workflows
- **In workflow outputs** to construct structured outputs with guaranteed field presence (e.g., build consistent outputs, ensure output completeness, create structured responses), enabling merge-to-output workflows
- **After filtering blocks** to handle cases where filters remove all data (e.g., provide defaults for filtered data, handle empty filter results, ensure output availability), enabling filter-to-merge workflows
- **Before data storage blocks** to ensure stored data is always present (e.g., store with defaults, ensure data completeness, provide fallback storage values), enabling merge-to-storage workflows
- **Between alternative processing paths** to combine results from different processing strategies (e.g., merge alternative processing results, combine different model outputs, unify processing strategies), enabling alternative-to-merge workflows

## Requirements

This block requires at least one data input reference (can accept multiple). The block accepts empty values (None), allowing it to process data from conditional execution branches. The `default` parameter is optional (defaults to None) and specifies the fallback value when all inputs are empty. Data inputs are processed in order, with the first non-empty value being selected. If all inputs are empty, the default value is returned. The output is always non-None, ensuring compatibility with blocks that don't accept empty values. This block is essential for merging alternative execution branches and ensuring structured outputs are always complete.
"""

SHORT_DESCRIPTION = (
    "Take the first non-empty data element or the configured default value."
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "First Non Empty Or Default",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-empty-set",
                "blockPriority": 7,
                "inDevelopment": True,
            },
        }
    )
    type: Literal[
        "roboflow_core/first_non_empty_or_default@v1", "FirstNonEmptyOrDefault"
    ]
    data: List[Selector()] = Field(
        description="List of data references (selectors) to check for non-empty values, in priority order. Each selector can reference outputs from different workflow steps or execution branches. The block iterates through this list and returns the first non-empty (non-None) value encountered. If all values in the list are empty/None, the default value is returned. Minimum 1 item required. Order matters: earlier items in the list have higher priority. Common use cases: merging outputs from conditional execution branches, providing fallback data sources, or combining results from alternative processing paths.",
        examples=[["$steps.my_step.predictions"], ["$steps.branch_a.output", "$steps.branch_b.output"], ["$steps.primary.result", "$steps.fallback.result", "$steps.alternative.result"]],
        min_items=1,
    )
    default: Any = Field(
        description="Default value to return when all data inputs are empty (None). This ensures the output is always non-None, making it safe for downstream blocks that don't accept empty values. The default can be any type: string (e.g., 'empty', 'N/A'), number (e.g., 0, -1), object (e.g., {}, []), null, or any other value. If not specified, defaults to None. Use this to provide fallback values when conditional execution branches or filtering removes all data, or to ensure structured outputs always have values.",
        examples=["empty", "N/A", 0, None, [], {}],
        default=None,
    )

    @classmethod
    def accepts_empty_values(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class FirstNonEmptyOrDefaultBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        data: Batch[Any],
        default: Any,
    ) -> BlockResult:
        result = default
        for data_element in data:
            if data_element is not None:
                return {"output": data_element}
        return {"output": result}
