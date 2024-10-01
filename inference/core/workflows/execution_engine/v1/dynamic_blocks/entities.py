from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SelectorType(Enum):
    INPUT_IMAGE = "input_image"
    STEP_OUTPUT_IMAGE = "step_output_image"
    INPUT_PARAMETER = "input_parameter"
    STEP_OUTPUT = "step_output"


class ValueType(Enum):
    ANY = "any"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DICT = "dict"
    LIST = "list"
    STRING = "string"


class DynamicInputDefinition(BaseModel):
    type: Literal["DynamicInputDefinition"]
    has_default_value: bool = Field(
        default=False,
        description="Flag to decide if default value is provided for input",
    )
    default_value: Any = Field(
        description="Definition of default value for a field. Use in combination with, "
        "`has_default_value` to decide on default value if field is optional.",
        default=None,
    )
    is_optional: bool = Field(
        description="Flag deciding if `default_value` will be added for manifest field annotation.",
        default=False,
    )
    is_dimensionality_reference: bool = Field(
        default=False,
        description="Flag deciding if declared property holds dimensionality reference - see how "
        "dimensionality works for statically defined blocks to discover meaning of the "
        "parameter.",
    )
    dimensionality_offset: int = Field(
        default=0,
        ge=-1,
        le=1,
        description="Accepted dimensionality offset for parameter. Dimensionality works the same as for "
        "traditional workflows blocks.",
    )
    selector_types: List[SelectorType] = Field(
        default_factory=list,
        description="Union of selector types accepted by input. Should be empty if field does not accept "
        "selectors.",
    )
    selector_data_kind: Dict[SelectorType, List[str]] = Field(
        default_factory=dict,
        description="Mapping of `selector_types` into names of kinds to be compatible. "
        "Empty dict (default value) means wildcard kind for all selectors. If name of kind given - "
        "must be valid kind, known for workflow execution engine.",
    )
    value_types: List[ValueType] = Field(
        default_factory=list,
        description="List of types representing union of types for static values (non selectors) "
        "that shall be accepted for input field. Empty list represents no value types allowed.",
    )


class DynamicOutputDefinition(BaseModel):
    type: Literal["DynamicOutputDefinition"]
    kind: List[str] = Field(
        default_factory=list,
        description="List representing union of kinds for defined output",
    )


class ManifestDescription(BaseModel):
    type: Literal["ManifestDescription"]
    block_type: str = Field(
        description="Field holds type of the bock to be dynamically created. Block can be initialised "
        "as step using the type declared in the field."
    )
    description: Optional[str] = Field(
        default=None, description="Description of the block to be used in manifest"
    )
    inputs: Dict[str, DynamicInputDefinition] = Field(
        description="Mapping name -> input definition for block inputs (parameters for run() function of"
        "dynamic block)"
    )
    outputs: Dict[str, DynamicOutputDefinition] = Field(
        default_factory=dict,
        description="Mapping name -> output kind for block outputs.",
    )
    output_dimensionality_offset: int = Field(
        default=0, ge=-1, le=1, description="Definition of output dimensionality offset"
    )
    accepts_batch_input: bool = Field(
        default=False,
        description="Flag to decide if function will be provided with batch data as whole or with singular "
        "batch elements while execution",
    )
    accepts_empty_values: bool = Field(
        default=False,
        description="Flag to decide if empty (optional) values will be shipped as run() function parameters",
    )


class PythonCode(BaseModel):
    type: Literal["PythonCode"]
    run_function_code: str = Field(
        description="Code of python function. Content should be properly formatted including indentations. "
        "Workflows execution engine is to create dynamic module with provided function - ensuring "
        "imports of the following symbols: [Any, List, Dict, Set, sv, np, math, time, json, os, "
        "requests, cv2, shapely, Batch, WorkflowImageData, BlockResult]. Expected signature is: "
        "def run(self, ... # parameters of manifest apart from name and type). Through self, "
        "one may access self._init_results which is dict returned by `init_code` if given."
    )
    run_function_name: str = Field(
        default="run", description="Name of the function shipped in `function_code`."
    )
    init_function_code: Optional[str] = Field(
        description="Code of the function to perform initialisation of the block. It must be "
        "parameter-free function with signature `def init() -> Dict[str, Any]` setting "
        "self._init_results on dynamic class initialisation",
        default=None,
    )
    init_function_name: str = Field(
        default="init",
        description="Name of init_code function.",
    )
    imports: List[str] = Field(
        default_factory=list,
        description="List of additional imports required to run the code",
    )


class DynamicBlockDefinition(BaseModel):
    type: Literal["DynamicBlockDefinition"]
    manifest: ManifestDescription = Field(
        description="Definition of manifest for dynamic block to be created in runtime by "
        "workflows execution engine."
    )
    code: PythonCode = Field(
        description="Code to be executed in run(...) method of block that will be dynamically "
        "created."
    )


BLOCK_SOURCE = "dynamic_workflows_blocks"
