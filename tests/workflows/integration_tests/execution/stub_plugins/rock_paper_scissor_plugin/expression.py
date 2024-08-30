"""
This is just example, test implementation, please do not assume it being fully functional.
This is extremely unsafe block - be aware for injected code execution!
"""

from typing import Any, Dict, List, Literal, Type, Union

from pydantic import BaseModel, ConfigDict, Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class PythonCodeBlock(BaseModel):
    type: Literal["PythonBlock"]
    code: str


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "",
        }
    )
    type: Literal["ExpressionTestBlock"]
    data: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=["$inputs.confidence", "$inputs.image", "$steps.my_step.top"],
        default_factory=dict,
    )
    output: Union[str, PythonCodeBlock]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]


class ExpressionBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self, data: Dict[str, Any], output: Union[str, PythonCodeBlock]
    ) -> BlockResult:
        if isinstance(output, str):
            return {"output": output}
        results = {}
        params = ", ".join(f"{k}={k}" for k in data)
        code = output.code + f"\n\nresult = function({params})"
        exec(code, data, results)
        result = {"output": results["result"]}
        print("result", result)
        return result
