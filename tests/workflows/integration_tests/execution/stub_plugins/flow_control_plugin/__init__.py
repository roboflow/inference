"""
This is just example, test implementation, please do not assume it being fully functional.
"""

import random
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    StatementGroup,
)
from inference.core.workflows.core_steps.common.query_language.evaluation_engine.core import (
    build_eval_function,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    STRING_KIND,
    Selector,
    StepOutputSelector,
    StepSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ABTestManifest(WorkflowBlockManifest):
    type: Literal["ABTest"]
    name: str = Field(description="name field")
    a_step: StepSelector
    b_step: StepSelector

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ABTestBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ABTestManifest

    def run(
        self,
        a_step: StepSelector,
        b_step: StepSelector,
    ) -> BlockResult:
        choice = a_step
        if random.random() > 0.5:
            choice = b_step
        return FlowControl(mode="select_step", context=choice)


LONG_DESCRIPTION = """
Based on provided configuration, block decides which execution path to take given
data fed into condition logic.
"""

SHORT_DESCRIPTION = "Creates alternative execution branches for data"


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "flow_control",
        }
    )
    type: Literal["Condition"]
    condition_statement: StatementGroup
    evaluation_parameters: Dict[
        str,
        Union[WorkflowImageSelector, WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parametrise operations",
        examples=["$inputs.confidence", "$inputs.image", "$steps.my_step.top"],
        default_factory=lambda: {},
    )
    steps_if_true: List[StepSelector] = Field(
        description="Reference to step which shall be executed if expression evaluates to true",
        examples=[["$steps.on_true"]],
    )
    steps_if_false: List[StepSelector] = Field(
        description="Reference to step which shall be executed if expression evaluates to false",
        examples=[["$steps.on_false"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class ConditionBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        condition_statement: StatementGroup,
        evaluation_parameters: Dict[str, Any],
        steps_if_true: List[StepSelector],
        steps_if_false: List[StepSelector],
    ) -> BlockResult:
        if not steps_if_true and not steps_if_false:
            return FlowControl(mode="terminate_branch")
        evaluation_function = build_eval_function(definition=condition_statement)
        evaluation_result = evaluation_function(evaluation_parameters)
        next_steps = steps_if_true if evaluation_result else steps_if_false
        if not next_steps:
            return FlowControl(mode="terminate_branch")
        flow_control = FlowControl(mode="select_step", context=next_steps)
        return flow_control


class AlwaysMoveManifest(WorkflowBlockManifest):
    type: Literal["AlwaysMove"]
    name: str = Field(description="name field")
    a_step: StepSelector
    evaluation_parameters: Dict[
        str,
        Selector(),
    ] = Field(
        description="Dictionary mapping operand names (used in condition_statement) to actual values from the workflow. These parameters provide the dynamic data that gets evaluated in the conditional statement. Keys match operand names in the condition (e.g., 'left', 'right'), and values are selectors referencing workflow inputs, step outputs, or computed values. Example: {'left': '$steps.detection.count', 'threshold': 5} where 'left' is referenced in the condition_statement.",
        examples=[{"left": "$inputs.some"}],
        default_factory=lambda: {},
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class AlwaysMoveBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return AlwaysMoveManifest

    def run(
        self,
        a_step: StepSelector,
        evaluation_parameters: dict,
    ) -> BlockResult:
        return FlowControl(context=a_step)


class AlwaysStopManifest(WorkflowBlockManifest):
    type: Literal["AlwaysStop"]
    name: str = Field(description="name field")
    a_step: StepSelector
    evaluation_parameters: Dict[
        str,
        Selector(),
    ] = Field(
        description="Dictionary mapping operand names (used in condition_statement) to actual values from the workflow. These parameters provide the dynamic data that gets evaluated in the conditional statement. Keys match operand names in the condition (e.g., 'left', 'right'), and values are selectors referencing workflow inputs, step outputs, or computed values. Example: {'left': '$steps.detection.count', 'threshold': 5} where 'left' is referenced in the condition_statement.",
        examples=[{"left": "$inputs.some"}],
        default_factory=lambda: {},
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class AlwaysStopBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return AlwaysStopManifest

    def run(
        self,
        a_step: StepSelector,
        evaluation_parameters: dict,
    ) -> BlockResult:
        return FlowControl(context=None)


class MoveEvenManifest(WorkflowBlockManifest):
    type: Literal["MoveEven"]
    name: str = Field(description="name field")
    a_step: StepSelector
    evaluation_parameters: Dict[
        str,
        Selector(),
    ] = Field(
        description="Dictionary mapping operand names (used in condition_statement) to actual values from the workflow. These parameters provide the dynamic data that gets evaluated in the conditional statement. Keys match operand names in the condition (e.g., 'left', 'right'), and values are selectors referencing workflow inputs, step outputs, or computed values. Example: {'left': '$steps.detection.count', 'threshold': 5} where 'left' is referenced in the condition_statement.",
        examples=[{"left": "$inputs.some"}],
        default_factory=lambda: {},
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class MoveEvenBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MoveEvenManifest

    def __init__(self):
        self._runs = 0

    def run(
        self,
        a_step: StepSelector,
        evaluation_parameters: dict,
    ) -> BlockResult:
        prev_runs = self._runs
        self._runs += 1
        if prev_runs % 2 == 0:
            return FlowControl(context=a_step)
        else:
            return FlowControl(context=None)


class DimensionalityIncreaseManifest(WorkflowBlockManifest):
    type: Literal["DimensionalityIncrease"]
    name: str = Field(description="name field")
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image or video frame to process for background subtraction. The block processes frames sequentially to build a background model - each frame updates the background model and creates a motion mask showing areas that differ from the learned background. Can be connected from workflow inputs or previous steps.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output", kind=[STRING_KIND])]

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1


class DimensionalityIncreaseBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DimensionalityIncreaseManifest

    def run(self, image) -> BlockResult:
        return [
            {"output": "a"},
            {"output": "b"},
            {"output": "c"},
            {"output": "d"},
        ]


class ImageAndTextsStitchManifest(WorkflowBlockManifest):
    type: Literal["ImageAndTextsStitch"]
    name: str = Field(description="name field")
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The input image or video frame to process for background subtraction. The block processes frames sequentially to build a background model - each frame updates the background model and creates a motion mask showing areas that differ from the learned background. Can be connected from workflow inputs or previous steps.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    texts: Selector(kind=[STRING_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output", kind=[STRING_KIND])]

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "image"

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {"texts": 1}


class ImageAndTextsStitchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ImageAndTextsStitchManifest

    def run(self, image, texts) -> BlockResult:
        result = []
        for text in texts:
            print("text", text)
            result.append(text)
        return {"output": ", ".join(result)}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        ABTestBlock,
        ConditionBlock,
        AlwaysMoveBlock,
        AlwaysStopBlock,
        MoveEvenBlock,
        DimensionalityIncreaseBlock,
        ImageAndTextsStitchBlock,
    ]
