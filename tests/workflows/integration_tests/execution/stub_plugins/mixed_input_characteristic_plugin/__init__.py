from typing import Any, List, Literal, Type, Union

from pydantic import ConfigDict

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class NonBatchInputBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["NonBatchInputBlock"]
    non_batch_parameter: Union[WorkflowParameterSelector(), Any]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class NonBatchInputBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonBatchInputBlockManifest

    def run(self, non_batch_parameter: Any) -> BlockResult:
        return {"float_value": 0.4}


class MixedInputWithoutBatchesBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["MixedInputWithoutBatchesBlock"]
    mixed_parameter: Union[
        WorkflowParameterSelector(),
        StepOutputSelector(),
        Any,
    ]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class MixedInputWithoutBatchesBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MixedInputWithoutBatchesBlockManifest

    def run(self, mixed_parameter: Any) -> BlockResult:
        return {"float_value": 0.4}


class MixedInputWithBatchesBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["MixedInputWithBatchesBlock"]
    mixed_parameter: Union[
        WorkflowParameterSelector(),
        StepOutputSelector(),
        Any,
    ]

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class MixedInputWithBatchesBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MixedInputWithBatchesBlockManifest

    def run(self, mixed_parameter: Union[Batch[Any], Any]) -> BlockResult:
        if isinstance(mixed_parameter, Batch):
            return [{"float_value": 0.4}] * len(mixed_parameter)
        return {"float_value": 0.4}


class BatchInputBlockProcessingBatchesManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["BatchInputBlockProcessingBatches"]
    batch_parameter: StepOutputSelector()

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class BatchInputProcessingBatchesBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BatchInputBlockProcessingBatchesManifest

    def run(self, batch_parameter: Batch[Any]) -> BlockResult:
        return [{"float_value": 0.4}] * len(batch_parameter)


class BatchInputBlockProcessingNotBatchesManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["BatchInputBlockNotProcessingBatches"]
    batch_parameter: StepOutputSelector()

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class BatchInputNotProcessingBatchesBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BatchInputBlockProcessingNotBatchesManifest

    def run(self, batch_parameter: Batch[Any]) -> BlockResult:
        return {"float_value": 0.4}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        NonBatchInputBlock,
        MixedInputWithBatchesBlock,
        MixedInputWithoutBatchesBlock,
        BatchInputProcessingBatchesBlock,
        BatchInputNotProcessingBatchesBlock,
    ]
