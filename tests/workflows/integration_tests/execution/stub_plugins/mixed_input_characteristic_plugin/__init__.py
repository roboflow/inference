from typing import Any, Dict, List, Literal, Type, Union

from pydantic import ConfigDict

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    Selector,
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
        Selector(),
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
        Selector(),
        Any,
    ]

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return ["mixed_parameter"]

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
    batch_parameter: Selector()

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["batch_parameter"]

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
        if not isinstance(batch_parameter, Batch):
            raise ValueError("Batch[X] must be provided")
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
    batch_parameter: Selector()

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

    def run(self, batch_parameter: Any) -> BlockResult:
        return {"float_value": 0.4}


class CompoundNonBatchInputBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["CompoundNonBatchInputBlock"]
    compound_parameter: Dict[str, Union[Selector(), Any]]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class CompoundNonBatchInputBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CompoundNonBatchInputBlockManifest

    def run(self, compound_parameter: Dict[str, Any]) -> BlockResult:
        return {"float_value": 0.4}


class CompoundMixedInputBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["CompoundMixedInputBlockManifestBlock"]
    compound_parameter: Dict[str, Union[Selector(), Any]]

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return ["compound_parameter"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class CompoundMixedInputBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CompoundMixedInputBlockManifest

    def run(self, compound_parameter: Dict[str, Any]) -> BlockResult:
        retrieved_batches = [
            v for v in compound_parameter.values() if isinstance(v, Batch)
        ]
        if not retrieved_batches:
            return {"float_value": 0.4}
        return [{"float_value": 0.4}] * len(retrieved_batches[0])


class CompoundStrictBatchBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["CompoundStrictBatchBlock"]
    compound_parameter: Dict[str, Selector()]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["compound_parameter"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class CompoundStrictBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CompoundStrictBatchBlockManifest

    def run(self, compound_parameter: Dict[str, Any]) -> BlockResult:
        retrieved_batches = [
            v for v in compound_parameter.values() if isinstance(v, Batch)
        ]
        return [{"float_value": 0.4}] * len(retrieved_batches[0])


class CompoundNonStrictBatchBlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "dummy",
        }
    )
    type: Literal["CompoundNonStrictBatchBlock"]
    compound_parameter: Dict[str, Union[Selector()]]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="float_value",
                kind=[FLOAT_ZERO_TO_ONE_KIND],
            ),
        ]


class CompoundNonStrictBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return CompoundNonStrictBatchBlockManifest

    def run(self, compound_parameter: Dict[str, Any]) -> BlockResult:
        return {"float_value": 0.4}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        NonBatchInputBlock,
        MixedInputWithBatchesBlock,
        MixedInputWithoutBatchesBlock,
        BatchInputProcessingBatchesBlock,
        BatchInputNotProcessingBatchesBlock,
        CompoundNonBatchInputBlock,
        CompoundMixedInputBlock,
        CompoundStrictBatchBlock,
        CompoundNonStrictBatchBlock,
    ]
