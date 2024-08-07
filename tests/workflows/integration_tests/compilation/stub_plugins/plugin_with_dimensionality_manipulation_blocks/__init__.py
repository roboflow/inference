from typing import Dict, List, Literal, Optional, Type, Union

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    StepOutputImageSelector,
    WorkflowImageSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockRequestingDifferentDimsManifest(WorkflowBlockManifest):
    type: Literal["BlockRequestingDifferentDims"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "images": 0,
            "crops": 1,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "images"


class BlockRequestingDifferentDimsBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockRequestingDifferentDimsManifest

    def run(
        self, images: Batch[WorkflowImageData], crops: Batch[Batch[WorkflowImageData]]
    ) -> BlockResult:
        pass


class BlockOffsetsNotInProperRangeManifest(WorkflowBlockManifest):
    type: Literal["BlockOffsetsNotInProperRange"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "images": 0,
            "crops": 2,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "images"


class BlockOffsetsNotInProperRangeBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockOffsetsNotInProperRangeManifest

    def run(
        self, images: Batch[WorkflowImageData], crops: Batch[Batch[WorkflowImageData]]
    ) -> BlockResult:
        pass


class BlockWithNegativeOffsetManifest(WorkflowBlockManifest):
    type: Literal["BlockWithNegativeOffset"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "images": -1,
            "crops": 0,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "images"


class BlockWithNegativeOffsetBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockWithNegativeOffsetManifest

    def run(
        self, images: Batch[WorkflowImageData], crops: Batch[Batch[WorkflowImageData]]
    ) -> BlockResult:
        pass


class NonSIMDWithOutputOffsetManifest(WorkflowBlockManifest):
    type: Literal["NonSIMDWithOutputOffset"]
    name: str = Field(description="name field")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return 1


class NonSIMDWithOutputOffsetBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonSIMDWithOutputOffsetManifest

    def run(
        self,
    ) -> BlockResult:
        pass


class DimensionalityReferencePropertyIsNotBatchManifest(WorkflowBlockManifest):
    type: Literal["DimensionalityReferencePropertyIsNotBatch"]
    name: str = Field(description="name field")
    dim_reference: str = Field(default="a")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "dim_reference"


class DimensionalityReferencePropertyIsNotBatchBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DimensionalityReferencePropertyIsNotBatchManifest

    def run(
        self,
        dim_reference: str,
    ) -> BlockResult:
        pass


class OutputDimensionalityInInvalidRangeManifest(WorkflowBlockManifest):
    type: Literal["OutputDimensionalityInInvalidRange"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return 2


class OutputDimensionalityInInvalidRangeBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OutputDimensionalityInInvalidRangeManifest

    def run(
        self, images: Batch[WorkflowImageData], crops: Batch[WorkflowImageData]
    ) -> BlockResult:
        pass


class LackOfZeroGroundOffsetManifest(WorkflowBlockManifest):
    type: Literal["LackOfZeroGroundOffset"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "images": 1,
            "crops": 2,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "images"


class LackOfZeroGroundOffsetBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LackOfZeroGroundOffsetManifest

    def run(
        self, images: Batch[WorkflowImageData], crops: Batch[Batch[WorkflowImageData]]
    ) -> BlockResult:
        pass


class LackOfRequiredReferencePropertyManifest(WorkflowBlockManifest):
    type: Literal["LackOfRequiredReferenceProperty"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "images": 0,
            "crops": 1,
        }


class LackOfRequiredReferencePropertyBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LackOfRequiredReferencePropertyManifest

    def run(
        self, images: Batch[WorkflowImageData], crops: Batch[Batch[WorkflowImageData]]
    ) -> BlockResult:
        pass


class ManipulationOutputDimensionalityWhenInvalidManifest(WorkflowBlockManifest):
    type: Literal["ManipulationOutputDimensionalityWhenInvalid"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {
            "images": 0,
            "crops": 1,
        }

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "images"

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return 1


class ManipulationOutputDimensionalityWhenInvalidBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ManipulationOutputDimensionalityWhenInvalidManifest

    def run(
        self, images: Batch[WorkflowImageData], crops: Batch[Batch[WorkflowImageData]]
    ) -> BlockResult:
        pass


class ExpectsTheSameDimensionalityManifest(WorkflowBlockManifest):
    type: Literal["ExpectsTheSameDimensionality"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]


class ExpectsTheSameDimensionalityBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ExpectsTheSameDimensionalityManifest

    def run(
        self, images: Batch[WorkflowImageData], crops: Batch[WorkflowImageData]
    ) -> BlockResult:
        pass


class DecreasingDimensionalityManifest(WorkflowBlockManifest):
    type: Literal["DecreasingDimensionality"]
    name: str = Field(description="name field")
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Reference an image to be used as input for step processing",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return -1


class DecreasingDimensionalityBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DecreasingDimensionalityManifest

    def run(self, images: Batch[WorkflowImageData]) -> BlockResult:
        pass


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        BlockRequestingDifferentDimsBlock,
        BlockOffsetsNotInProperRangeBlock,
        BlockWithNegativeOffsetBlock,
        NonSIMDWithOutputOffsetBlock,
        DimensionalityReferencePropertyIsNotBatchBlock,
        OutputDimensionalityInInvalidRangeBlock,
        LackOfZeroGroundOffsetBlock,
        LackOfRequiredReferencePropertyBlock,
        ManipulationOutputDimensionalityWhenInvalidBlock,
        ExpectsTheSameDimensionalityBlock,
        DecreasingDimensionalityBlock,
    ]
