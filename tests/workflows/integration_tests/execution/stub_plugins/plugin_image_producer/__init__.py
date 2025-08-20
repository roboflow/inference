import json
from typing import Literal, List, Optional, Type, Any
from uuid import uuid4

import numpy as np

from inference.core.workflows.execution_engine.entities.base import OutputDefinition, WorkflowImageData, \
    ImageParentMetadata, Batch
from inference.core.workflows.execution_engine.entities.types import IMAGE_KIND, Selector, STRING_KIND
from inference.core.workflows.prototypes.block import WorkflowBlockManifest, WorkflowBlock, BlockResult


class ImageProducerBlockManifest(WorkflowBlockManifest):
    type: Literal["ImageProducer"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="image", kind=[IMAGE_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class ImageProducerBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return ImageProducerBlockManifest

    def run(self) -> BlockResult:
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id=f"image_producer.{uuid4()}"),
            numpy_image=np.zeros((192, 168, 3), dtype=np.uint8)
        )
        return {"image": image}


class SingleImageConsumerManifest(WorkflowBlockManifest):
    type: Literal["ImageConsumer"]
    images: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="shapes", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images"]


class SingleImageConsumer(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SingleImageConsumerManifest

    def run(self, images: Batch[WorkflowImageData]) -> BlockResult:
        results = []
        for image in images:
            results.append({"shapes": json.dumps(image.numpy_image.shape)})
        return results


class MultiSIMDImageConsumerManifest(WorkflowBlockManifest):
    type: Literal["MultiSIMDImageConsumer"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="metadata", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images_x", "images_y"]


class MultiSIMDImageConsumer(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MultiSIMDImageConsumerManifest

    def run(self, images_x: Batch[WorkflowImageData], images_y: Batch[WorkflowImageData]) -> BlockResult:
        results = []
        for image_x, image_y in zip(images_x, images_y):
            results.append({"shapes": json.dumps(image_x.numpy_image.shape) + json.dumps(image_y.numpy_image.shape)})
        return results


class MultiImageConsumerManifest(WorkflowBlockManifest):
    type: Literal["MultiImageConsumer"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="metadata", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MultiImageConsumer(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MultiImageConsumerManifest

    def run(self, images_x: WorkflowImageData, images_y: WorkflowImageData) -> BlockResult:
        return {"shapes": json.dumps(images_x.numpy_image.shape) + json.dumps(images_y.numpy_image.shape)}


class MultiImageConsumerRaisingDimManifest(WorkflowBlockManifest):
    type: Literal["MultiImageConsumerRaisingDim"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="metadata", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1


class MultiImageConsumerRaisingDim(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MultiImageConsumerRaisingDimManifest

    def run(self, images_x: WorkflowImageData, images_y: WorkflowImageData) -> BlockResult:
        return [{"shapes": json.dumps(images_x.numpy_image.shape) + json.dumps(images_y.numpy_image.shape)}]


class MultiSIMDImageConsumerRaisingDimManifest(WorkflowBlockManifest):
    type: Literal["MultiSIMDImageConsumerRaisingDim"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="metadata", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1


class MultiSIMDImageConsumerRaisingDim(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MultiSIMDImageConsumerRaisingDimManifest

    def run(self, images_x: Batch[WorkflowImageData], images_y: Batch[WorkflowImageData]) -> BlockResult:
        results = []
        for image_x, image_y in zip(images_x, images_y):
            results.append([{"shapes": json.dumps(image_x.numpy_image.shape) + json.dumps(image_y.numpy_image.shape)}])
        return results


class IdentityManifest(WorkflowBlockManifest):
    type: Literal["Identity"]
    x: Selector()

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class IdentityBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return IdentityManifest

    def run(self, x: Any) -> BlockResult:
        return {"x": x}


class IdentitySIMDManifest(WorkflowBlockManifest):
    type: Literal["IdentitySIMD"]
    x: Selector()

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class IdentitySIMDBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return IdentitySIMDManifest

    def run(self, x: Batch[Any]) -> BlockResult:
        assert isinstance(x, Batch)
        return [{"x": x_el} for x_el in x]


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        ImageProducerBlock,
        SingleImageConsumer,
        MultiSIMDImageConsumer,
        MultiImageConsumer,
        MultiImageConsumerRaisingDim,
        MultiSIMDImageConsumerRaisingDim,
        IdentityBlock,
        IdentitySIMDBlock
    ]
