import json
from typing import Any, List, Literal, Optional, Tuple, Type, Union, Dict
from uuid import uuid4

import numpy as np
from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    IMAGE_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class ImageProducerBlockManifest(WorkflowBlockManifest):
    type: Literal["ImageProducer"]
    shape: Tuple[int, int, int] = Field(default=(192, 168, 3))

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

    def run(self, shape: Tuple[int, int, int]) -> BlockResult:
        image = WorkflowImageData(
            parent_metadata=ImageParentMetadata(parent_id=f"image_producer.{uuid4()}"),
            numpy_image=np.zeros(shape, dtype=np.uint8),
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


class SingleImageConsumerNonSIMDManifest(WorkflowBlockManifest):
    type: Literal["ImageConsumerNonSIMD"]
    images: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="shapes", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class SingleImageConsumerNonSIMD(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SingleImageConsumerNonSIMDManifest

    def run(self, images: WorkflowImageData) -> BlockResult:
        return {"shapes": json.dumps(images.numpy_image.shape)}


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

    def run(
        self, images_x: Batch[WorkflowImageData], images_y: Batch[WorkflowImageData]
    ) -> BlockResult:
        results = []
        for image_x, image_y in zip(images_x, images_y):
            results.append(
                {
                    "metadata": json.dumps(image_x.numpy_image.shape)
                    + json.dumps(image_y.numpy_image.shape)
                }
            )
        return results


class MultiImageConsumerManifest(WorkflowBlockManifest):
    type: Literal["MultiImageConsumer"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="shapes", kind=[STRING_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MultiImageConsumer(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MultiImageConsumerManifest

    def run(
        self, images_x: WorkflowImageData, images_y: WorkflowImageData
    ) -> BlockResult:
        return {
            "shapes": json.dumps(images_x.numpy_image.shape)
            + json.dumps(images_y.numpy_image.shape)
        }


class MultiImageConsumerRaisingDimManifest(WorkflowBlockManifest):
    type: Literal["MultiImageConsumerRaisingDim"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="shapes", kind=[STRING_KIND])]

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

    def run(
        self, images_x: WorkflowImageData, images_y: WorkflowImageData
    ) -> BlockResult:
        return [
            {
                "shapes": json.dumps(images_x.numpy_image.shape)
                + json.dumps(images_y.numpy_image.shape)
            }
        ]


class MultiSIMDImageConsumerRaisingDimManifest(WorkflowBlockManifest):
    type: Literal["MultiSIMDImageConsumerRaisingDim"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="shapes", kind=[STRING_KIND])]

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

    def run(
        self, images_x: Batch[WorkflowImageData], images_y: Batch[WorkflowImageData]
    ) -> BlockResult:
        results = []
        for image_x, image_y in zip(images_x, images_y):
            results.append(
                [
                    {
                        "shapes": json.dumps(image_x.numpy_image.shape)
                        + json.dumps(image_y.numpy_image.shape)
                    }
                ]
            )
        return results


class MultiNonSIMDImageConsumerDecreasingDimManifest(WorkflowBlockManifest):
    type: Literal["MultiNonSIMDImageConsumerDecreasingDim"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])
    additional: Union[Selector(), float]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="shapes", kind=[STRING_KIND])]

    @classmethod
    def get_parameters_enforcing_auto_batch_casting(cls) -> List[str]:
        return ["images_x", "images_y"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return -1


class MultiNonSIMDImageConsumerDecreasingDim(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MultiNonSIMDImageConsumerDecreasingDimManifest

    def run(
        self, images_x: Batch[WorkflowImageData], images_y: Batch[WorkflowImageData], additional: Any
    ) -> BlockResult:
        assert not isinstance(additional, Batch)
        print("images_x", images_x, "images_y", images_y)
        results = []
        for image_x, image_y in zip(images_x, images_y):
            results.append(
                json.dumps(image_x.numpy_image.shape)
                + json.dumps(image_y.numpy_image.shape)
            )
        return {"shapes": "\n".join(results)}


class MultiSIMDImageConsumerDecreasingDimManifest(WorkflowBlockManifest):
    type: Literal["MultiSIMDImageConsumerDecreasingDim"]
    images_x: Selector(kind=[IMAGE_KIND])
    images_y: Selector(kind=[IMAGE_KIND])
    additional: Union[Selector(), float]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="shapes", kind=[STRING_KIND])]

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["images_x", "images_y"]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return -1


class MultiSIMDImageConsumerDecreasingDim(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MultiSIMDImageConsumerDecreasingDimManifest

    def run(
        self, images_x: Batch[Batch[WorkflowImageData]], images_y: Batch[Batch[WorkflowImageData]], additional: Any
    ) -> BlockResult:
        assert not isinstance(additional, Batch)
        print("images_x", images_x, "images_y", images_y)
        results = []
        for image_x_batch, image_y_batch in zip(images_x, images_y):
            print("image_x_batch", image_x_batch, "image_x_batch", image_y_batch)
            result = []
            for image_x, image_y in zip(image_x_batch, image_y_batch):
                result.append(
                    json.dumps(image_x.numpy_image.shape)
                    + json.dumps(image_y.numpy_image.shape)
                )
            results.append({"shapes": "\n".join(result)})
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

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["x"]


class IdentitySIMDBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return IdentitySIMDManifest

    def run(self, x: Batch[Any]) -> BlockResult:
        assert isinstance(x, Batch)
        return [{"x": x_el} for x_el in x]


class BoostDimensionalityManifest(WorkflowBlockManifest):
    type: Literal["BoostDimensionality"]
    x: Selector()

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1


class BoostDimensionalityBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoostDimensionalityManifest

    def run(self, x: Any) -> BlockResult:
        return [{"x": x}, {"x": x}]


class DoubleBoostDimensionalityManifest(WorkflowBlockManifest):
    type: Literal["DoubleBoostDimensionality"]
    x: Selector()
    y: Selector()

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x"), OutputDefinition(name="y")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return 1


class DoubleBoostDimensionalityBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return DoubleBoostDimensionalityManifest

    def run(self, x: Any, y: Any) -> BlockResult:
        return [{"x": x, "y": y}, {"x": x, "y": y}]


class NonSIMDConsumerAcceptingListManifest(WorkflowBlockManifest):
    type: Literal["NonSIMDConsumerAcceptingList"]
    x: List[Selector(kind=[IMAGE_KIND])]
    y: List[Selector(kind=[IMAGE_KIND])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x"), OutputDefinition(name="y")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class NonSIMDConsumerAcceptingListBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonSIMDConsumerAcceptingListManifest

    def run(self, x: list, y: list) -> BlockResult:
        return {"x": x, "y": y}


class NonSIMDConsumerAcceptingDictManifest(WorkflowBlockManifest):
    type: Literal["NonSIMDConsumerAcceptingDict"]
    x: Dict[str, Selector(kind=[IMAGE_KIND])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class NonSIMDConsumerAcceptingDictBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonSIMDConsumerAcceptingDictManifest

    def run(self, x: dict) -> BlockResult:
        sorted_keys = sorted(x.keys())
        return {"x": [x[k] for k in sorted_keys]}


class NonSIMDConsumerAcceptingListIncDimManifest(WorkflowBlockManifest):
    type: Literal["NonSIMDConsumerAcceptingListIncDim"]
    x: List[Selector(kind=[IMAGE_KIND])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return 1


class NonSIMDConsumerAcceptingListIncDimBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonSIMDConsumerAcceptingListIncDimManifest

    def run(self, x: list) -> BlockResult:
        return [{"x": x}, {"x": x}]


class NonSIMDConsumerAcceptingDictIncDimManifest(WorkflowBlockManifest):
    type: Literal["NonSIMDConsumerAcceptingDictIncDim"]
    x: Dict[str, Selector(kind=[IMAGE_KIND])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(
        cls,
    ) -> int:
        return 1


class NonSIMDConsumerAcceptingDictIncDimBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonSIMDConsumerAcceptingDictIncDimManifest

    def run(self, x: dict) -> BlockResult:
        sorted_keys = sorted(x.keys())
        return [{"x": [x[k] for k in sorted_keys]}, {"x": [x[k] for k in sorted_keys]}]


class NonSIMDConsumerAcceptingListDecDimManifest(WorkflowBlockManifest):
    type: Literal["NonSIMDConsumerAcceptingListDecDim"]
    x: List[Selector(kind=[IMAGE_KIND])]
    y: Union[Selector(), str]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(
            cls,
    ) -> int:
        return -1

    @classmethod
    def get_parameters_enforcing_auto_batch_casting(cls) -> List[str]:
        return ["x"]


class NonSIMDConsumerAcceptingListDecDimBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonSIMDConsumerAcceptingListDecDimManifest

    def run(self, x: Batch[list], y: str) -> BlockResult:
        assert not isinstance(y, Batch)
        return {"x": [f for e in x for f in e]}


class NonSIMDConsumerAcceptingDictDecDimManifest(WorkflowBlockManifest):
    type: Literal["NonSIMDConsumerAcceptingDictDecDim"]
    x: Dict[str, Selector(kind=[IMAGE_KIND])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_output_dimensionality_offset(
            cls,
    ) -> int:
        return -1

    @classmethod
    def get_parameters_enforcing_auto_batch_casting(cls) -> List[str]:
        return ["x"]


class NonSIMDConsumerAcceptingDictDecDimBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NonSIMDConsumerAcceptingDictDecDimManifest

    def run(self, x: dict) -> BlockResult:
        results = []
        sorted_keys = sorted(x.keys())
        for k in sorted_keys:
            v = x[k]
            assert isinstance(v, Batch)
            result = [e for e in v]
            results.append(result)
        return {"x": results}


def load_blocks() -> List[Type[WorkflowBlock]]:
    return [
        ImageProducerBlock,
        SingleImageConsumer,
        SingleImageConsumerNonSIMD,
        MultiSIMDImageConsumer,
        MultiImageConsumer,
        MultiImageConsumerRaisingDim,
        MultiSIMDImageConsumerRaisingDim,
        IdentityBlock,
        IdentitySIMDBlock,
        MultiNonSIMDImageConsumerDecreasingDim,
        MultiSIMDImageConsumerDecreasingDim,
        BoostDimensionalityBlock,
        DoubleBoostDimensionalityBlock,
        NonSIMDConsumerAcceptingListBlock,
        NonSIMDConsumerAcceptingDictBlock,
        NonSIMDConsumerAcceptingListIncDimBlock,
        NonSIMDConsumerAcceptingDictIncDimBlock,
        NonSIMDConsumerAcceptingListDecDimBlock,
        NonSIMDConsumerAcceptingDictDecDimBlock
    ]
