import json
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
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
    Selector, StepSelector,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl
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
        self,
        images_x: Batch[WorkflowImageData],
        images_y: Batch[WorkflowImageData],
        additional: Any,
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
        self,
        images_x: Batch[Batch[WorkflowImageData]],
        images_y: Batch[Batch[WorkflowImageData]],
        additional: Any,
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


class SIMDConsumerAcceptingListManifest(WorkflowBlockManifest):
    type: Literal["SIMDConsumerAcceptingList"]
    x: List[Selector(kind=[IMAGE_KIND])]
    y: List[Selector(kind=[IMAGE_KIND])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x"), OutputDefinition(name="y")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["x", "y"]


class SIMDConsumerAcceptingListBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SIMDConsumerAcceptingListManifest

    def run(
        self, x: List[Batch[WorkflowImageData]], y: List[Batch[WorkflowImageData]]
    ) -> BlockResult:
        idx2x = defaultdict(list)
        idx2y = defaultdict(list)
        for batch_x in x:
            for idx, el in enumerate(batch_x):
                idx2x[idx].append(el)
        for batch_y in y:
            for idx, el in enumerate(batch_y):
                idx2y[idx].append(el)
        indices_x = sorted(idx2x.keys())
        indices_y = sorted(idx2y.keys())
        assert indices_x == indices_y
        results = []
        for idx in indices_x:
            results.append({"x": idx2x[idx], "y": idx2y[idx]})
        return results


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


class SIMDConsumerAcceptingDictManifest(WorkflowBlockManifest):
    type: Literal["SIMDConsumerAcceptingDict"]
    x: Dict[str, Selector(kind=[IMAGE_KIND])]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="x")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["x"]


class SIMDConsumerAcceptingDictBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SIMDConsumerAcceptingDictManifest

    def run(self, x: Dict[str, Batch[Any]]) -> BlockResult:
        sorted_keys = sorted(x.keys())
        keys_stashes = {k: {} for k in sorted_keys}
        for key, key_batch in x.items():
            assert isinstance(key_batch, Batch)
            for idx, key_batch_el in enumerate(key_batch):
                keys_stashes[key][idx] = key_batch_el
        reference_indices = None
        for stash in keys_stashes.values():
            sorted_idx = sorted(stash.keys())
            if reference_indices is None:
                reference_indices = sorted_idx
            assert sorted_idx == reference_indices
        assert reference_indices is not None
        results = []
        for idx in reference_indices:
            results.append({"x": [keys_stashes[k][idx] for k in sorted_keys]})
        return results


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


class SIMDConsumerAcceptingDictIncDimManifest(WorkflowBlockManifest):
    type: Literal["SIMDConsumerAcceptingDictIncDim"]
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

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["x"]


class SIMDConsumerAcceptingDictIncDimBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SIMDConsumerAcceptingDictIncDimManifest

    def run(self, x: Dict[str, Batch[Any]]) -> BlockResult:
        sorted_keys = sorted(x.keys())
        keys_stashes = {k: {} for k in sorted_keys}
        for key, key_batch in x.items():
            assert isinstance(key_batch, Batch)
            for idx, key_batch_el in enumerate(key_batch):
                keys_stashes[key][idx] = key_batch_el
        reference_indices = None
        for stash in keys_stashes.values():
            sorted_idx = sorted(stash.keys())
            if reference_indices is None:
                reference_indices = sorted_idx
            assert sorted_idx == reference_indices
        assert reference_indices is not None
        results = []
        for idx in reference_indices:
            results.append(
                [
                    {"x": [keys_stashes[k][idx] for k in sorted_keys]},
                    {"x": [keys_stashes[k][idx] for k in sorted_keys]},
                ]
            )
        return results


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


class SIMDConsumerAcceptingListDecDimManifest(WorkflowBlockManifest):
    type: Literal["SIMDConsumerAcceptingListDecDim"]
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
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["x"]


class SIMDConsumerAcceptingListDecDimBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SIMDConsumerAcceptingListDecDimManifest

    def run(self, x: List[Batch[Batch[WorkflowImageData]]], y: str) -> BlockResult:
        assert not isinstance(y, Batch)
        idx2x = defaultdict(list)
        for batch_x in x:
            for idx, el in enumerate(batch_x):
                idx2x[idx].extend(list(el))
        indices_x = sorted(idx2x.keys())
        results = []
        for idx in indices_x:
            results.append({"x": idx2x[idx]})
        return results


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


class SIMDConsumerAcceptingDictDecDimManifest(WorkflowBlockManifest):
    type: Literal["SIMDConsumerAcceptingDictDecDim"]
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
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["x"]


class SIMDConsumerAcceptingDictDecDimBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return SIMDConsumerAcceptingDictDecDimManifest

    def run(self, x: Dict[str, Batch[Batch[Any]]]) -> BlockResult:
        sorted_keys = sorted(x.keys())
        keys_stashes = {k: {} for k in sorted_keys}
        for key, key_batch in x.items():
            assert isinstance(key_batch, Batch)
            for idx, key_batch_el in enumerate(key_batch):
                assert isinstance(key_batch_el, Batch)
                keys_stashes[key][idx] = list(key_batch_el)
        reference_indices = None
        for stash in keys_stashes.values():
            sorted_idx = sorted(stash.keys())
            if reference_indices is None:
                reference_indices = sorted_idx
            assert sorted_idx == reference_indices
        assert reference_indices is not None
        results = []
        for idx in reference_indices:
            merged = []
            for k in sorted_keys:
                merged.append(keys_stashes[k][idx])
            results.append({"x": merged})
        return results


class AlwaysTerminateManifest(WorkflowBlockManifest):
    type: Literal["AlwaysTerminate"]
    x: Union[Selector(), Any]
    next_steps: List[StepSelector] = Field(
        description="Steps to execute if the condition evaluates to true.",
        examples=[["$steps.on_true"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AlwaysTerminateBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return AlwaysTerminateManifest

    def run(self, x: Any, next_steps: List[StepSelector]) -> BlockResult:
        return FlowControl(mode="terminate_branch")


class AlwaysPassManifest(WorkflowBlockManifest):
    type: Literal["AlwaysPass"]
    x: Union[Selector(), Any]
    next_steps: List[StepSelector] = Field(
        description="Steps to execute if the condition evaluates to true.",
        examples=[["$steps.on_true"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class AlwaysPassBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return AlwaysPassManifest

    def run(self, x: Any, next_steps: List[StepSelector]) -> BlockResult:
        return FlowControl(mode="select_step", context=next_steps)


class EachSecondPassManifest(WorkflowBlockManifest):
    type: Literal["EachSecondPass"]
    x: Union[Selector(), Any]
    next_steps: List[StepSelector] = Field(
        description="Steps to execute if the condition evaluates to true.",
        examples=[["$steps.on_true"]],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class EachSecondPassBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return EachSecondPassManifest

    def __init__(self):
        self._last_passed = False

    def run(self, x: Any, next_steps: List[StepSelector]) -> BlockResult:
        if self._last_passed:
            self._last_passed = False
            return FlowControl(mode="terminate_branch")
        self._last_passed = True
        return FlowControl(mode="select_step", context=next_steps)


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
        NonSIMDConsumerAcceptingDictDecDimBlock,
        SIMDConsumerAcceptingListBlock,
        SIMDConsumerAcceptingDictBlock,
        SIMDConsumerAcceptingDictIncDimBlock,
        SIMDConsumerAcceptingDictDecDimBlock,
        SIMDConsumerAcceptingListDecDimBlock,
        AlwaysTerminateBlock,
        AlwaysPassBlock,
        EachSecondPassBlock,
    ]
