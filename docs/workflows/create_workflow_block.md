# `workflows block` creation crash course

At start, we need to see what is required to be implemented (via `block` base class interface). That would
be the following methods:

```python
class WorkflowBlock(ABC):

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        pass

    @abstractmethod
    async def run_locally(
        self,
        *args,
        **kwargs,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], FlowControl]]:
        pass
```

Let's start from input manifest assuming we want to build cropping `block`. We would need the following as
input:

- image - in `workflows` it may come as selector either to workflow input or other step output

- predictions - predictions with bounding boxes (made against the image) - that we can use to crop

Implementation:

```python
from typing import Literal, Union

from pydantic import AliasChoices, ConfigDict, Field
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    WorkflowImageSelector,
    StepOutputImageSelector,
    StepOutputSelector,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "description": "This block produces dynamic crops based on detections from detections-based model.",
            "docs": "https://inference.roboflow.com/workflows/crop",
            "block_type": "transformation",
        }
    )
    type: Literal["Crop"]
    image: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="The image to infer on",
        examples=["$inputs.image", "$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference to predictions of detection-like model, that can be based of cropping "
                    "(detection must define RoI - eg: bounding box)",
        examples=["$steps.my_object_detection_model.predictions"],
        validation_alias=AliasChoices("predictions", "detections"),
    )
```

As an output we are going to provide cropped images, so we need to declare that:

```python
from typing import List

from inference.core.workflows.prototypes.block import (
    WorkflowBlockManifest,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BATCH_OF_PARENT_ID_KIND,
)


class BlockManifest(WorkflowBlockManifest):
    # [...] input properties hidden

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="crops", kind=[BATCH_OF_IMAGES_KIND]),
            OutputDefinition(name="parent_id", kind=[BATCH_OF_PARENT_ID_KIND]),
        ]
```
In the current version, it is required to define `parent_id` for each element that we output from steps.

Then we define implementation starting from class method that will provide manifest:

```python
from typing import Type

from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class DynamicCropBlock(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest
```

Finally, we need to provide implementation for the logic:

```python
from typing import List, Tuple, Any
import itertools
import numpy as np

from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl


class DynamicCropBlock(WorkflowBlock):

    async def run_locally(
            self,
            image: List[dict],
            predictions: List[List[dict]],
    ) -> Tuple[List[Any], FlowControl]:
        decoded_images = [load_image(e) for e in image]
        decoded_images = [
            i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_images
        ]
        origin_image_shape = extract_origin_size_from_images(
            input_images=image,
            decoded_images=decoded_images,
        )
        result = list(
            itertools.chain.from_iterable(
                crop_image(image=i, predictions=d, origin_size=o)
                for i, d, o in zip(decoded_images, predictions, origin_image_shape)
            )
        )
        if len(result) == 0:
            return result, FlowControl(mode="terminate_branch")
        return result, FlowControl(mode="pass")


def crop_image(
        image: np.ndarray,
        predictions: List[dict],
        origin_size: dict,
) -> List[Dict[str, Union[dict, str]]]:
    crops = []
    for detection in predictions:
        x_min, y_min, x_max, y_max = detection_to_xyxy(detection=detection)
        cropped_image = image[y_min:y_max, x_min:x_max]
        crops.append(
            {
                "crops": {
                    IMAGE_TYPE_KEY: ImageType.NUMPY_OBJECT.value,
                    IMAGE_VALUE_KEY: cropped_image,
                    PARENT_ID_KEY: detection[DETECTION_ID_KEY],
                    ORIGIN_COORDINATES_KEY: {
                        CENTER_X_KEY: detection["x"],
                        CENTER_Y_KEY: detection["y"],
                        WIDTH_KEY: detection[WIDTH_KEY],
                        HEIGHT_KEY: detection[HEIGHT_KEY],
                        ORIGIN_SIZE_KEY: origin_size,
                    },
                },
                "parent_id": detection[DETECTION_ID_KEY],
            }
        )
    return crops
```

Point out few details:
- image come as list of dicts - each element is standard `inference` image description ("type" and "value" provided
so `inference` loader can be used)

- results of steps are provided as **list of dicts** - each element of that list ships two keys - `crops` 
and `parent_id` - which are exactly matching outputs that we defined previously.

- we use `FlowControl` here - which is totally optional, but if result is a tuple with second element being
`FlowControl` object - step may influence execution of `wokrflow` - in this case, we decide to `terminate_branch`
(stop computations that follows this `step`) - given that we are not able to find any crops after processing.