# Data representations in Workflows

Many frameworks enforce standard data types for developers to work with, and the Workflows ecosystem is no 
exception. While the [kind](/workflows/kinds) in a Workflow represents a high-level abstraction of the data 
being passed through, it's important to understand the specific data types that will be provided to the 
`WorkflowBlock.run(...)` method when building Workflow blocks.

And this is exactly what you will learn here.


## `Batch`

When a Workflow block declares batch processing, it uses a special container type called Batch. 
All batch-oriented parameters are wrapped with `Batch[X]`, where X is the data type:

```python
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.prototypes.block import BlockResult

# run method of Workflow block
def run(self, x: Batch[int], y: Batch[float]) -> BlockResult:
   pass
```

The `Batch` type functions similarly to a Python list, with one key difference: it is **read-only**. You cannot modify 
its elements, nor can you add or remove elements. However, several useful operations are available:

### Iteration through elements

```python
from inference.core.workflows.execution_engine.entities.base import Batch


def iterate(batch: Batch[int]) -> None:
    for element in batch:
        print(element)
```

### Zipping multiple batches 

!!! Note "Do not worry about batches alignment"

    The Execution Engine ensures that batches provided to the run method are of equal size, preventing any loss of 
    elements due to unequal batch sizes during iteration.


```python
from inference.core.workflows.execution_engine.entities.base import Batch


def zip_batches(batch_1: Batch[int], batch_2: Batch[int]) -> None:
    for element_1, element_2 in zip(batch_1, batch_2):
        print(element_1, element_2)
```

### Getting batch element indices

This returns a list of tuples where each tuple represents the position of the batch element in 
potentially nested batch structures.

```python
from inference.core.workflows.execution_engine.entities.base import Batch


def discover_indices(batch: Batch[int]) -> None:
    for index in batch.indices:
        print(index)  # e.g., (0,) for 1D batch, (1, 3) for 2D nested batch, etc.
```

### Iterating while retrieving both elements and their indices

```python
from inference.core.workflows.execution_engine.entities.base import Batch


def iterate_with_indices(batch: Batch[int]) -> None:
    for index, element in batch.iter_with_indices():
        print(index, element)
```

### Additional methods of `Batch` container

There are other methods in the Batch interface, such as `remove_by_indices(...)` or `broadcast(...)`, but these 
are not intended for use within Workflow blocks. These methods are primarily used 
**by the Execution Engine when providing data to the block.**


## `WorkflowImageData`

`WorkflowImageData` is a dataclass that encapsulates an image along with its metadata, providing useful methods to 
manipulate the image representation within a Workflow block.

Some users may expect an `np.ndarray` to be provided directly by the Execution Engine when an `image` kind is declared. 
While that could be a convenient and straightforward approach, it introduces limitations, such as:

* **Lack of metadata:** With only an `np.ndarray`, there's no way to attach metadata such as data lineage or the 
image's location within the original file (e.g., when working with cropped images).

* **Inability to cache multiple representations:** If multiple blocks need to serialize and send the image over 
HTTP, WorkflowImageData allows caching of different image representations, such as base64-encoded versions, 
improving efficiency.

Operating on `WorkflowImageData` is fairly simple once you understand its interface. Here are some of the key
methods and properties:

```python
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def operate_on_image(workflow_image: WorkflowImageData) -> None:
    # Getting an np.ndarray representing the image.
    numpy_image = workflow_image.numpy_image  
    
    # Getting base64-encoded JPEG representation of the image, ideal for API transmission.
    base64_image: str = workflow_image.base64_image  
    
    # Converting the image into a format compatible with inference models.
    inference_format = workflow_image.to_inference_format()  
    
    # Accesses metadata related to the parent image
    parent_metadata = workflow_image.parent_metadata
    print(parent_metadata.parent_id)  # parent identifier
    origin_coordinates = parent_metadata.origin_coordinates  # optional object with coordinates
    print(
        origin_coordinates.left_top_x, origin_coordinates.left_top_y, 
        origin_coordinates.origin_width, origin_coordinates.origin_height,
    )
    
    # or the same for root metadata (the oldest ancestor of the image - Workflow input image)
    root_metadata = workflow_image.workflow_root_ancestor_metadata
```

Below you can find an example showcasing how to preserve metadata, while transforming image

```python
import numpy as np

from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def transform_image(image: WorkflowImageData) -> WorkflowImageData:
    transformed_image = some_transformation(image.numpy_image)
    return WorkflowImageData(
        parent_metadata=image.parent_metadata,
        workflow_root_ancestor_metadata=image.workflow_root_ancestor_metadata,
        numpy_image=transformed_image,
    )

def some_transformation(image: np.ndarray) -> np.ndarray:
    ...
```

??? tip "Images cropping"

    When your block increases dimensionality and provides output with `image` kind - usually that means cropping the 
    image. Below you can find scratch of implementation for that operation:
    
    ```python
    from typing import List, Tuple
    
    from dataclasses import replace
    from inference.core.workflows.execution_engine.entities.base import \
        WorkflowImageData, ImageParentMetadata, OriginCoordinatesSystem
    
    
    def crop_images(
        image: WorkflowImageData, 
        crops: List[Tuple[str, int, int, int, int]],
    ) -> List[WorkflowImageData]:
        crops = []
        original_image = image.numpy_image
        for crop_id, x_min, y_min, x_max, y_max in crops:
            cropped_image = original_image[y_min:y_max, x_min:x_max]
            crop_parent_metadata = ImageParentMetadata(
                parent_id=crop_id,
                origin_coordinates=OriginCoordinatesSystem(
                    left_top_x=x_min,
                    left_top_y=y_min,
                    origin_width=original_image.shape[1],
                    origin_height=original_image.shape[0],
                ),
            )
            # adding shift to root ancestor coordinates system
            crop_root_ancestor_coordinates = replace(
                image.workflow_root_ancestor_metadata.origin_coordinates,
                left_top_x=image.workflow_root_ancestor_metadata.origin_coordinates.left_top_x + x_min,
                left_top_y=image.workflow_root_ancestor_metadata.origin_coordinates.left_top_y + y_min,
            )
            workflow_root_ancestor_metadata = ImageParentMetadata(
                parent_id=image.workflow_root_ancestor_metadata.parent_id,
                origin_coordinates=crop_root_ancestor_coordinates,
            )
            result_crop = WorkflowImageData(
                parent_metadata=crop_parent_metadata,
                workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
                numpy_image=cropped_image,
            )
            crops.append(result_crop)
        return crops
    ```


## `VideoMetadata`

!!! warning "Early adoption"

    `video_metadata` kind and `VideoMetadata` data representatio are in early adoption at the moment. They represent
    new batch-oriented data type added to Workflows ecosystem that should provide extended set of metadata on top
    of video frame, to make it possible to create stateful video processing blocks like ByteTracker. 

    Authors still experiment with different, potenially more handy ways of onboarding video processing. Stay tuned 
    and observe [video processing updates](/workflows/video_processing/overview/).

`VideoMetadata` is a dataclass that provides the following metadata about video frame and video source:

```python
from inference.core.workflows.execution_engine.entities.base import VideoMetadata


def inspect_vide_metadata(video_metadata: VideoMetadata) -> None:
    # Identifier string for video. To be treated as opaque.
    print(video_metadata.video_identifier)
    
    # Sequential number of the frame
    print(video_metadata.frame_number)
    
    # The timestamp of video frame. When processing video it is suggested that "
    # "blocks rely on `fps` and `frame_number`, as real-world time-elapse will not "
    # "match time-elapse in video file
    print(video_metadata.frame_timestamp)
    
    # Field represents FPS value (if possible to be retrieved) (optional)
    print(video_metadata.fps)
    
    # Field is a flag telling if frame comes from video file or stream.
    # If not possible to be determined - None
    print(video_metadata.comes_from_video_file)
```
