# Data representations in Workflows

Many frameworks enforce standard data types for developers to work with, and the Workflows ecosystem is no 
exception. While the [kind](/workflows/kinds/index.md) in a Workflow represents a high-level abstraction of the data 
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

!!! Note "Video Metadata"
    
    Since Execution Enginge `v1.2.0`, we have added `video_metadata` into `WorkflowImageData`. This 
    object is supposed to hold the context of video processing and will only be relevant for video processing
    blocks. Other blocks may ignore it's existance if not creating output image (covered in the next section). 

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
    
    # retrieving `VideoMetadata` object - see the usage guide section below
    # if `workflow_image` is not provided with `VideoMetadata` - default metadata object will 
    # be created on accessing the property
    video_metadata = workflow_image.video_metadata 
```

Below you can find an example showcasing how to preserve metadata, while transforming image

```python
import numpy as np

from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


def transform_image(image: WorkflowImageData) -> WorkflowImageData:
    transformed_image = some_transformation(image.numpy_image)
    # `WorkflowImageData` exposes helper method to return a new object with
    # updated image, but with preserved metadata. Metadata preservation
    # should only be used when the output image is compatible regarding
    # data lineage (the predecessor-successor relation for images).
    # Lineage is not preserved for cropping and merging images (without common predecessor)
    # - below you may find implementation tips.
    return WorkflowImageData.copy_and_replace(
        origin_image_data=image,
        numpy_image=transformed_image,
    )


def some_transformation(image: np.ndarray) -> np.ndarray:
    ...
```

??? tip "Images cropping"

    When your block increases dimensionality and provides output with `image` kind - usually that means cropping the 
    image. In such cases input image `video_metadata` is to be removed (as usually it does not make sense to
    keep them, as underlying video processing blocks will not work correctly when for dynamically created blocks).

    Below you can find scratch of implementation for that operation:

    ```python
    from typing import List, Tuple
    
    from dataclasses import replace
    from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

    def crop_images(
        image: WorkflowImageData, 
        crops: List[Tuple[str, int, int, int, int]],
    ) -> List[WorkflowImageData]:
        crops = []
        original_image = image.numpy_image
        for crop_id, x_min, y_min, x_max, y_max in crops:
            cropped_image = original_image[y_min:y_max, x_min:x_max]
            if not cropped_image.size:
                # discarding empty crops
                continue
            result_crop = WorkflowImageData.create_crop(
                origin_image_data=image, 
                crop_identifier=crop_id,
                cropped_image=cropped_image,
                offset_x=x_min,
                offset_y=y_min,
            )
            crops.append(result_crop)
        return crops
    ```

    In some cases you may want to preserve `video_metadata`. Example of such situation is when 
    your block produces crops based on fixed coordinates (like video single footage with multiple fixed Regions of 
    Interest to be applied individual trackers) - then you want result crops to be processed in context of video,
    as if they were produced by separate cameras. To adjust behaviour of `create_crop(...)` method, simply add 
    `preserve_video_metadata=True`:

    ```{ .py linenums="1" hl_lines="11"}
    def crop_images(
        image: WorkflowImageData, 
        crops: List[Tuple[str, int, int, int, int]],
    ) -> List[WorkflowImageData]:
        # [...]
        result_crop = WorkflowImageData.create_crop(
            origin_image_data=image, 
            crop_identifier=crop_id,
            cropped_image=cropped_image,
            offset_x=x_min,
            offset_y=y_min,
            preserve_video_metadata=True
        )
        # [...]
    ```


??? tip "Merging images without common predecessor"

    If common `parent_metadata` cannot be pointed for multiple images you try to merge, you should denote that
    "a new" image appears in the Workflow. To do it simply:

    ```python
    from typing import List, Tuple
    
    from dataclasses import replace
    from inference.core.workflows.execution_engine.entities.base import \
        WorkflowImageData, ImageParentMetadata

    def merge_images(image_1: WorkflowImageData, image_2: WorkflowImageData) -> WorkflowImageData:
        merged_image = some_mergin_operation(
            image_1=image_1.numpy_image,
            image_2=image_2.numpy_image
        )
        new_parent_metadata = ImageParentMetadata(
            # this is just one of the option for creating id, yet sensible one
            parent_id=f"{image_1.parent_metadata.parent_id} + {image_2.parent_metadata.parent_id}"
        )
        return WorkflowImageData(
            parent_metadata=new_parent_metadata,
            numpy_image=merged_imagem
        )
    ```


## `VideoMetadata`

!!! warning "Deprecation"

    [`video_metadata` kind](/workflows/kinds/video_metadata.md) is deprecated - we advise not using that kind in new 
    blocks. `VideoMetadata` data representation became a member of `WorkflowImageData` in Execution Engine `v1.2.0` 
    (`inference` release `v0.23.0`)

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
    
    # Field represents measured FPS of live stream (optional)
    print(video_metadata.measured_fps)
    
    # Field is a flag telling if frame comes from video file or stream.
    # If not possible to be determined - None
    print(video_metadata.comes_from_video_file)
```
