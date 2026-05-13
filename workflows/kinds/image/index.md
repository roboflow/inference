
# `image` Kind

Image in workflows

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `dict`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `WorkflowImageData`

## Details


This is the representation of image in `workflows`. Underlying data type has different internal and
external representation. As an input we support:

!!! note "Update added in Execution Engine `v1.2.0`"

    `video_metadata` added as optional property - should be injected in context of video processing to 
    provide necessary context for blocks dedicated to video processing.
    

* `np.ndarray` image when Workflows Execution Engine is used directly in `inference` python package (array can be
provided in a form of dictionary presented below, if `video_metadata` is intended to be injected)

* dictionary compatible with [inference image utils](https://inference.roboflow.com/reference/inference/core/utils/image_utils/):

```python
{
    "type": "url",   # there are different types supported, including np arrays and PIL images
    "value": "..."   # value depends on `type`,
    "video_metadata": {  
        # optional - can be added in context of video processing - introduced in 
        # Execution Engine `v1.2.0` - released in inference `v0.23.0`
        "video_identifier": "rtsp://some.com/stream1",
        "comes_from_video_file": False,
        "fps": 23.99,
        "measured_fps": 20.05,
        "frame_number": 24,
        "frame_timestamp": "2024-08-21T11:13:44.313999", 
    }  
}
```

Whe using Workflows Execution Engine exposed behind `inference` server, two most common `type` values are `base64` and 
`url`.

Internally, [`WorkflowImageData`](/workflows/internal_data_types/#workflowimagedata) is used. If you are a
Workflow block developer, we advise checking out [usage guide](/workflows/internal_data_types/#workflowimagedata).


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
