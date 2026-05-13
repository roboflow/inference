
# `video_metadata` Kind

Video image metadata

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

Type: `VideoMetadata`

## Details



!!! warning "Deprecated since Execution Engine `v1.2.0`"

    `inference` maintainers decided to sunset `video_metadata` kind in favour of
    auxiliary metadata added to `image` kind. 

This is representation of metadata that describe images that come from videos.  
It is helpful in cases of stateful video processing, as the metadata may bring 
pieces of information that are required by specific blocks.

The kind has different internal end external representation. As input we support:
```
{
    "video_identifier": "rtsp://some.com/stream1",
    "comes_from_video_file": False,
    "fps": 23.99,
    "measured_fps": 20.05,
    "frame_number": 24,
    "frame_timestamp": "2024-08-21T11:13:44.313999", 
}   
```
Internally, [`VideoMetadata`](/workflows/internal_data_types/#videometadata) is used. If you are a
Workflow block developer, we advise checking out [usage guide](/workflows/internal_data_types/#videometadata).


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
