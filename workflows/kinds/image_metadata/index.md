
# `image_metadata` Kind

Dictionary with image metadata required by supervision

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `dict`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `dict`

## Details


This kind represent batch of prediction metadata providing information about the image that prediction was made against.

Examples:
```
[{"width": 1280, "height": 720}, {"width": 1920, "height": 1080}]
[{"width": 1280, "height": 720}]
```


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
