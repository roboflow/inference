
# `image_keypoints` Kind

Image keypoints detected by classical Computer Vision method

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


The kind represents image keypoints that are detected by classical Computer Vision methods.
Underlying representation is serialised OpenCV KeyPoint object.

Examples:
```
{
    "pt": (2.429290294647217, 1197.7939453125),
    "size": 1.9633429050445557,
    "angle": 183.4322509765625,
    "response": 0.03325376659631729,
    "octave": 6423039,
    "class_id": -1
}
``` 


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
