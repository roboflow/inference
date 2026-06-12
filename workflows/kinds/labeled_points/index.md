
# `labeled_points` Kind

List of 2D points with positive/negative labels

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `List[Dict[str, Any]]`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `List[Dict[str, Any]]`

## Details


This kind represents a list of 2D points with positive/negative labels, used for interactive
point prompting of segmentation models (like the SAM model family).

Each point is a dict with `x` and `y` absolute pixel coordinates and a `positive` flag:
positive points mark the object to segment, negative points mark regions to exclude.

Example:
```
[
    {"x": 300, "y": 220, "positive": true},
    {"x": 380, "y": 260, "positive": false}
]
```

Points may also be provided as `(x, y)` or `(x, y, positive)` sequences - missing `positive`
labels are assumed to be positive. Runtime inputs of this kind are normalised to the dict form.


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
