
# `contours` Kind

List of numpy arrays where each array represents contour points

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `List[list]`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `List[np.ndarray]`

## Details


This kind represents a value of a list of numpy arrays where each array represents contour points.

Example:
```
[
    np.array([[10, 10],
              [20, 20],
              [30, 30]], dtype=np.int32),
    np.array([[50, 50],
              [60, 60],
              [70, 70]], dtype=np.int32)
]
```


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
