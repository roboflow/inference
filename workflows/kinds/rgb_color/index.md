
# `rgb_color` Kind

RGB color

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `Tuple[int, int, int]`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `Tuple[int, int, int]`

## Details


This kind represents RGB color as a tuple (R, G, B).

Examples:
```
(128, 32, 64)
(255, 255, 255)
```


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
