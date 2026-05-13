
# `list_of_values` Kind

List of values of any type

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `List[Any]`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `List[Any]`

## Details


This kind represents Python list of Any values.

Examples:
```
["a", 1, 1.0]
["a", "b", "c"]
```


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
