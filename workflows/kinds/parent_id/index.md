
# `parent_id` Kind

Identifier of parent for step output

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `str`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `str`

## Details


This kind represent batch of prediction metadata providing information about the context of prediction.
For example - whenever there is a workflow with multiple models - such that first model detect objects 
and then other models make their predictions based on crops from first model detections - `parent_id`
helps to figure out which detection of the first model is associated to which downstream predictions.

Examples:
```
"uuid-1"
"uuid-2"
```


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
