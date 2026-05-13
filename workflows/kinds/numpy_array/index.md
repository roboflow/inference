
# `numpy_array` Kind

Numpy array

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `list`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `np.ndarray`

## Details

Any np.ndarray object

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
