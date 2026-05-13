
# `roboflow_solution` Kind

Roboflow Vision Events use case identifier

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


This kind represents a Roboflow Vision Events use case identifier. Use cases are used to
namespace and organize vision events within a workspace. In the workflow builder UI, this
is presented as a "Use Case" selector where users can pick an existing use case or create
a new one.


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
