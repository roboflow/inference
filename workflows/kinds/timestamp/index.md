
# `timestamp` Kind

Timestamp object

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `str`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `datetime`

## Details


Representation of timestamp in Workflows. 

Internally represented as `datetime.datetime` object (with time-zone info not required). 
Can be serialized / deserialized to / from [ISO-format timestamps](https://docs.python.org/3/library/datetime.html#datetime.date.fromisoformat).


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
