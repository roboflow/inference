
# `serialised_payloads` Kind

Serialised element that is usually accepted by sink

## Data representation


!!! Warning "Data representation"

    This kind has a different internal and external representation. **External** representation is relevant for 
    integration with your workflow, whereas **internal** one is an implementation detail useful for Workflows
    blocks development.



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `List[Union[str, dict]]`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `List[Union[str, bytes, dict]]`

## Details


This value represents list of serialised values. Each serialised value is either
string or bytes - if something else is provided - it will be attempted to be serialised 
to JSON.

Examples:
```
["some", b"other", {"my": "dictionary"}]
```

This kind is to be used in combination with sinks blocks and serializers blocks.
Serializer should output value of this kind which shall then be accepted by sink.


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
