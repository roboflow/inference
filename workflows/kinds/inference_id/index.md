
# `inference_id` Kind

Inference identifier

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


This kind represents identifier of inference process, which is usually opaque string used as correlation
identifier for external systems (like Roboflow Model Monitoring).

Examples:
```
b1851e3d-a145-4540-a39e-875f21f6cd84
```


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
