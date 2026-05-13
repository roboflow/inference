
# `embedding` Kind

A list of floating point numbers representing a vector embedding.

## Data representation



### External

External data representation is relevant for Workflows clients - it dictates what is the input and output format of
data.

Type: `List[float]`

### Internal

Internal data representation is relevant for Workflows blocks creators - this is the type that will be provided
by Execution Engine in runtime to the block that consumes input of this kind.

Type: `List[float]`

## Details


This kind represents a vector embedding. It is a list of floating point numbers.

Embeddings are used in various machine learning tasks like clustering, classification,
and similarity search. They are used to represent data in a continuous, low-dimensional space.

Typically, vectors that are close to each other in the embedding space are considered similar.


<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>
