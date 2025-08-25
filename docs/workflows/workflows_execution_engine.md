# Workflows Execution Engine in details

The [compilation process](/workflows/workflows_compiler.md) creates a Workflow Execution graph, which 
holds all the necessary details to run a Workflow definition. In this section, we'll explain the details 
of the execution process.

At a high level, the process does the following:

1. **Validates runtime input:** it checks that all required placeholders from the Workflow definition are filled 
with data and ensures the data types are correct.

2. **Determines execution order:** it defines the order in which the steps are executed.

3. **Prepares step inputs and caches outputs:** it organizes the inputs for each step and saves the outputs 
for future use.

4. **Builds the final Workflow outputs:** it assembles the overall result of the Workflow.

## Validation of runtime input

The Workflow definition specifies the expected inputs for Workflow execution. As discussed 
[earlier](/workflows/definitions.md), inputs can be either batch-oriented data to be processed by steps or parameters that 
configure the step execution. This distinction is crucial to how the Workflow runs and will be explored throughout 
this page.

Starting with input validation, the Execution Engine has a dedicated component that parses and prepares the input 
for use. It recognizes batch-oriented inputs from the Workflow definition and converts them into an internal 
representation (e.g., `WorkflowImage` becomes `Batch[WorkflowImageData]`). This allows block developers to easily 
work with the data. Non-batch-oriented parameters are checked for type consistency against the block manifests 
used to create steps that require those parameters. This ensures that type errors are caught early in the 
execution process.

!!! note

    All batch-oriented inputs must have a size of either 1 or n. When a batch contains only a single element, it is 
    automatically broadcasted across the entire batch.


## Determining execution order

The Workflow Execution Graph is a [directed acyclic graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph), 
which allows us to determine the topological order. Topological order refers to a sequence in which Workflow steps are 
executed, ensuring that each step's dependencies are met before it runs. In other words, if a step relies on the output 
of another step, the Workflow Engine ensures that the dependency step is executed first.

Additionally, the topological structure allows us to identify which steps can be executed in parallel without causing 
race conditions. Parallel execution is the default mode in the Workflows Execution Engine. This means that multiple 
independent steps, such as those used in model ensembling can run simultaneously, resulting in significant 
improvements in execution speed compared to sequential processing.


!!! warning
    
    Due to the parallel execution mode in the Execution Engine (and to avoid unnecessary data copying when passing 
    it to each step), we strongly urge all block developers to avoid mutating any data passed to the block's `run(...)` 
    method. If modifications are necessary, always make a copy of the input object before making changes!


## Handling step inputs and outputs

Handling step inputs and outputs is a complex task for the Execution Engine. This involves:

* Differentiating between SIMD (Single Instruction, Multiple Data) and non-SIMD blocks in relation to their inputs.

* Preparing step inputs while considering conditional execution and the expected input dimensionality.

* Managing outputs from steps that control the flow of execution.

* Registering outputs from data-processing steps, ensuring they match the output dimensionality declared by the blocks.

Let’s explore each of these topics in detail.


### SIMD vs non-SIMD steps

As the definition suggests, a SIMD (Single Instruction, Multiple Data) step processes batch-oriented data, where the 
same operation is applied to each data point, potentially using non-batch-oriented parameters for configuration. 
The output from such a step is expected to be a batch of elements, preserving the order of the input batch elements. 
This applies to both regular processing steps and flow-control steps (see 
[blocks development guide](/workflows/create_workflow_block.md)), where flow-control decisions 
affect each batch element individually.

In essence, the type of data fed into the step determines whether it's SIMD or non-SIMD. If a step requests any 
batch-oriented input, it will be treated as a SIMD step.

Non-SIMD steps, by contrast, are expected to deliver a single result for the input data. In the case of non-SIMD 
flow-control steps, they affect all downstream steps as a whole, rather than individually for each element in a batch.

Historically, Execution Engine could not handle well al scenarios when non-SIMD steps' outputs were fed into SIMD steps
inputs - causing compilation error due to lack of ability to automatically cast such outputs into batches when feeding
into SIMD seps. Starting with Execution Engine `v1.6.0`, the handling of SIMD and non-SIMD blocks has been improved 
through the introduction of **Auto Batch Casting**:

* When a SIMD input is detected but receives scalar data, the Execution Engine automatically casts it into a batch.

* The dimensionality of the batch is determined at compile time, using *lineage* information from other 
batch-oriented inputs when available. Missing dimensions are generated in a manner similar to `torch.unsqueeze(...)`.

* Outputs are evaluated against the casting context - leaving them as scalars when block keeps or decreases output 
dimensionality or **creating new batches** when increase of dimensionality is expected.

!!! warning "We don't support multiple sources of batch-oriented data"

    While Auto Batch Casting simplifies mixing SIMD and non-SIMD blocks, there is one major limitation to be aware of.

    If multiple first-level batches are created from different origins (for instance inputs and steps taking scalars
    and raising output dimensionality into batch at first level of depth), the Execution Engine cannot deterministically 
    construct the output. In previous versions, the assumption was that **outputs were lists directly tied to inputs 
    batch order**. With Auto Batch Casting, batches may also be generated dynamically, and no deterministic ordering 
    can be guaranteed (imagine scenario when you feed batch of 4 images, and there is a block generating dynamic batch 
    with 3 images - when results are to be returned, Execution Engine is unable to determine a single input batch which 
    would dictate output order alignment, which is a hard requirement caused by falty design choices). 

    To prevent unpredictable behaviour, the Execution Engine asserts in this scenario and raises an error instead of 
    proceeding. Resolving this design flaw requires breaking changes and is therefore deferred to 
    **Execution Engine v2.0.**


### Preparing step inputs

Each requested input element may be batch-oriented or not. Non-batch inputs are relatively easy, 
they do not require special treatment. With batch-oriented ones, there is a lot more of a hustle.
Execution Engine maintains indices for each batch-oriented datapoints, for instance:

- if there is input images batch, each element will achieve their own unique index - let's say there is
four input images, the batch indices will be `[(0, ), (1, ), (2, ), (3, )]`.

- step output being not-nested batch will also be indexed, for instance predictions from a model for each of
image mentioned above will also be indexed `[(0, ), (1, ), (2, ), (3, )]`.

- having a block that increases `dimensionality level` - let's say a Dynamic Crop based on
predictions from object-detection model - having 2 crops for first image, 1 for second and three for fourth -
output of such step will be indexed in the following way: `[(0, 0), (0, 1), (1, 0), (3, 0), (3, 1), (3, 2)]`.

Indexing of elements is important while gathering inputs for steps execution. Thanks to them, all batch oriented 
inputs may be aligned - such that Execution Engine will always ship prediction `(3, )` with image `(3, )` and 
crops batch of crops `[(3, 0), (3, 1), (3, 2)]` when any step requests it. 

Each requested input element can either be batch-oriented or non-batch. Non-batch inputs are straightforward and don't 
require special handling. However, batch-oriented inputs involve more complexity. The Execution Engine tracks indices 
for each batch-oriented data point. For example:

- If there's a batch of input images, each element receives its own unique index. For a batch of four images, the 
indices would be `[(0,), (1,), (2,), (3,)]`.

- A step output which do not increase dimensionality will also be indexed similarly. For example, model predictions 
for each of the four images would have indices `[(0,), (1,), (2,), (3,)]`.

* If a block increases the `dimensionality_level` (e.g., a dynamic crop based on predictions from an object 
detection model), the output will be indexed differently. Suppose there are 2 crops for the first image, 
1 for the second, and 3 for the fourth. The indices for this output would be 
`[(0, 0), (0, 1), (1, 0), (3, 0), (3, 1), (3, 2)]`.

Indexing is crucial for aligning inputs during step execution. The Execution Engine ensures that all batch-oriented 
inputs are synchronized. For example, it will match prediction `(3,)` with image `(3,)` and the corresponding batch 
of crops `[(3, 0), (3, 1), (3, 2)]` when a step requests them.

!!! Note

    Keeping data lineage in order during compilation simplifies execution. The Execution Engine doesn't need to 
    verify if dynamically created nested batches come from the same source. Its job is to align indices when 
    preparing step inputs.


#### Additional Considerations

##### Input Dimensionality Offsets
Workflow blocks define how input dimensionality is handled. 
If the Execution Engine detects a difference in input dimensionality, it will wrap the larger dimension into a batch. 
For example, if a block processes both input images and dynamically cropped images, the latter will be wrapped into a 
batch so that each top-level image is processed with its corresponding batch of crops.

##### Deeply nested batches are flattened before step execution 
Given the block defines all input at the same dimensionality level, no matter how deep the nesting of input batches is, 
step input will be flattened to a single batch and indices in the outputs will be automagically maintained by 
Execution Engine.

##### Conditional Execution
Flow-control blocks manage which steps should be executed based on certain conditions. During compilation, 
steps affected by these conditions are flagged. When constructing their inputs, a mask for flow-control exclusion 
(both SIMD- and non-SIMD-oriented) is applied. Based on this mask, specific input elements will be replaced with 
`None`, representing an empty value.

By default, blocks don't accept empty values, so any `None` at index `(i,)` in a batch will cause that index to be 
excluded from processing. This is how flow control is managed within the Execution Engine. Some blocks, however, 
are designed to handle empty inputs. In such cases, while the flow-control mask will be applied, empty inputs 
won't be eliminated from the input batch.


##### Batch Processing Mode
Blocks may either process batch inputs all at once or, by default, require the Execution Engine to loop over each 
input and repeatedly invoke the block's `run(...)` method.

### Managing flow-control steps outputs

The outputs of flow-control steps are unique because these steps determine which data points should be 
passed to subsequent steps which is roughly similar to outcome of this pseudocode:

```python
if condition(A):
    step_1(A)
    step_2(A)
else:
    step_3(A)
```

The Workflows Execution Engine parses the outputs from flow-control steps and creates execution branches. 
Each branch has an associated mask:

* For **SIMD branches**, the mask contains a set of indices that will remain active for processing.

* For **non-SIMD branches**, the mask is a simple `True` / `False` value that determines whether the entire 
branch is active.

After a flow-control step executes, this mask is registered and applied to any steps affected by the decision. 
This allows the Engine to filter out specific data points from processing in the downstream branch. 
If a data point is excluded from the first step in a branch (due to the masking), that data point is automatically 
eliminated from the entire branch (as a result of exclusion of empty inputs by default).

### Caching steps outputs

It's not just the outcomes of flow-control steps that need to be managed carefully—data processing steps also require 
attention to ensure their results are correctly passed to other steps. The key aspect here is properly indexing 
the outputs.

In simple cases where all inputs share the same `dimensionality level` and the output maintains that same 
dimensionality, the Execution Engine's main task is to preserve the order of input indices. However, when input 
dimensionalities differ, the Workflow block used to create the step determines how indexing should be handled.

If the dimensionality changes during processing, the Execution Engine either uses the high-level index or creates 
nested dimensions dynamically based on the length of element lists in the output. This ensures proper alignment and 
tracking of data across steps.


## Building Workflow outputs


For details on how outputs are constructed, please refer to the information provided on the 
[Workflows Definitions](/workflows/definitions.md) page and the 
[Output Construction](/workflows/workflow_execution.md#output-construction) section of the Workflow Execution 
documentation.
