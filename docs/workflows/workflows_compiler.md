# Compilation of Workflow Definition

Compilation is a process that takes a document written in a programming language, checks its correctness, 
and transforms it into a format that the execution environment can understand.

A similar process happens in the Workflows ecosystem whenever you want to run a Workflow Definition. 
The Workflows Compiler performs several steps to transform a JSON document into a computation graph, which 
is then executed by the Workflows Execution Engine. While this process can be complex, understanding it can 
be helpful for developers contributing to the ecosystem. In this document, we outline key details of the 
compilation process to assist in building Workflow blocks and encourage contributions to the core Execution Engine.


!!! Note

    This document covers the design of Execution Engine `v1` (which is current stable version). Please 
    acknowledge information about [versioning](/workflows/versioning.md) to understand Execution Engine 
    development cycle.

## Stages of compilation

Workflow compilation involves several stages, including:

1. Loading available blocks: Gathering all the blocks that can be used in the workflow based on 
configuration of execution environment

2. Compiling dynamic blocks: Turning [dynamic blocks definitions](/workflows/custom_python_code_blocks.md) into 
standard Workflow Blocks

3. Parsing the Workflow Definition: Reading and interpreting the JSON document that defines the workflow, detecting 
syntax errors

4. Building Workflow Execution Graph: Creating a graph that defines how data will flow through the workflow 
during execution and verifying Workflow integrity

5. Initializing Workflow steps from blocks: Setting up the individual workflow steps based on the available blocks, 
steps definitions and configuration of execution environment.

Let's take a closer look at each of the workflow compilation steps.

### Workflows blocks loading

As described in the [blocks bundling guide](/workflows/blocks_bundling.md), a group of Workflow blocks can be packaged 
into a workflow plugin. A plugin is essentially a standard Python library that, in its main module, exposes specific 
functions allowing Workflow Blocks to be dynamically loaded.

The Workflows Compiler and Execution Engine are designed to be independent of specific Workflow Blocks, and the 
Compiler has the ability to discover and load blocks from plugins.

Roboflow provides the `roboflow_core` plugin, which includes a set of basic Workflow Blocks that are always 
loaded by the Compiler, as both the Compiler and these blocks are bundled in the `inference` package.

For custom plugins, once they are installed in the Python environment, they need to be referenced using an environment 
variable called `WORKFLOWS_PLUGINS`. This variable should contain the names of the Python packages that contain the 
plugins, separated by commas. 

For example, if you have two custom plugins, `numpy_plugin` and `pandas_plugin`, you can enable them in 
your Workflows environment by setting:
```bash
export WORKFLOWS_PLUGINS="numpy_plugin,pandas_plugin"
```

Both `numpy_plugin` and `pandas_plugin` **are not paths to library repositories**, but rather
names of the main modules of libraries shipping plugins (`import numpy_plugin` must work in your 
Python environment for the plugin to be possible to be loaded).

Once Compiler loads all plugins it is ready for the next stage of compilation.

### Compilation of dynamic blocks

!!! Note

    The topic of [dynamic Python blocks](/workflows/custom_python_code_blocks.md) is covered
    in separate docs page. To unerstand the content of this section you only need to know that
    there is a way to define Workflow Blocks in-place in Workflow Definition - specifying
    both block manifest and Python code in JSON document. This functionality only works if you
    run Workflows Execution Engine on your hardware and is disabled ad Roboflow hosted platform.

The Workflows Compiler can transform Dynamic Python Blocks, defined directly in a Workflow Definition, into 
full-fledged Workflow Blocks at runtime. The Compiler generates these block classes dynamically based on the 
block's definition, eliminating the need for developers to manually create them as they would in a plugin.

Once this process is complete, the dynamic blocks are added to the pool of available Workflow Blocks. These blocks 
can then be used in the `steps` section of your Workflow Definition, just like any other standard block.

### Parsing Workflow Definition

Once all Workflow Blocks are loaded, the Compiler retrieves the manifest classes for each block. 
These manifests are `pydantic` data classes that define the structure of step entries in definition.
At parsing stage, errors with Workflows Definition are alerted, for example:

- usage of non-existing blocks

- invalid configuration of steps

- lack of required parameters for steps

Thanks to `pydantic`, the Workflows Compiler doesn't need its own parser. Additionally, blocks creators use standard 
Python library to define block manifests.


### Building Workflow Execution Graph

Building the Workflow Execution graph is the most critical stage of Workflow compilation. 
Here's how it works:

#### Adding Vertices
First, each input, step and output are added as vertices in the graph, with each vertex given a special label 
for future identification. These vertices also include metadata, like marking input vertices with seeds for data 
lineage tracking (more on this later).

#### Adding Edges
After placing the vertices, the next step is to create edges between them based on the selectors defined in 
the Workflow. The Compiler examines the block manifests to determine which properties can accept selectors 
and the expected "kind" of those selectors. This enables the Compiler to detect errors in the Workflow 
definition, such as:

- Providing an output kind from one step that doesn't match the expected input kind of the next step.

- Referring to non-existent steps or inputs.

Each edge also contains metadata indicating which input property is being fed by the output data, which is 
helpful at later stages of compilation and during execution

!!! Note
    
    Normally, step inputs "request" data from step outputs, forming an edge from Step A's output to Step B's input 
    during Step B's processing. However, [control-flow blocks](/workflows/create_workflow_block.md) are an exception, 
    as they both accept data and declare other steps in the manifest, creating a special flow-control edge in the graph.

#### Structural Validation

Once the graph is constructed, the Compiler checks for structural issues like cycles to ensure the graph can be 
executed properly.

#### Data Lineage verification

Finally, data lineage properties are populated from input nodes and carried through the graph. So, what is 
data lineage? Lineage is a list of identifiers that track the creation and nesting of batches through the steps, 
determining:

- the source path of data

- `dimensionality level` of data

- compatibility of different pieces of data that may be referred by a step - ensuring that step will only 
take corresponding batches elements from multiple sources (such that batch element index `example: (1, 2)` refers to
the exact same piece of data when two batch-oriented inputs are connected into the step and not to some randomly 
provided batches with different lineage that does not make sense to process together)

Each time a new nested batch is created by a step, a unique identifier is added to the lineage of the output. 
This allows the Compiler to track and verify if the inputs across steps are compatible.

!!! Note
    
    Fundamental assumption of data lineage is that all batch-oriented inputs are granted
    the same lineage identifier - so implicitly it enforces all input batches to be fed with 
    data that has corresponding data-points at corresponding positions in batches. For instance, 
    if your Workflow compares `image_1` to `image_2` (and you declare those two inputs in Wofklow Definition),
    the Compiler assumes the elements of `image_1[3]` to correspond with `image_2[3]`.


Thanks to lineage tracking, the Compiler can detect potential mistakes. For example, if you attempt to connect two 
dynamic crop outputs to a single step's inputs, the Compiler will notice that the number of crops in each 
output may not match. This would result in nested batch elements with mismatched indices, which could lead to 
unpredictable results during execution if the situation is not prevented.

!!! Tip "Example of lineage missmatch"

    Imagine the following scenario:
    
    - you declare single image input in your Workflow

    - at first you perform object detection using two different models

    - you use two dynamic crop steps - to crop based on first and second model predictions 
    respectivelly

    - now you want to use block to compare two images features (using classical Compute Vision methods)

    What would you expect to happen when you plug inputs from those two crop steps into comparison block?
    
    - **Without** tracing the lineage you would "flatten" and "zip" those two batches and
    pass pairs of images to comparison block - the problem is that in this case you cannot 
    determine if the comparisons between those elements actually makes sense - probably do not!
    
    - **With** lineage tracing - Compiler knows that you attempt to feed two batches with lineages
    that do not match regarding last nesting level and raises compilation error.

    One may ask - "ok, but maybe I would like to apply secondary classifier on both crops and 
    merge results at the end to get all results in single output - is that possible?". The answer is
    **yes** - as mentioned above, nested batches differ only at the last lineage level - so when we use 
    some blocks from "dimensionality collapse" category - we will align the results of secondary classifiers 
    into batches at `dimensionality level` 1 with matching lineage.


As outlined in the section dedicated to [blocks development](/workflows/create_workflow_block.md), each block can define 
the expected dimensionality of its inputs and outputs. This refers to how the data should be structured. 
For example, if a block needs an `image` input that's one level above a batch of `predictions`, the Compiler will 
check that this requirement is met when verifying the Workflow step. If the connections between steps don’t match 
the expected dimensionality, an error will occur. Additionally, each input is also verified to ensure it is compatible 
based on data lineage. Once the step passes validation, the output dimensionality is determined and will be used to 
check compatibility with subsequent steps.

It’s important to note that blocks define dimensionality requirements in relative terms, not absolute. This means 
a block specifies the difference (or offset) in dimensionality between its inputs and outputs. This approach allows 
blocks to work flexibly at any dimensionality level.


!!! Note

    In version 1, the Workflows Compiler only supports blocks that work across two different `dimensionality levels`. 
    This was done to keep the design straightforward. If there's a need for blocks that handle more 
    `dimensionality levels` in the future, we will consider expanding this support.


#### Denoting flow-control

The Workflows Compiler helps the Execution Engine manage flow-control structures in workflows. It marks specific 
attributes that allow the system to understand how flow-control impacts building inputs for certain steps and the 
execution of the workflow graph (for more details, see the 
[Execution Engine docs](/workflows/workflows_execution_engine.md)).

To ensure the workflow structure is correct, the Compiler checks data lineage for flow-control steps in a 
similar way as described in the section on [data-lineage verification](#data-lineage-verification).

The Compiler assumes flow-control steps can affect other steps if:

* **The flow-control step operates on non-batch-oriented inputs** - in this case, the flow-control step can 
either allow or prevent the connected step (and related steps) from running entirely, even if the input 
is a batch of data - all batch elements are affected.

* **The flow-control step operates on batch-oriented inputs with compatible lineage** - here, the flow-control step 
can decide separately for each element in the batch which ones will proceed and which ones will be stopped.

#### Batch-orientation compatibility

As it was outlined, Workflows define **batch-oriented data** and **scalars**.
From [the description of the nature of data in Workflows](/workflows/workflow_execution.md#what-is-the-data), 
you can also conclude that operations which are executed against batch-oriented data
have two almost equivalent ways of running:

* **all-at-once:** taking whole batches of data and processing them

* **one-by-one:** looping over batch elements and getting results sequentially

Since the default way for Workflow blocks to deal with the batches is to consume them element-by-element, 
**there is no real difference** between **batch-oriented data** and **scalars** 
in such case. Execution Engine simply unpacks scalars from batches and pass them to each step.

The process may complicate when block accepts batch input. You will learn the 
details in [blocks development guide](/workflows/create_workflow_block.md), but 
block is required to denote each input that must be provided *batch-wise* and all inputs
which can be fed with both batch-oriented data and scalars at the same time (which is much 
less common case). In such cases, *lineage* is used to deduce if the actual data fed into 
every step input is *batch* or *scalar*. When violation is detected (for instance *scalar* is provided for input 
that requires batches or vice versa) - the error is raised.


!!! Note "Potential future improvements"

    At this moment, we are not sure if the behaviour described above is limiting the potential of
    Workflows ecosystem. If you see that your Workflows cannot run due to the errors 
    being result of described mechanism - please let us know in 
    [GitHub issues](https://github.com/roboflow/inference/issues). 


## Initializing Workflow steps from blocks

The documentation often refers to a Workflow Step as an instance of a Workflow Block, which serves as its prototype. 
To put it simply, a Workflow Block is a class that implements specific behavior, which can be customized by 
configuration—whether it's set by the environment running the Execution Engine, the Workflow definition, 
or inputs at runtime.

In programming, we create an instance of a class using a constructor, usually requiring initialization parameters. 
One the same note, Workflow Blocks are initialized by the Workflows Compiler whenever a step in the Workflow 
references that block. Some blocks may need specific initialization parameters, while others won't.

When a block requires initialization parameters:

* The block must declare the parameters it needs, as explained in detail in 
the [blocks development guide](/workflows/create_workflow_block.md)

* The values for these parameters must be provided from the environment where the Workflow is being executed.

* The values for these parameters must be provided from the environment where the Workflow is being executed.

This second part might seem tricky, so let’s look at an example. In the [in user guide](/workflows/modes_of_running.md),
under the section showing how to integrate with Workflows using the `inference` Python package, you might come 
across code like this:

```python
[...]
# example init paramaters for blocks - dependent on set of blocks 
# used in your workflow
workflow_init_parameters = {
    "workflows_core.model_manager": model_manager,
    "workflows_core.api_key": "<YOUR-API-KEY>,
    "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
}

# instance of Execution Engine - init(...) method invocation triggers
# the compilation process
execution_engine = ExecutionEngine.init(
    ...,
    init_parameters=workflow_init_parameters,
    ...,
)
[...]
```

In this example, `workflow_init_parameters contains` values that the Compiler uses when 
initializing Workflow steps based on block requests.


Initialization parameters (often called "init parameters") can be passed to the Compiler in two ways:

* **Explicitly:** You provide specific values (numbers, strings, objects, etc.).

* **Implicitly:** Default values are defined within the Workflows plugin, which can either be specific values or 
functions (taking no parameters) that generate values dynamically, such as from environmental variables.


The dictionary `workflow_init_parameters` shows explicitly passed init parameters. The structure of the keys 
is important: `{plugin_name}.{init_parameter_name}`. You can also specify just `{init_parameter_name}`, but this 
changes how parameters are resolved.

### How Parameters Are Resolved?

When the Compiler looks for a block’s required init parameter, it follows this process:

1. **Exact Match:** It first checks the explicitly provided parameters for an exact match to 
`{plugin_name}.{init_parameter_name}`.

2. **Default Parameters:** If no match is found, it checks the plugin’s default parameters.

3. **General Match:** Finally, it looks for a general match with just `{init_parameter_name}` in the explicitly 
provided parameters.

This mechanism allows flexibility, as some block parameters can have default values while others must be 
provided explicitly. Additionally, it lets certain parameters be shared across different plugins.
