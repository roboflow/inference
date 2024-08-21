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
    acknowledge information about [versioning](/workflows/versioning) to understand Execution Engine 
    development cycle.

## Stages of compilation

Workflow compilation involves several stages, including:

1. Loading available blocks: Gathering all the blocks that can be used in the workflow based on 
configuration of execution environment

2. Compiling dynamic blocks: Turning [dynamic blocks definitions](/workflows/custom_python_code_blocks) into 
standard Workflow Blocks

3. Parsing the Workflow Definition: Reading and interpreting the JSON document that defines the workflow, detecting 
syntax errors

4. Building Workflow Execution Graph: Creating a graph that defines how data will flow through the workflow 
during execution and verifying Workflow integrity

5. Initializing Workflow steps from blocks: Setting up the individual workflow steps based on the available blocks, 
steps definitions and configuration of execution environment.

Let's take a closer look at each of the workflow compilation steps.

### Workflows blocks loading

As described in the [blocks bundling guide](/workflows/blocks_bundling/), a group of Workflow blocks can be packaged 
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

    The topic of [dynamic Python blocks](/workflows/custom_python_code_blocks) is covered
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
