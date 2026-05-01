# Rules dictating which blocks can be connected

A natural question you might ask is: *How do I know which blocks to connect to achieve my desired outcome?* 
This is a crucial question, which is why we've created auto-generated 
[documentation for all supported Workflow blocks](/workflows/blocks/index.md). In this guide, we’ll show you how to use 
these docs effectively and explain key details that help you understand why certain connections between 
blocks are possible, while others may not be.

!!! Note 
    
    Using the Workflows UI in the Roboflow app, you may find compatible connections between steps automatically,
    without any input from you. This page briefly explains how to determine whether two
    blocks can be connected, making it possible to connect steps manually if needed. This page appears
    before the link to the [block gallery](/workflows/blocks/index.md) because it explains how to use those docs effectively.
    It also introduces references to concepts covered in the User and Developer Guide. Please continue reading those sections
    if you find some concepts presented here need further explanation.


## Navigating the blocks documentation

When you open the blocks documentation, you’ll see a list of all blocks supported by Roboflow. Each block entry 
includes a name, brief description, category, and license for the block. You can click on any block to see more 
detailed information.

On the block details page, you’ll find documentation for all supported versions of that block, 
starting with the latest version. For each version, you’ll see:

- detailed description of the block

- type identifier, which is required in Workflow definitions to identify the specific block used for a step

- table of configuration properties, listing the fields that can be specified in a Workflow definition, 
including their types, descriptions, and whether they can accept a dynamic selector or just a fixed value.

- Available connections, showing which blocks can provide inputs to this block and which can use its outputs.

- A list of input and output bindings:
  
  - input bindings are the names of step definition properties that can hold selectors, along with the type 
  (or `kind`) of data they pass.

  - output bindings are names and kinds for block outputs that can be used as inputs by steps defined in 
  the Workflow definition

- An example of a Workflow step based on the documented block.

The `kind` mentioned above refers to the type of data flowing through the connection during execution, 
and this is further explained in the developer guide.

## What makes connections valid?

Each block provides a manifest that lists the fields to be included in the Workflow definition when creating a step.
The values of these fields in a Workflow definition may contain:

- References ([selectors](/workflows/definitions.md)) to data the block will process, such as step outputs or 
[batch-oriented workflow inputs](/workflows/workflow_execution.md)

- Configuration values: Specific settings for the step or references ([selectors](/workflows/definitions.md)) that 
provide configuration parameters dynamically during execution.

The manifest also includes the block's outputs.

For each step definition field (if it can hold a [selector](/workflows/definitions.md)) and step output, 
the expected [kind](/workflows/kinds/index.md) is specified. A [kind](/workflows/kinds/index.md) is a high-level definition 
of the type of data that will be passed during workflow execution. Simply put, it describes the data that 
will replace the [selector](/workflows/definitions.md) during block execution.

To ensure steps are correctly connected, the Workflow Compiler checks if the input and output [kinds](/workflows/kinds/index.md)
match. If they do, the connection is valid.

Additionally, the [`dimensionality level`](/workflows/workflow_execution.md#dimensionality-level) of the data is considered when 
validating connections. This ensures that data from multiple sources is compatible across the entire Workflow, 
not just between two connected steps. More details on dimensionality levels can be found in the 
[user guide describing workflow execution](/workflows/workflow_execution.md).
