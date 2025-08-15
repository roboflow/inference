# Extend Inference

Inference is designed to be the engine behind your computer vision application.

Because Inference runs in Python, you can take the results from your Workflow and integrate them into whatever system you want, even if there is not a direct integration supported in Inference.

In addition, you can extend Inference in two ways:

1. Write code in a [custom Python block](/extend/python/) that runs in a Workflow, and;
2. Make a [custom Workflow block](/workflows/create_workflow_block/).

This section of documentation walks through both how to make a custom Python block and how to create your own reusable block.

## Custom Python Blocks

Custom Python blocks are ideal if you need to run custom code inside your Workflows. They are configured in your Workflows editor, like this:

![](https://blog.roboflow.com/content/images/2025/05/compo2.png)

Custom Python blocks only work on your own hardware, and are unsupported on Roboflow's cloud hosting options.

## Custom Workflow Blocks

Making a custom Workflow block is ideal if you want to create a block native to Workflows. You can optionally contribute your block back to the community by filing a Pull Request to the Inference repository.

Custom Workflow blocks will appear in the "Custom Blocks" tab of your Workflows editor.