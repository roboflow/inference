# Extend Inference

Inference is designed to be the engine behind your computer vision application.

Because Inference runs in Python, you can take the results from your Workflow and integrate them into whatever system you want, even if there is not a direct integration supported in Inference.

In addition, you can extend Inference in two ways:

1. Create a custom Python block that runs in a Workflow, and;
2. Make a reusable Workflow block.

Custom Python blocks are ideal if you need to run custom code inside your Workflows.

Custom Python blocks only work on your own hardware, and are unsupported on Roboflow's cloud hosting options.

Making a reusable Workflow block is ideal if you want to create a block native to Workflows. You can optionally contribute your block back to the community by filing a Pull Request to the Inference repository.

This section of documentation walks through both how to make a custom Python block and how to create your own reusable block.