# The pillars of Workflows

In the Roboflow Workflows ecosystem, various components work together seamlessly to power your applications. 
Some of these elements will be part of your daily interactions, while others operate behind the scenes to ensure 
smooth and efficient application performance. The core pillars of the ecosystem include:

- **Workflows UI:** The intuitive interface where you design and manage your workflows.

- **Workflow Definition:** An interchangeable format that serves as a program written in the "workflows" language, 
defining how your workflows operate.

- **Workflows Blocks:** Modular components that perform specific tasks within your workflows, organised in plugins 
which can be easily created and injected into ecosystem.

- **Workflows Compiler and Execution Engine:** The systems that compile your workflow definitions and execute them, 
ensuring everything runs smoothly in your environment of choice.


We will explore each of these components, providing you with a foundational understanding to help you navigate and 
utilize the full potential of Roboflow Workflows effectively.

## Workflows UI

Traditionally, building machine learning applications involves complex coding and deep technical expertise. 
Roboflow Workflows simplifies this process in two key ways: providing pre-built blocks (which will be described later), 
and delivering user-friendly GUI. 

The interface allows you to design applications without needing to write code. You can easily connect components 
together and achieve your goals without a deep understanding of Python or the underlying workflow language.

Thanks to UI, creating powerful machine learning solutions is straightforward and accessible, allowing you to focus 
on innovation rather than intricate programming.

While not essential, the UI is a highly valuable component of the Roboflow Workflows ecosystem. At the end of the 
workflow creation process it creates workflow definition required for Compiler and Execution engine to run the workflow.


<img src="https://media.roboflow.com/inference/add-output.gif" alt="UI preview" width="100%"/>




## Workflow definition

A workflow definition is essentially a document written in the internal programming language of Roboflow Workflows. 
It allows you to separate the design of your workflow from its execution. You can create a workflow definition once 
and run it in various environments using the Workflows Compiler and Execution Engine.

You have two options for creating a workflow definition: UI to design it visually or write it from scratch 
if you’re comfortable with the workflows language. More details on writing definitions manually 
are available [here](todo). For now, it's important to grasp the role of the definition within the ecosystem.

A workflow definition outlines:

- **Inputs:** These are either images or configuration parameters that influence how the workflow operates. 
Instead of hardcoding values, inputs are placeholders that will be replaced with actual data during execution.

- **Steps:** These are instances of workflow blocks. Each step takes inputs from either the workflow inputs or the 
outputs of previous steps. The sequence and connections between steps determine the execution order.

- **Outputs:** specify field names in execution result and reference step outputs. During runtime, referred values
are dynamically provide based on results of workflow execution.
