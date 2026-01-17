# What is Roboflow Workflows?

Roboflow Workflows is an ecosystem that enables users to create machine learning applications using a wide range
of pluggable and reusable blocks. These blocks are organized in a way that makes it easy for users to design
and connect different components. The graphical interface allows you to visually construct workflows
without needing extensive technical expertise. Once the workflow is designed, the Workflows engine runs the
application, ensuring all the components work together seamlessly. This provides a rapid transition
from prototype to production-ready solutions, allowing you to quickly iterate and deploy applications.

Roboflow offers a growing selection of Workflow blocks, and the community can also create new blocks. This ensures
that the ecosystem continuously expands and evolves. Moreover, Roboflow provides flexible deployment options,
including on-premises and cloud-based solutions, allowing users to deploy their applications in the environment 
that best suits their needs.

With Workflows, you can:

- Detect, classify, and segment objects in images using state-of-the-art models.

- Use Large Multimodal Models (LMMs) to make determinations at any stage in a workflow.

- Introduce elements of business logic to translate model predictions into your domain language

<div class="button-holder">
<a href="/workflows/blocks/" class="button half-button">Explore all Workflow blocks</a>
<a href="https://app.roboflow.com/workflows" class="button half-button">Begin building with Workflows</a>
</div>

![A license plate detection workflow implemented in Workflows](https://media.roboflow.com/inference/workflow-example.png)


In this section of the documentation, we'll walk through everything you need to know to create and run workflows. Let’s get started!

Next, [create and run a workflow](./create_and_run.md) or
[browse example Workflows](/workflows/gallery/index.md).

<style>
.button-holder {
  margin-bottom: 1.5rem;
}

.button {
  background-color: var(--md-primary-fg-color);
  display: flex;
  padding: 10px;
  color: white !important;
  border-radius: 5px;
  text-align: center;
  align-items: center;
  justify-content: center;
}
</style>
