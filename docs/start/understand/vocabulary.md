# Glossary

Inference uses some terms (and terms of art) that may be unfamiliar to some
readers. This page aims to disambiguate and clarify their meaning, specifically
in the context of Roboflow Workflows Blocks.

## API
An interface that allows software applications to communicate with each other.
In Roboflow, the API provides access to workflows and blocks, enabling users to
configure and execute inference pipelines programmatically.

## Block
The fundamental unit of a Roboflow Workflow. Blocks perform specific tasks, such
running model inference, performing logic, or interfacing with external services.

## CLI
Command-Line Interface. A tool used to interact with Inference, such
as starting the server, running benchmarks, or executing Workflows.

## Client
A software application or library that interacts with the API. Our Python
SDK is a client that abstracts the REST API into a more user-friendly
interface.

## Commercial License
A license permitting businesses to use models under terms suited for commercial
applications, typically involving subscription plans or usage fees. While
Inference uses a permissive Apache 2.0 license, some models have terms that
may require downstream users to open source their own codebase. A commercial
license removes that restriction.

## Dedicated
A configuration where cloud compute resources for running Inference are allocated
exclusively to a single user or organization, ensuring optimal performance and
enabling additional functionality. Billed based on running time & usage.

## Definition
The structured JSON description of a Workflow, including its sequence of
Blocks, input parameters, response format, and any custom logic.

## Dynamic Block
A type of Workflows Block using custom Python Code included in the JSON
Definition of the Workflow allowing advanced customization at runtime.

## Enterprise
A service tier for large organizations, offering enhanced capabilities,
scalability, and support. Source code for enterprise functionality is
included in the `enterprise` folder of the repo but may only be used in
conjunction with an active Enterprise license.

## Execution Engine
The backend system responsible for executing Workflows, managing the execution
of Blocks, and optimizing resource allocation.

## Fine-Tuned Model
A model that has been trained on a specific dataset for improved performance
in a targeted application. For example, a "Scratch Detection" model tuned to
find scratches on a specific automotive component.

## Foundation Model
A general-purpose model that knows a lot about a lot and does not necessarily
need to be fine-tuned on a specific dataset and can be used "zero-shot".

## Kind
A categorization of data types used in Workflows. Defining the input and output
Kinds of a Workflow Block allows the Execution Engine to validate, serialize,
and optimize data and connections. Example Kinds are `detection` (representing
a prediction from an object detection model), `image` (containing pixels and
their metadata), and `float_zero_to_one`.

## Inference Pipeline
An asynchronous interface for video streaming that handles efficiently
consuming and routing video frames from a camera source and through a
Workflow while maintaining state.

## LMM
Large Multimodal Model. A Block type that processes multiple data types,
such as images and text. Examples are Florence-2 and OpenAI's GPT-4o.

## Managed
A deployment running in Roboflow's Cloud environment where the scaling and
infrastructure are provided as a service. Contrast with "Self-Hosted" where
the customer installs the software on their own infrastructure and is
responsible for its setup and maintenance.

## Metered
A billing approach based on usage metrics, such as the number of Workflow
executions or hours of video processed. Models and Workflows that require
an API Key to access Roboflow's Cloud Services are metered.

## Model
A trained machine learning artifact used for inference tasks such as object
detection or classification. Consists of an architecture (like ResNet-32)
and trained weights.

## Parameter
An input to a Workflow. Can be an image or data like strings, numbers,
arrays, or objects. Used as inputs to Blocks.

## Platform
The Roboflow ecosystem, which includes end-to-end tools for collecting and
organizing data, annotating images, creating datasets, training models,
building Workflows, and monitoring deployments.

## Pre-Trained Model
A model architecture (like YOLOv11) loaded with weights that have been trained
on a generic dataset (like Microsoft COCO). Contrast with a Fine-Tuned model
that has been trained on a domain-specific dataset.

## Public
A dataset, model, or Workflow that is accessible to all users within the
Roboflow ecosystem. These can be found and distributed on Roboflow Universe.

## Schema
The structured format of input and output data for a Workflow, defining the
properties and types a downstream application needs to pass and parse to
integrate with the Workflow.

## SDK
Software Development Kit. A set of tools and libraries provided for integrating
with the API using user-friendly abstractions.

## Server
A device running Inference's HTTP interface, usually through Docker on port 9001.

## Serverless
An execution model for Inference where resources scale on demand, without
requiring manual resource management. Contrast with Dedicated. Billed solely
based on usage.

## Traditional CV
Blocks in Workflows that implement traditional computer vision techniques, such as
filtering or edge detection, without relying on machine learning models.

## Universe
A large collection of datasets, pre-trained models, and other resources shared
publicly by other users and available for use within the Roboflow ecosystem.

## Weights
The learned parameters of a machine learning model determining its predictions
for a given input.

## Workflow
A series of interconnected Blocks designed to process data and produce desired
outcomes through a defined sequence of operations. Can be used for chaining models,
maintaining state, performing custom logic, and integrating with external systems.

## Workspace
A collaborative environment in Roboflow where users can create datasets, train models,
and build Workflows. A Workspace is the container for data and arbiter of access
via seats and API Keys.
