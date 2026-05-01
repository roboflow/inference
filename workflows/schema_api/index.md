# Workflows Schema API

The Workflows Schema API provides developers with a clear, programmatic understanding of a workflow's structure, inputs, and outputs. It addresses the challenge of programmatically determining a workflow's input requirements and output types.

## Purpose and Benefits

The API offers structured access to:

1. **Input Parameters**: Required workflow inputs.
2. **Output Structure**: Details of the returned data.
3. **Type Hints**: Expected data types for inputs and outputs.
4. **Schemas of Kinds**: Detailed schemas for complex data types.

This enables developers to:

- Validate inputs programmatically
- Understand output data structures
- Integrate workflows into larger systems
- Generate documentation or UIs based on workflow requirements

## Using the API
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/inference/blob/main/docs/notebooks/workflow_schema_api.ipynb)

```python
import requests

WORKSPACE_NAME = "workspace-name"
WORKFLOW_ID = "workflow-id"
INFERENCE_SERVER_URL = "https://detect.roboflow.com"

WORKFLOW_SCHEMA_ENDPOINT = f"{INFERENCE_SERVER_URL}/{WORKSPACE_NAME}/workflows/{WORKFLOW_ID}/describe_interface"
ROBOFLOW_API_KEY = "Your Roboflow API Key"

headers = {
    "Content-Type": "application/json",
}

data = {
    "api_key": ROBOFLOW_API_KEY,
}

res = requests.post(WORKFLOW_SCHEMA_ENDPOINT, headers=headers, json=data)

schema = res.json()

inputs = schema["inputs"]
outputs = schema["outputs"]
kinds_schemas = schema["kinds_schemas"]
typing_hints = schema["typing_hints"]
```

## Inputs and Outputs

The `inputs` and `outputs` keys show all of the inputs and outputs the workflow expects to run and return.

## Typing Hints

The `typing_hints` key shows the data types of the inputs and outputs.

## Kinds Schemas

The `kinds_schemas` key returns an OpenAPI specification with more detailed information about the data 
types being returned and how to parse them. For example, the `object_detection_prediction` contains 
information about the nested data that will be present.