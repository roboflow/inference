## Using Your New Server

Once you hav a server running, you can access it via [its API](/api.md) or using
[the Python SDK](/inference_helpers/inference_sdk.md). You can also use it to build Workflows
using [the Roboflow Platform UI](https://docs.roboflow.com/workflows/create-a-workflow).

=== "Python SDK"
    ### Install the SDK

    ```bash
    pip install inference-sdk
    ```

    ### Run a workflow

    This code runs [an example model comparison Workflow](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSHhIODdZR0FGUWhaVmtOVWNEeVUiLCJ3b3Jrc3BhY2VJZCI6IlhySm9BRVFCQkFPc2ozMmpYZ0lPIiwidXNlcklkIjoiNXcyMFZ6UU9iVFhqSmhUanE2a2FkOXVicm0zMyIsImlhdCI6MTczNTIzNDA4Mn0.AA78pZnlivFs5pBPVX9cMigFAOIIMZk0dA4gxEF5tj4)
    on an Inference Server running on your local machine:

    ```python
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="http://localhost:9001", # use local inference server
        # api_key="<YOUR API KEY>" # optional to access your private data and models
    )

    result = client.run_workflow(
        workspace_name="roboflow-docs",
        workflow_id="model-comparison",
        images={
            "image": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
        },
        parameters={
            "model1": "yolov8n-640",
            "model2": "yolov11n-640"
        }
    )

    print(result)
    ```

=== "Node.js"
    From a JavaScript app, hit your new server with an HTTP request.

    ```js
    const response = await fetch('http://localhost:9001/infer/workflows/roboflow-docs/model-comparison', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            // api_key: "<YOUR API KEY>" // optional to access your private data and models
            inputs: {
                "image": {
                    "type": "url",
                    "value": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
                },
                "model1": "yolov8n-640",
                "model2": "yolov11n-640"
            }
        })
    });

    const result = await response.json();
    console.log(result);
    ```

    !!! Warning
        Be careful not to expose your API Key to external users
        (in other words: don't use this snippet in a public-facing front-end app).

=== "HTTP / cURL"
    Using the server's API you can access it from any other client application.
    From the command line using cURL:

    ```bash
    curl -X POST "http://localhost:9001/infer/workflows/roboflow-docs/model-comparison" \
    -H "Content-Type: application/json" \
    -d '{
        "api_key": "<YOUR API KEY -- REMOVE THIS LINE IF NOT FILLING>",
        "inputs": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
            },
            "model1": "yolov8n-640",
            "model2": "yolov11n-640"
        }
    }'
    ```

    !!! Tip
        ChatGPT is really good at converting snippets like this into other languages.
        If you need help, try pasting it in and asking it to translate it to your
        language of choice.