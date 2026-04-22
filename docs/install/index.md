# Install Inference Server

Choose the installation method that matches your platform:

=== ":fontawesome-brands-docker: Docker"

    The preferred way to use Inference is via Docker
    (see [Why Docker](/understand/architecture.md#why-docker)).
    This works on Linux, macOS, Jetson, and other Docker-capable devices.

    [Install Docker](https://docs.docker.com/engine/install/) (and
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    for GPU acceleration if you have a CUDA-enabled GPU). Then install and run [inference-cli](../inference_helpers/inference_cli.md) :

    ```bash
    pip install inference-cli && inference server start
    ```

    This automatically chooses and configures the optimal container for your machine.

    The `--dev` flag starts a companion Jupyter notebook server with a quickstart guide on
    [`localhost:9002`](http://localhost:9002):

    ```bash
    inference server start --dev
    ```

=== ":fontawesome-brands-windows: Windows (Native App)"

    Download and run the installer to get an Inference Server on Windows — no Docker required.

    <div class="download-container">
        <div class="download-card">
            <a href="/download" class="download-button">
                <img src="/images/windows-icon.svg" alt="Windows" /> Download for Windows
            </a>
        </div>
    </div>

    1. [Download the latest installer](/download) and run it
    2. When the install finishes, it will offer to launch the Inference Server
    3. To stop the server, close the terminal window it opens
    4. To start it again later, find **Roboflow Inference** in your Start Menu

    <p style="font-size: 0.9em;">
        <a href="https://github.com/roboflow/inference/releases">I need a previous release</a>
    </p>

=== ":fontawesome-brands-apple: macOS (Native App)"

    Download the native app to get an Inference Server on macOS — no Docker required.

    <div class="download-container">
        <div class="download-card">
            <a href="/download" class="download-button">
                <img src="/images/macos-icon.svg" alt="macOS" /> Download for Mac
            </a>
        </div>
    </div>

    1. [Download the DMG](/download) and open it
    2. Drag the Roboflow Inference app to your Applications folder
    3. Double-click the app in Applications to start the server

    <p style="font-size: 0.9em;">
        <a href="https://github.com/roboflow/inference/releases">I need a previous release</a>
    </p>


## Device-specific documentation

Special installation notes and performance tips by device are available.
Browse the navigation on the left for detailed install guides:

- :fontawesome-brands-linux: [Linux](linux.md)
- :fontawesome-brands-windows: [Windows](windows.md)
- :fontawesome-brands-apple: [Mac](mac.md)
- :simple-nvidia: [Jetson](jetson.md)
- :fontawesome-brands-raspberry-pi: [Raspberry Pi](raspberry-pi.md)

## Using Your New Server

Once you have [Inference server](../quickstart/docker.md) running,
you can access it via [its API](../api.md) or using
the [Python Inference SDK](../inference_helpers/inference_sdk.md).

=== "Python SDK"
    Install the [Python Inference SDK](../inference_helpers/inference_sdk.md)

    ```bash
    pip install inference-sdk
    ```

    Run [an example model comparison Workflow](https://app.roboflow.com/workflows/embed/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ3b3JrZmxvd0lkIjoiSHhIODdZR0FGUWhaVmtOVWNEeVUiLCJ3b3Jrc3BhY2VJZCI6IlhySm9BRVFCQkFPc2ozMmpYZ0lPIiwidXNlcklkIjoiNXcyMFZ6UU9iVFhqSmhUanE2a2FkOXVicm0zMyIsImlhdCI6MTczNTIzNDA4Mn0.AA78pZnlivFs5pBPVX9cMigFAOIIMZk0dA4gxEF5tj4)
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
            "model1": "rfdetr-small",
            "model2": "rfdetr-medium"
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
                "model1": "rfdetr-small",
                "model2": "rfdetr-medium"
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
            "model1": "rfdetr-small",
            "model2": "rfdetr-medium"
        }
    }'
    ```

    Tip: AI Coding agents are really good at converting snippets like this into other languages.

