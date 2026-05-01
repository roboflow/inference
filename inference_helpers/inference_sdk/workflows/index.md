# Workflows

Start a local inference server with `inference server start`, then run a workflow:

```python
from inference_sdk import InferenceHTTPClient

# Replace ROBOFLOW_API_KEY with your Roboflow API Key
CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="ROBOFLOW_API_KEY"
)

CLIENT.run_workflow(
    specification={
        "version": "1.0",
        "inputs": [
            {"type": "InferenceImage", "name": "image"},
            {"type": "InferenceParameter", "name": "my_param"},
        ],
        # ...
    },
    # OR
    # workspace_name="my_workspace_name",
    # workflow_id="my_workflow_id",

    images={
        "image": "url or your np.array",
    },
    parameters={
        "my_param": 37,
    },
)
```

Please note that either `specification` is provided with specification of workflow as described
[here](../../workflows/definitions.md) or
both `workspace_name` and `workflow_id` are given to use workflow predefined in Roboflow app. `workspace_name`
can be found in Roboflow APP URL once browser shows the main panel of workspace.

**Server-side caching of Workflow definitions:** When you use `run_workflow(...)` with `workspace_name` and `workflow_id`, the server will cache the definition for 15 minutes. If you change the definition in Workflows UI and re-run the method, you may not see the change. To force processing without cache, pass `use_cache=False` as a parameter of `run_workflow(...)`.

**Workflows profiling:** You may request a profiler trace by passing `enable_profiling=True` to `run_workflow(...)`. If server configuration enables traces exposure, you will find a JSON file with the trace in the directory specified by `profiling_directory` parameter of `InferenceConfiguration` (default: `inference_profiling` in your current working directory). The traces can be loaded and rendered in Google Chrome at `chrome://tracing`.
