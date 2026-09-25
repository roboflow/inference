This directory is intended to run with pytest. The tests will run automatically via github actions.

To run files in this directory, the scripts need access to Roboflow API keys via env variables with the naming convention <PROJECT SLUG>_API_KEY=<ROBOFLOW API KEY>. For example, for a test with the project `asl-poly-instance-seg` to run succesfully, there needs to be an environment variable `asl_poly_instance_seg_API_KEY=<API KEY>`. If adding a test, be sure to add this environment variable to the github actions secrets via the github console and the file `.github/workflows/test.yml`.

The tests run using `tests.json`.  To add to this file, add an entry in the root list with all keys except for the `expected_response` key.  To add this key, run the `populate_expected_responses.py` script.

## Workflow workload introspection

`test_workflow_workload_endpoints.py` sends real HTTP requests to
`POST /workflows/describe_workload` and `GET /model/registry` on a running
server. It sends inline definitions only. It sends no image, runs no workflow,
loads no model and evaluates no custom Python.

Start a CPU server from the repository root, with this checkout mounted over
the installed packages:

```bash
docker run -p 9001:9001 \
    -v ./inference:/app/inference \
    -v ./inference_models/inference_models:/usr/local/lib/python3.11/site-packages/inference_models/ \
    -v ./workflows/roboflow_workflows:/usr/local/lib/python3.11/site-packages/roboflow_workflows/ \
    roboflow/roboflow-inference-server-cpu
```

The `site-packages` paths must match the Python version of the image.

Run the module from the repository root, in a second shell:

```bash
python -m pytest tests/inference/integration_tests/test_workflow_workload_endpoints.py
```

Environment:

- `PORT` (default `9001`), `BASE_URL` (default `http://localhost`) and
  `MAX_WAIT` (seconds, default `30`) select and wait for the server.
- `API_KEY` is optional. The route requires a key in the body or in the
  `Authorization: Bearer` header, but it does not need a valid one for
  structural answers. Without `API_KEY` the tests send a placeholder key.
- Keep the server defaults: `DISABLE_WORKFLOW_ENDPOINTS` and
  `DISABLE_WORKFLOW_WORKLOAD_ENDPOINTS` unset or `False`, and
  `GET_MODEL_REGISTRY_ENABLED=True`. A disabled route fails the tests; they do
  not skip.

Model metadata varies by server. `metadata_status` is `available`, `disabled`
or `unavailable`, depending on `USE_INFERENCE_MODELS`, `OFFLINE_MODE`, the key
and platform reachability. The tests accept any status and only check that
`metadata` agrees with it. They assert the graph, dimensionality, declared
model ids and completeness instead.
