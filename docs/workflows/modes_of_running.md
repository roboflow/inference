Workflows can be executed in `local` environment, or `remote` environment can be used. `local` means that model steps
will be executed within the context of process running the code. `remote` will re-direct model steps into remote API
using HTTP requests to send images and get predictions back. 

When `workflows` are used directly, in Python code - `compile_and_execute(...)` and `compile_and_execute_async(...)`
functions accept `step_execution_mode` parameter that controls the execution mode.

Additionally, `max_concurrent_steps` parameter dictates how many steps in parallel can be executed. This will
improve efficiency of `remote` execution (up to the limits of remote API capacity) and can improve `local` execution
if `model_manager` instance is capable of running parallel requests (only using extensions from 
`inference.enterprise.parallel`).

There are environmental variables that controls `workflows` behaviour:
* `DISABLE_WORKFLOW_ENDPOINTS` - disabling workflows endpoints from HTTP API
* `WORKFLOWS_STEP_EXECUTION_MODE` - with values `local` and `remote` allowed to control how `workflows` are executed
in `inference` HTTP container
* `WORKFLOWS_REMOTE_API_TARGET` - with values `hosted` and `self-hosted` allowed - to point API to be used in `remote`
execution mode
* `LOCAL_INFERENCE_API_URL` will be used if `WORKFLOWS_REMOTE_API_TARGET=self-hosted` and 
`WORKFLOWS_STEP_EXECUTION_MODE=remote`
* `WORKFLOWS_MAX_CONCURRENT_STEPS` - max concurrent steps to be allowed by `workflows` executor
* `WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE` - max batch size for requests into remote API made when `remote`
execution mode is chosen
* `WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS` - max concurrent requests to be possible in scope of
single step execution when `remote` execution mode is chosen
