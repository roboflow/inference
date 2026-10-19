# Opt-in in-process model ownership

`ManagedModels` connects the optional `inference-model-manager` package to V2
model blocks. The existing `ModelCache` and supplied-model path remain available
and remain the default. Installing Workflows does not install the manager.
Configure the manager and its model runtime dependencies separately on the host.

The host owns one manager for one credential policy, device, and lifetime. CUDA
devices must include an index such as `cuda:0`; thread-relative `cuda` is refused. Loads
use the direct backend, pinned models, and the exact `ModelRequest` identity.
Identical concurrent requests share one load. Different package IDs stay separate.
A request without a package ID permits normal package selection; use an explicit
package ID for reproducible deployments and comparisons.

```python
from roboflow_workflows.v2.blocks.models.cache import ModelRequest
from roboflow_workflows.v2.blocks.models.managed import ManagedModels

# `plan` is a compiled V2 workflow whose model step requests this same pair.
models = ManagedModels(api_key=host_api_key, device="cuda:0", namespace="tenant-a")
try:
    native_model = models.preload(ModelRequest(model_id, model_package_id))
    session = models.create_session(plan, resources=other_host_resources)
    result = session.run(workflow_inputs)
    consume_and_finish_outputs(result)
finally:
    models.close()
```

`other_host_resources` must not contain `models` or a scoped `*.models` override.
The returned session is the native V2 session. Its model resource returns the raw
native model, so blocks retain their existing `pre_process`, `forward`, and
`post_process` calls, streams, readiness events, and prediction representations.
The manager's `process`, `process_async`, and `submit` APIs are not used. Preloading
is optional but lets the host finish loading before a measurement or live run.
There is no automatic warmup; the host performs its normal warmup through the
same execution path it intends to use.

For an active session, first stop the producer and wait for the active run. If
cancellation is necessary, cancel and wait for its completion. Finish passive
pipelines and direct calls, then finish any external consumers of returned GPU
outputs before calling `models.close()`. Session closure does not prove external
GPU consumers are finished. The owner deliberately does not unload a model after
each phase or call, and it does not add inference serialization or stream handoffs.

`close()` refuses unfinished sessions, waits for admitted session construction and
loads, then shuts down its manager. A timeout or busy-session refusal leaves the
models available and allows a later close retry. Earlier sessions in that close
attempt may already be closed; recreate those sessions if more work is needed.
Concurrent close calls are refused. If manager shutdown itself fails, new work
stays disabled; retain the owner and retry close after resolving the failure.
`timeout` bounds waits for construction and loads, not the implementations of
native session cleanup or manager shutdown. The owner cannot cancel a blocked
native model constructor.

Keep the owner alive for every session it creates. Do not use a model returned by
`preload` after closing the owner; retaining that reference may also retain model
memory. Directly invoking that model is outside the owner's session tracking.
There is no shared-manager mode, general eviction support, or memory admission
controller. Pinned models may exceed the manager's capacity setting. Bound the
set of requests in the host, and use separate owners for different devices or
credential policies. A namespace distinguishes ownership; it is not an access
control mechanism. Do not put credentials in workflow JSON.

The supplied-model `ModelCache` remains useful for tests and controlled A/B
comparisons. Opting in does not claim performance equivalence: qualify the exact
package, native outputs, readiness behavior, and representative throughput before
changing a deployment's host configuration.
