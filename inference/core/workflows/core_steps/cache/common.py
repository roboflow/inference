from inference.core.workflows.prototypes.block import (
    Runtime,
    RuntimeRestriction,
    Severity,
)

# Cache Get / Cache Set keep entries in this process's memory. They have no
# remote code path, so *where model steps execute* (step_execution_mode) is
# irrelevant - what matters is whether one long-lived process sees every
# request of a stream. On stateless / multi-replica HTTP runtimes it does not,
# which is why this restriction is keyed on the runtime only and is not
# narrowed to a step execution mode or an input mode: a still-image request
# degrades the same way a video frame does.
IN_PROCESS_CACHE_HTTP_SOFT_RESTRICTION = RuntimeRestriction(
    severity=Severity.SOFT,
    note=(
        "Cache entries live in this worker's process memory, namespaced by "
        "video_metadata.video_identifier (for still images that falls back to "
        "the input parameter name). On stateless or multi-replica HTTP "
        "runtimes successive requests may be served by different worker "
        "processes, so Cache Get can miss values Cache Set stored, while "
        "requests that do land on the same worker share the namespace. The "
        "cache is only reliable when one long-lived process handles every "
        "frame of a video, e.g. an InferencePipeline or a stateful video "
        "worker."
    ),
    applies_to_runtimes=[Runtime.HOSTED_SERVERLESS, Runtime.DEDICATED_DEPLOYMENT],
)
