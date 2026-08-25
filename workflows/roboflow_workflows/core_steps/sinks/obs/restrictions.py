"""Runtime restrictions shared by the OBS Workflow blocks."""

from roboflow_workflows.execution_engine.entities.workload import RuntimeRestriction
from roboflow_workflows.prototypes.block import Runtime, Severity

# OBS requests are sent from the process running the Workflow, whatever the step
# execution mode, so only network reachability of OBS restricts these blocks. Same
# code and axes as the ONVIF camera and PLC restrictions, with an OBS-specific note.
OBS_LAN_ACCESS_RESTRICTION = RuntimeRestriction(
    code="requires_lan_access_to_device",
    severity=Severity.HARD,
    note=(
        "Block sends requests to an OBS Studio websocket server, which normally runs "
        "on the same machine as the Workflow. Hosted Serverless and Roboflow "
        "Dedicated Deployments cannot reach it."
    ),
    applies_to_runtimes=[
        Runtime.HOSTED_SERVERLESS,
        Runtime.DEDICATED_DEPLOYMENT,
    ],
)
