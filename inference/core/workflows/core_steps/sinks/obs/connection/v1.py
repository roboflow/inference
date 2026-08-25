from typing import List, Literal, Optional, Tuple, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.sinks.obs.client import call_with_reconnect
from inference.core.workflows.core_steps.sinks.obs.discovery import (
    discover_password,
    is_local_host,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    INTEGER_KIND,
    OBS_CONNECTION_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    Runtime,
    RuntimeRestriction,
    Severity,
    WorkflowBlock,
    WorkflowBlockManifest,
)

CONNECTION_OUTPUT_KEY: str = "connection"

LONG_DESCRIPTION = """
Connect to a running OBS Studio instance through its built-in websocket server, so that other
blocks can drive scenes, sources, filters and recording in response to model predictions.

## How This Block Works

OBS Studio ships a websocket server (bundled since OBS 28, configured under Tools > WebSocket
Server Settings). This block holds the address and credentials for that server in one place and
emits a connection descriptor that OBS Action blocks consume. The block:

1. Reads the host, port and password of the OBS websocket server
2. When no password is given and OBS runs on this machine, reads the password from OBS
   Studio's own config file, so a local OBS needs no credential configuration at all. The
   block reports where the credential came from in its `message` output
3. Optionally opens a connection immediately and asks OBS for its version, so a misconfigured
   address fails at the top of the Workflow instead of on the first triggered action
4. Emits a `connection` output that every downstream OBS Action block accepts

Connections are pooled per host and port and shared by all OBS Action blocks in the Workflow,
so a Workflow with ten actions still holds a single websocket. If OBS restarts, the next action
transparently reconnects.

## Common Use Cases

- **Live production automation**: drive scene changes from what a model sees on camera
- **Presentation and demo control**: switch scenes or overlays when an object is held up to the lens
- **Privacy in streams and calls**: enable a blur filter the moment a badge or document is detected
- **On-screen analytics**: push live counts into an OBS text source
- **Automated capture**: start and stop recording when activity begins and ends

## Connecting to Other Blocks

- **Before OBS Action blocks**, which require this block's `connection` output
- **After an Environment Secrets Store block**, to supply the websocket password without embedding
  it in the Workflow definition

## Requirements

OBS Studio must be running on a machine reachable from the process executing the Workflow, with its
websocket server enabled (Tools > WebSocket Server Settings). When OBS runs on the same machine,
no password needs to be supplied - the block reads it from the local OBS config. A remote OBS
always requires an explicit password, since this machine's config holds the wrong credential. Because that is nearly always the same machine that runs the Workflow,
this block only works on a self-hosted `inference` server or an Inference Pipeline - Roboflow Hosted
Serverless and Dedicated Deployments cannot reach a local OBS instance. The `obsws-python` package
must be installed in the environment running `inference`.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "OBS Connection",
            "version": "v1",
            "short_description": "Connect to an OBS Studio websocket server.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
            "ui_manifest": {
                "section": "video",
                "icon": "far fa-tower-observation",
                "blockPriority": 2,
                "popular": False,
            },
        }
    )
    type: Literal["roboflow_core/obs_connection@v1"]
    host: Union[str, Selector(kind=[STRING_KIND])] = Field(
        default="127.0.0.1",
        description="Host running the OBS websocket server. Use `127.0.0.1` when OBS runs on the "
        "same machine as `inference`.",
        examples=["127.0.0.1", "$inputs.obs_host"],
    )
    port: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=4455,
        description="Port of the OBS websocket server. OBS defaults to 4455.",
        examples=[4455, "$inputs.obs_port"],
    )
    password: Optional[Union[str, Selector(kind=[SECRET_KIND, STRING_KIND])]] = Field(
        default=None,
        description="Password of the OBS websocket server, found under Tools > WebSocket Server "
        "Settings. Pass it as a Workflow parameter or from a secrets provider rather than "
        "hardcoding it. Leave empty when authentication is disabled.",
        examples=["$inputs.obs_password", "$steps.secrets.obs_password"],
    )
    timeout: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=3,
        description="Seconds to wait for OBS to answer a request before failing.",
        examples=[3],
    )
    discover_password: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="When no password is given and OBS runs on this machine, read the password "
        "from OBS Studio's own config file. Leave enabled so a local OBS needs no configuration; "
        "disable to fail instead of falling back to the local credential.",
        examples=[True],
    )
    verify_connection: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(
        default=True,
        description="Contact OBS when this block runs to confirm the connection works. Disable to "
        "build a Workflow while OBS is not yet running.",
        examples=[True],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name=CONNECTION_OUTPUT_KEY, kind=[OBS_CONNECTION_KIND]),
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
            OutputDefinition(name="obs_version", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        return [
            RuntimeRestriction(
                severity=Severity.HARD,
                note=(
                    "Block requires network access to an OBS Studio instance, which normally "
                    "runs on the same machine as the Workflow. Hosted Serverless and Roboflow "
                    "Dedicated Deployments cannot reach a local OBS websocket server."
                ),
                applies_to_runtimes=[
                    Runtime.HOSTED_SERVERLESS,
                    Runtime.DEDICATED_DEPLOYMENT,
                ],
            )
        ]


class OBSConnectionBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @staticmethod
    def _resolve_password(
        host: str, password: Optional[str], allow_discovery: bool
    ) -> Tuple[Optional[str], str]:
        """Return the password to use plus a one-line note on where it came from.

        The note is surfaced in the block's `message` so a discovered credential is
        never a silent substitution. The password itself is never included.
        """
        if password:
            return password, "password supplied by Workflow"
        if not allow_discovery:
            return password, "no password supplied, discovery disabled"
        if not is_local_host(host):
            return (
                password,
                f"no password supplied; {host} is not local, so this machine's OBS "
                "config was not consulted",
            )
        discovered = discover_password()
        if discovered is None:
            return password, "no password supplied and no local OBS config found"
        if not discovered.auth_required:
            return password, f"OBS at {host} has authentication disabled"
        note = f"password discovered from {discovered.source}"
        if not discovered.server_enabled:
            note += " (warning: websocket server is disabled in that config)"
        return discovered.password, note

    def run(
        self,
        host: str,
        port: int,
        password: Optional[str],
        timeout: int,
        discover_password: bool,
        verify_connection: bool,
    ) -> BlockResult:
        password, credential_note = self._resolve_password(
            host=host, password=password, allow_discovery=discover_password
        )
        connection = {
            "host": host,
            "port": port,
            "password": password,
            "timeout": timeout,
        }
        if not verify_connection:
            return {
                CONNECTION_OUTPUT_KEY: connection,
                "error_status": False,
                "message": f"Connection not verified ({credential_note})",
                "obs_version": "",
            }
        try:
            version = call_with_reconnect(
                host=host,
                port=port,
                password=password,
                timeout=timeout,
                operation=lambda client: client.get_version(),
            )
            obs_version = getattr(version, "obs_version", "") or ""
            return {
                CONNECTION_OUTPUT_KEY: connection,
                "error_status": False,
                "message": (
                    f"Connected to OBS Studio {obs_version} at {host}:{port} "
                    f"({credential_note})"
                ),
                "obs_version": obs_version,
            }
        except Exception as error:  # noqa: BLE001 - surfaced through error_status
            return {
                CONNECTION_OUTPUT_KEY: connection,
                "error_status": True,
                "message": f"Could not connect to OBS at {host}:{port}: {error}",
                "obs_version": "",
            }
