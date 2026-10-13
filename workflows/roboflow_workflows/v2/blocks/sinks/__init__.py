"""Sink blocks: effects outside the workflow, such as files and HTTP requests.

Sinks are never pruned: their steps run even when no output is demanded.
``runtime`` holds the shared outcome helpers and ``BackgroundEffects``, the
host-owned resource behind ``fire_and_forget``.
"""

from roboflow_workflows.v2.blocks.sinks.email import (
    EmailNotificationBlock,
)
from roboflow_workflows.v2.blocks.sinks.event_writer import (
    EventWriterSinkBlock,
)
from roboflow_workflows.v2.blocks.sinks.local_file import (
    LocalFileSinkBlock,
)
from roboflow_workflows.v2.blocks.sinks.s3 import (
    S3SinkBlock,
)
from roboflow_workflows.v2.blocks.sinks.slack import (
    SlackNotificationBlock,
)
from roboflow_workflows.v2.blocks.sinks.twilio import (
    TwilioSmsNotificationBlock,
)
from roboflow_workflows.v2.blocks.sinks.webhook import (
    WebhookSinkBlock,
)

BLOCKS = (
    EmailNotificationBlock,
    LocalFileSinkBlock,
    S3SinkBlock,
    SlackNotificationBlock,
    TwilioSmsNotificationBlock,
    WebhookSinkBlock,
    EventWriterSinkBlock,
)

__all__ = [
    "BLOCKS",
    "EmailNotificationBlock",
    "LocalFileSinkBlock",
    "S3SinkBlock",
    "SlackNotificationBlock",
    "TwilioSmsNotificationBlock",
    "WebhookSinkBlock",
    "EventWriterSinkBlock",
]
