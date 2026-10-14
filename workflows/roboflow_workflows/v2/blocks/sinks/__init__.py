"""Sink blocks: effects outside the workflow, such as files and HTTP requests.

Sinks are never pruned: their steps run even when no output is demanded.
``runtime`` holds the shared outcome helpers and ``BackgroundEffects``, the
host-owned resource behind ``fire_and_forget``.
"""

from roboflow_workflows.v2.blocks.sinks.email_notification.v2 import EmailNotificationBlock

from roboflow_workflows.v2.blocks.sinks.local_file_sink.v1 import LocalFileSinkBlock
from roboflow_workflows.v2.blocks.sinks.s3_sink.v1 import S3SinkBlock
from roboflow_workflows.v2.blocks.sinks.slack_notification.v1 import SlackNotificationBlock
from roboflow_workflows.v2.blocks.sinks.twilio_sms_notification.v2 import TwilioSmsNotificationBlock
from roboflow_workflows.v2.blocks.sinks.webhook_sink.v1 import WebhookSinkBlock

BLOCKS = (
    EmailNotificationBlock,
    LocalFileSinkBlock,
    S3SinkBlock,
    SlackNotificationBlock,
    TwilioSmsNotificationBlock,
    WebhookSinkBlock,
)

__all__ = [
    "BLOCKS",
    "EmailNotificationBlock",
    "LocalFileSinkBlock",
    "S3SinkBlock",
    "SlackNotificationBlock",
    "TwilioSmsNotificationBlock",
    "WebhookSinkBlock",
]
