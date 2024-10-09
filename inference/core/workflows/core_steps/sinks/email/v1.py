import smtplib
import ssl
from copy import copy
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    STRING_KIND,
    StepOutputSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Email Sink",
            "version": "v1",
            "short_description": "",
            "long_description": "",
            "license": "Apache-2.0",
            "block_type": "sink",
        }
    )
    type: Literal["roboflow_core/email_sink@v1"]
    subject: str = Field(description="Subject of the message")
    message: str = Field(
        description="Content of the message to send",
    )
    smtp_server: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="SMTP server to use",
    )
    sender_email: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="E-mail to be used to send the message",
    )
    sender_email_password: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = (
        Field(
            description="Sender e-mail password to use SMTP server",
            private=True,
        )
    )
    receiver_email: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = Field(
        description="Destination e-mail address",
    )
    message_parameters: Dict[
        str,
        Union[WorkflowParameterSelector(), StepOutputSelector()],
    ] = Field(
        description="References data to be used to construct each and every column",
        examples=[
            {
                "predictions": "$steps.model.predictions",
                "reference": "$inputs.reference_class_names",
            }
        ],
        default_factory=dict,
    )
    message_parameters_operations: Dict[str, List[AllOperationsType]] = Field(
        description="UQL definitions of operations to be performed on defined data w.r.t. each message parameter",
        examples=[
            {
                "predictions": [
                    {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
                ]
            }
        ],
        default_factory=dict,
    )
    attachments: Dict[str, StepOutputSelector(kind=[STRING_KIND])] = Field(
        description="Attachments",
        default_factory=dict,
    )
    smtp_port: int = Field(
        default=465,
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class EmailBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        subject: str,
        message: str,
        smtp_server: str,
        sender_email: str,
        sender_email_password: str,
        receiver_email: str,
        message_parameters: Dict[str, Any],
        message_parameters_operations: Dict[str, List[AllOperationsType]],
        attachments: Dict[str, str],
        smtp_port: int,
    ) -> BlockResult:
        message_parameters = copy(message_parameters)
        for variable_name, operations in message_parameters_operations.items():
            operations_chain = build_operations_chain(operations=operations)
            message_parameters[variable_name] = operations_chain(
                message_parameters[variable_name], global_parameters={}
            )
        for variable_name, value in message_parameters.items():
            message = message.replace(f"`$parameters.{variable_name}`", str(value))
        try:
            e_mail_message = MIMEMultipart()
            e_mail_message["From"] = sender_email
            e_mail_message["To"] = receiver_email
            e_mail_message["Subject"] = subject
            e_mail_message.attach(MIMEText(message, "plain"))
            for attachment_name, attachment_content in attachments.items():
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment_content.encode("utf-8"))
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename= {attachment_name}",
                )
                e_mail_message.attach(part)
            to_sent = e_mail_message.as_string()
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
                server.login(sender_email, sender_email_password)
                server.sendmail(sender_email, receiver_email, to_sent)
            return {"error_status": False, "message": "Message sent successfully"}
        except Exception as error:
            return {"error_status": True, "message": str(error)}
