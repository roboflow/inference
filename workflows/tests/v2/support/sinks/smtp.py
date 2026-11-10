"""A loopback SMTP server for notification tests; nothing leaves ``127.0.0.1``.

``smtplib`` talks to it exactly as to a real server::

    smtplib.SMTP / SMTP_SSL ──▶ LoopbackSmtpServer (127.0.0.1:<free port>)
        EHLO ─▶ AUTH PLAIN|LOGIN ─▶ MAIL FROM ─▶ RCPT TO… ─▶ DATA ─▶ QUIT
                     │                                         │
                  logins                                    received

It speaks only what ``smtplib.login`` and ``sendmail`` use. ``tls_contexts``
creates a throwaway self-signed certificate, so the block's default implicit
TLS connector can be tested against a private trust root.
"""

import base64
import datetime
import socketserver
import ssl
import tempfile
import threading
from dataclasses import dataclass
from email import message_from_string
from email.message import Message
from pathlib import Path
from typing import Any, List, Optional, Tuple


@dataclass(frozen=True)
class ReceivedMail:
    """One message accepted by ``LoopbackSmtpServer``.

    Args:
        sender: Envelope sender (``MAIL FROM``).
        recipients: Envelope recipients (``RCPT TO``), in order.
        data: Message text after dot-unstuffing.
    """

    sender: str
    recipients: Tuple[str, ...]
    data: str

    def parsed(self) -> Message:
        """Return the message parsed by the standard library."""
        message = message_from_string(self.data)

        return message


class LoopbackSmtpServer:
    """SMTP server on ``127.0.0.1`` with a free port. Use as a context manager.

    Args:
        tls_context: Server TLS context for implicit TLS (SMTPS); ``None``
            serves plain SMTP.
        accept_password: Only this password logs in; ``None`` accepts any.
    """

    def __init__(
        self,
        *,
        tls_context: Optional[ssl.SSLContext] = None,
        accept_password: Optional[str] = None,
    ) -> None:
        self.logins: List[Tuple[str, str]] = []
        self.received: List[ReceivedMail] = []
        self._accept_password = accept_password
        self._lock = threading.Lock()
        owner = self

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

            def get_request(self) -> Any:
                connection, address = super().get_request()
                if tls_context is not None:
                    connection = tls_context.wrap_socket(connection, server_side=True)
                return connection, address

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                owner._serve(self.rfile, self.wfile)

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )

    @property
    def host(self) -> str:
        """Server address."""
        return self._server.server_address[0]

    @property
    def port(self) -> int:
        """Server port."""
        return self._server.server_address[1]

    def __enter__(self) -> "LoopbackSmtpServer":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _serve(self, rfile: Any, wfile: Any) -> None:
        def reply(line: str) -> None:
            wfile.write(f"{line}\r\n".encode("utf-8"))
            wfile.flush()

        def read_line() -> Optional[str]:
            raw = rfile.readline()
            if not raw:
                return None
            return raw.decode("utf-8").rstrip("\r\n")

        reply("220 loopback ESMTP")
        sender = None
        recipients: List[str] = []
        while True:
            line = read_line()
            if line is None:
                return
            verb = line.split(" ", 1)[0].upper()
            argument = line[len(verb) :].strip()
            if verb in ("EHLO", "HELO"):
                reply("250-loopback")
                reply("250 AUTH PLAIN LOGIN")
            elif verb == "AUTH":
                user, password = self._authenticate(argument, reply, read_line)
                if self._accept_password not in (None, password):
                    reply("535 5.7.8 Authentication credentials invalid")
                    continue
                with self._lock:
                    self.logins.append((user, password))
                reply("235 2.7.0 Authentication successful")
            elif verb == "MAIL":
                sender = argument.split(":", 1)[1].strip().strip("<>").split(" ")[0]
                recipients = []
                reply("250 OK")
            elif verb == "RCPT":
                recipients.append(argument.split(":", 1)[1].strip().strip("<>"))
                reply("250 OK")
            elif verb == "DATA":
                reply("354 End data with <CR><LF>.<CR><LF>")
                lines = []
                while (data_line := read_line()) not in (None, "."):
                    lines.append(
                        data_line[1:] if data_line.startswith(".") else data_line
                    )
                with self._lock:
                    self.received.append(
                        ReceivedMail(sender, tuple(recipients), "\n".join(lines))
                    )
                reply("250 OK queued")
            elif verb == "QUIT":
                reply("221 Bye")
                return
            else:
                reply("250 OK")

    @staticmethod
    def _authenticate(argument: str, reply: Any, read_line: Any) -> Tuple[str, str]:
        mechanism, _, initial = argument.partition(" ")
        if mechanism.upper() == "PLAIN":
            if not initial:
                reply("334 ")
                initial = read_line() or ""
            _, user, password = base64.b64decode(initial).decode().split("\0")
            return user, password

        reply("334 VXNlcm5hbWU6")
        user = base64.b64decode(read_line() or "").decode()
        reply("334 UGFzc3dvcmQ6")
        password = base64.b64decode(read_line() or "").decode()

        return user, password


def tls_contexts() -> Tuple[ssl.SSLContext, ssl.SSLContext]:
    """Create a throwaway self-signed certificate for ``127.0.0.1``.

    Requires the ``cryptography`` package; callers skip when it is missing.

    Returns:
        A server context presenting the certificate, and a client context
        that trusts only it and checks the ``127.0.0.1`` address.
    """
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "loopback-smtp")])
    now = datetime.datetime.now(datetime.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(hours=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )

    server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    with tempfile.TemporaryDirectory() as directory:
        certificate_path = Path(directory) / "certificate.pem"
        key_path = Path(directory) / "key.pem"
        certificate_path.write_bytes(certificate_pem)
        key_path.write_bytes(key_pem)
        server_context.load_cert_chain(certificate_path, key_path)
    client_context = ssl.create_default_context(cadata=certificate_pem.decode("ascii"))

    return server_context, client_context
