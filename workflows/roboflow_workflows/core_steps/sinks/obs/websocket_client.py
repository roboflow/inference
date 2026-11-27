"""Minimal client for the obs-websocket 5.x protocol.

Speaks the published obs-websocket JSON protocol over ``websocket-client``:
OBS greets with a Hello, the client answers with an Identify that carries the
optional password proof, then every call is one Request answered by one
RequestResponse. The OBS Workflow blocks only issue one-shot requests, so the
client subscribes to no events and skips any message that is not the response
it is waiting for.

Implemented from the protocol specification:
https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md
"""

import base64
import hashlib
import json
import uuid
from typing import Any, Dict, Optional

import websocket

RPC_VERSION = 1
SUBPROTOCOL = "obswebsocket.json"
NO_EVENT_SUBSCRIPTIONS = 0

OP_HELLO = 0
OP_IDENTIFY = 1
OP_IDENTIFIED = 2
OP_REQUEST = 6
OP_REQUEST_RESPONSE = 7


class OBSRequestError(Exception):
    """OBS received a request and refused it.

    Raised for application-level failures such as an unknown scene or source.
    Reconnecting cannot change the outcome, so callers should not retry.

    Attributes:
        request_type (str): The obs-websocket request that was refused.
        code (Optional[int]): The obs-websocket request status code.
        comment (Optional[str]): The explanation OBS gave, if any.
    """

    def __init__(
        self,
        request_type: str,
        *,
        code: Optional[int],
        comment: Optional[str],
    ):
        self.request_type = request_type
        self.code = code
        self.comment = comment
        # OBS comments end with a full stop; drop it so callers can append context
        details = (comment or "no details").rstrip(".")
        super().__init__(f"OBS rejected {request_type} (code {code}): {details}")


class OBSAuthenticationError(Exception):
    """OBS refused the credentials offered when opening the connection."""


def compute_authentication(password: str, *, salt: str, challenge: str) -> str:
    """Build the password proof obs-websocket expects in the Identify message.

    The proof is ``base64(sha256(secret + challenge))`` where
    ``secret = base64(sha256(password + salt))``, so the password itself never
    crosses the wire.

    Args:
        password (str): The websocket server password configured in OBS.
        salt (str): The salt OBS sent in its Hello message.
        challenge (str): The challenge OBS sent in its Hello message.

    Returns:
        str: The base64-encoded authentication string.
    """
    # obs-websocket fixes this exact SHA-256 scheme and OBS checks it on its side, so a
    # slow password hash would simply be rejected; the proof is never stored
    # codeql[py/weak-sensitive-data-hashing]: OBS protocol auth proof; not stored.
    secret_digest = hashlib.sha256((password + salt).encode("utf-8")).digest()
    secret = base64.b64encode(secret_digest).decode("utf-8")

    proof_digest = hashlib.sha256((secret + challenge).encode("utf-8")).digest()
    authentication = base64.b64encode(proof_digest).decode("utf-8")

    return authentication


class OBSWebSocketClient:
    """A single authenticated connection to an OBS websocket server.

    Not thread-safe: one socket carries every request, so concurrent callers
    must serialise access. The OBS blocks do so with a lock per connection.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        password: Optional[str],
        timeout: float,
    ):
        """Connect to OBS and complete the identification handshake.

        Args:
            host (str): Host name or address of the machine running OBS.
            port (int): Port of the OBS websocket server.
            password (Optional[str]): The websocket password, if OBS requires one.
            timeout (float): Seconds to wait for connecting and for each reply.

        Raises:
            OBSAuthenticationError: If OBS requires a password that is missing
                or rejects the one supplied.
            ConnectionError: If OBS does not follow the expected handshake.
            websocket.WebSocketException: If the connection cannot be opened.
            OSError: If the host cannot be reached.
        """
        self._socket = websocket.create_connection(
            f"ws://{host}:{port}",
            timeout=timeout,
            subprotocols=[SUBPROTOCOL],
        )
        try:
            self._identify(password=password)
        except Exception:
            self.close()
            raise

    def call(
        self,
        request_type: str,
        *,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send one request to OBS and wait for its response.

        Args:
            request_type (str): The obs-websocket request type, for example
                ``"SetCurrentProgramScene"``.
            data (Optional[Dict[str, Any]]): The request's ``requestData`` fields.

        Returns:
            Dict[str, Any]: The response's ``responseData``, empty when OBS
            returns none.

        Raises:
            OBSRequestError: If OBS refuses the request.
            websocket.WebSocketException: If the connection fails or times out.
        """
        request_id = uuid.uuid4().hex
        request = {"requestType": request_type, "requestId": request_id}
        if data:
            request["requestData"] = data

        self._send(op=OP_REQUEST, data=request)

        while True:
            message = self._receive()
            if message.get("op") != OP_REQUEST_RESPONSE:
                continue

            response = message.get("d", {})
            if response.get("requestId") != request_id:
                continue

            status = response.get("requestStatus", {})
            if not status.get("result"):
                raise OBSRequestError(
                    request_type,
                    code=status.get("code"),
                    comment=status.get("comment"),
                )

            response_data = response.get("responseData") or {}

            return response_data

    def close(self) -> None:
        """Close the connection. Safe to call on an already closed connection."""
        self._socket.close()

    def _identify(self, *, password: Optional[str]) -> None:
        hello = self._receive()
        if hello.get("op") != OP_HELLO:
            raise ConnectionError(
                f"Expected a Hello message from OBS, got op {hello.get('op')}"
            )

        identify = {
            "rpcVersion": RPC_VERSION,
            "eventSubscriptions": NO_EVENT_SUBSCRIPTIONS,
        }
        authentication = hello.get("d", {}).get("authentication")
        if authentication:
            if not password:
                raise OBSAuthenticationError(
                    "OBS requires a websocket password, but none was supplied"
                )

            identify["authentication"] = compute_authentication(
                password,
                salt=authentication["salt"],
                challenge=authentication["challenge"],
            )

        self._send(op=OP_IDENTIFY, data=identify)

        try:
            identified = self._receive()
        except websocket.WebSocketConnectionClosedException as error:
            # OBS drops the socket instead of answering when the proof is wrong
            raise OBSAuthenticationError(
                "OBS closed the connection during identification; the websocket "
                "password is most likely wrong"
            ) from error

        if identified.get("op") != OP_IDENTIFIED:
            raise ConnectionError(
                f"Expected an Identified message, got op {identified.get('op')}"
            )

    def _send(self, *, op: int, data: Dict[str, Any]) -> None:
        self._socket.send(json.dumps({"op": op, "d": data}))

    def _receive(self) -> Dict[str, Any]:
        raw = self._socket.recv()
        if not raw:
            # websocket-client returns an empty payload for a close frame
            raise websocket.WebSocketConnectionClosedException(
                "OBS closed the websocket connection"
            )

        message = json.loads(raw)

        return message
