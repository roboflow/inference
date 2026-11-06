"""Unit tests for the obs-websocket 5.x protocol client used by the OBS blocks.

A scripted socket plays OBS's side of the conversation, so these tests cover the
handshake, password proof, request/response matching and error mapping without
a running OBS.
"""

import json

import pytest
import websocket
from roboflow_workflows.core_steps.sinks.obs import websocket_client
from roboflow_workflows.core_steps.sinks.obs.websocket_client import (
    OBSAuthenticationError,
    OBSRequestError,
    OBSWebSocketClient,
    compute_authentication,
)

# Captured from an independent obs-websocket client talking to a mock server with
# these exact inputs, so the proof is checked against another implementation
# rather than against itself.
PASSWORD = "supersecretpassword"
SALT = "PZVbYpvAnZut2SS6JNJytDm9"
CHALLENGE = "ztTBnnuqrqaKDzRM3xcVdbYm"
EXPECTED_AUTHENTICATION = "zZgWipvwSGrw748kHN4gNpBC1IaeiiWX3Hjkrm849Sc="


def _message(op, data):
    return json.dumps({"op": op, "d": data})


def _hello(auth_required=False):
    data = {"obsWebSocketVersion": "5.7.4", "rpcVersion": 1}
    if auth_required:
        data["authentication"] = {"salt": SALT, "challenge": CHALLENGE}
    return _message(0, data)


def _response(request, *, result=True, code=100, comment=None, response_data=None):
    status = {"result": result, "code": code}
    if comment:
        status["comment"] = comment
    data = {
        "requestType": request["requestType"],
        "requestId": request["requestId"],
        "requestStatus": status,
    }
    if response_data is not None:
        data["responseData"] = response_data
    return _message(7, data)


class ScriptedSocket:
    """Plays OBS's side of an obs-websocket conversation.

    `on_identify` returns the frames OBS sends after Identify (Identified by
    default; an empty frame models OBS closing the socket). `on_request` maps a
    request to the frames OBS sends back.
    """

    def __init__(self, *, hello, on_identify=None, on_request=None):
        self.sent = []
        self.closed = False
        self._inbox = [hello]
        self._on_identify = on_identify or (
            lambda identify: [_message(2, {"negotiatedRpcVersion": 1})]
        )
        self._on_request = on_request or (lambda request: [_response(request)])

    def send(self, raw):
        message = json.loads(raw)
        self.sent.append(message)
        if message["op"] == 1:
            self._inbox.extend(self._on_identify(message["d"]))
        elif message["op"] == 6:
            self._inbox.extend(self._on_request(message["d"]))

    def recv(self):
        if not self._inbox:
            raise websocket.WebSocketTimeoutException("timed out")
        return self._inbox.pop(0)

    def close(self):
        self.closed = True


@pytest.fixture
def connect_to(monkeypatch):
    """Route `create_connection` to a scripted socket and record how it was called."""
    opened = {}

    def install(socket):
        def create_connection(url, **options):
            opened["url"] = url
            opened["options"] = options
            return socket

        monkeypatch.setattr(
            websocket_client.websocket, "create_connection", create_connection
        )
        return opened

    return install


def test_proof_matches_reference_vector():
    authentication = compute_authentication(PASSWORD, salt=SALT, challenge=CHALLENGE)

    assert authentication == EXPECTED_AUTHENTICATION


def test_identifies_without_password_when_obs_has_auth_disabled(connect_to):
    socket = ScriptedSocket(hello=_hello(auth_required=False))
    opened = connect_to(socket)

    OBSWebSocketClient("127.0.0.1", 4455, password=None, timeout=3)

    assert opened["url"] == "ws://127.0.0.1:4455"
    assert opened["options"] == {"timeout": 3, "subprotocols": ["obswebsocket.json"]}
    assert socket.sent == [
        {"op": 1, "d": {"rpcVersion": 1, "eventSubscriptions": 0}},
    ]


def test_identify_carries_password_proof_not_password(connect_to):
    socket = ScriptedSocket(hello=_hello(auth_required=True))
    connect_to(socket)

    OBSWebSocketClient("127.0.0.1", 4455, password=PASSWORD, timeout=3)

    identify = socket.sent[0]["d"]
    assert identify["authentication"] == EXPECTED_AUTHENTICATION
    assert PASSWORD not in json.dumps(socket.sent)


def test_missing_password_is_reported_before_anything_is_sent(connect_to):
    socket = ScriptedSocket(hello=_hello(auth_required=True))
    connect_to(socket)

    with pytest.raises(OBSAuthenticationError, match="requires a websocket password"):
        OBSWebSocketClient("127.0.0.1", 4455, password=None, timeout=3)

    assert socket.sent == []
    assert socket.closed is True


def test_rejected_password_is_an_authentication_error(connect_to):
    # OBS answers a wrong proof by closing the socket; websocket-client surfaces
    # a close frame as an empty payload
    socket = ScriptedSocket(
        hello=_hello(auth_required=True), on_identify=lambda identify: [""]
    )
    connect_to(socket)

    with pytest.raises(
        OBSAuthenticationError, match="password is most likely wrong"
    ) as error:
        OBSWebSocketClient("127.0.0.1", 4455, password="wrong-password", timeout=3)

    assert "wrong-password" not in str(error.value)
    assert socket.closed is True


def test_unexpected_handshake_message_is_a_connection_error(connect_to):
    socket = ScriptedSocket(hello=_message(5, {"eventType": "ExitStarted"}))
    connect_to(socket)

    with pytest.raises(ConnectionError, match="Expected a Hello"):
        OBSWebSocketClient("127.0.0.1", 4455, password=None, timeout=3)

    assert socket.closed is True


def test_call_returns_response_data_and_skips_unrelated_messages(connect_to):
    def on_request(request):
        stale = dict(request, requestId="an-earlier-request")
        return [
            _message(5, {"eventType": "CurrentProgramSceneChanged"}),
            _response(stale, response_data={"sceneName": "stale"}),
            _response(request, response_data={"obsVersion": "32.2.2"}),
        ]

    connect_to(ScriptedSocket(hello=_hello(), on_request=on_request))
    client = OBSWebSocketClient("127.0.0.1", 4455, password=None, timeout=3)

    response = client.call("GetVersion")

    assert response == {"obsVersion": "32.2.2"}


def test_call_sends_request_data_and_omits_it_when_absent(connect_to):
    socket = ScriptedSocket(hello=_hello())
    connect_to(socket)
    client = OBSWebSocketClient("127.0.0.1", 4455, password=None, timeout=3)

    with_data = client.call("SetCurrentProgramScene", data={"sceneName": "Coffee"})
    without_data = client.call("StartVirtualCam")

    requests = [message["d"] for message in socket.sent if message["op"] == 6]
    assert requests[0]["requestType"] == "SetCurrentProgramScene"
    assert requests[0]["requestData"] == {"sceneName": "Coffee"}
    assert "requestData" not in requests[1]
    assert requests[0]["requestId"] != requests[1]["requestId"]
    assert with_data == {} and without_data == {}


def test_refused_request_raises_request_error_with_code_and_comment(connect_to):
    def on_request(request):
        return [
            _response(
                request,
                result=False,
                code=600,
                comment="No source was found by the name of `Coffe`",
            )
        ]

    connect_to(ScriptedSocket(hello=_hello(), on_request=on_request))
    client = OBSWebSocketClient("127.0.0.1", 4455, password=None, timeout=3)

    with pytest.raises(OBSRequestError) as error:
        client.call("SetCurrentProgramScene", data={"sceneName": "Coffe"})

    assert error.value.request_type == "SetCurrentProgramScene"
    assert error.value.code == 600
    assert "No source was found" in str(error.value)
    assert "600" in str(error.value)


def test_silence_from_obs_surfaces_as_a_timeout(connect_to):
    connect_to(ScriptedSocket(hello=_hello(), on_request=lambda request: []))
    client = OBSWebSocketClient("127.0.0.1", 4455, password=None, timeout=3)

    with pytest.raises(websocket.WebSocketTimeoutException):
        client.call("GetVersion")


def test_close_closes_the_socket(connect_to):
    socket = ScriptedSocket(hello=_hello())
    connect_to(socket)
    client = OBSWebSocketClient("127.0.0.1", 4455, password=None, timeout=3)

    client.close()

    assert socket.closed is True


def test_request_error_message_drops_trailing_full_stop_but_keeps_comment():
    error = OBSRequestError(
        "SetCurrentProgramScene", code=600, comment="No source was found."
    )

    assert (
        str(error)
        == "OBS rejected SetCurrentProgramScene (code 600): No source was found"
    )
    assert error.comment == "No source was found."
