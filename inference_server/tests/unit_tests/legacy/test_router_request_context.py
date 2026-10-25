from inference_model_manager.model_manager import ModelManager
from inference_model_manager.registry_defaults import registry
from inference_model_manager.serializers_typed import serialize_passthrough
from inference_model_manager.validators import validate_passthrough
from inference_models.models.vllm_proxy.vllm_client import get_request_id
from inference_server.gateway import ModelManagerGateway
from tests.unit_tests.legacy.test_router_core_models import _image


class ContextProbeModel:
    seen_request_ids: list = []

    def prompt(self, images=None, prompt=None, **kwargs):
        self.seen_request_ids.append(get_request_id())

        return "ok"


registry.register(
    ContextProbeModel,
    "prompt",
    method="prompt",
    default=True,
    params=["images", "prompt"],
    validator=validate_passthrough,
    serializer=serialize_passthrough,
    response_type="roboflow-text-v1",
)


class ProbeBackend:
    def __init__(self):
        self.model = ContextProbeModel()
        self.state = "loaded"
        self.is_accepting = True
        self.is_healthy = True
        self.device = "cpu"
        self.last_used_ts = None

    def record_inference(self, t0, error=False):
        pass

    def stats(self):
        return {
            "backend_type": "fake",
            "state": self.state,
            "is_accepting": True,
            "inference_count": 0,
            "error_count": 0,
        }

    def unload(self, drain=False, drain_timeout_s=30.0):
        pass


REQUEST_ID = "0123456789abcdef0123456789abcdef"


def test_lmm_model_call_sees_the_request_correlation_id(legacy_client, fake_stat):
    fake_stat["qwen/1"] = ("vlm", "prompt")
    manager = ModelManager()
    manager._create_backend = (
        lambda model_id, api_key, backend, **kwargs: ProbeBackend()
    )
    ContextProbeModel.seen_request_ids = []

    response = legacy_client(ModelManagerGateway(manager)).post(
        "/infer/lmm",
        json={"model_id": "qwen/1", "image": _image(), "prompt": "hi"},
        headers={"X-Request-ID": REQUEST_ID},
    )

    assert response.status_code == 200, response.text
    assert ContextProbeModel.seen_request_ids == [REQUEST_ID]
