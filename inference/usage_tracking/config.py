from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from inference.core.env import METRICS_COLLECTOR_BASE_URL
from inference.core.utils.url_utils import wrap_url


class TelemetrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="telemetry_")

    api_usage_endpoint_url: str = wrap_url(
        f"{METRICS_COLLECTOR_BASE_URL}/usage/inference"
    )
    api_plan_endpoint_url: str = wrap_url(f"{METRICS_COLLECTOR_BASE_URL}/usage/plan")
    flush_interval: int = Field(default=10, ge=10, le=300)
    opt_out: Optional[bool] = False
    queue_size: int = Field(default=10, ge=10, le=10000)

    @model_validator(mode="after")
    def check_values(cls, inst: TelemetrySettings):
        inst.flush_interval = min(max(inst.flush_interval, 10), 300)
        inst.queue_size = min(max(inst.queue_size, 10), 10000)
        return inst


@lru_cache
def get_telemetry_settings() -> TelemetrySettings:
    return TelemetrySettings()
