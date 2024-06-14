from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Optional


class TelemetrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="telemetry")

    opt_out: Optional[bool] = False
    queue_size: int = 10


@lru_cache
def get_telemetry_settings() -> TelemetrySettings:
    return TelemetrySettings()
