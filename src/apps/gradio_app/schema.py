"""Pydantic v2 models for the Gradio correction app (feature 015)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class RunResultEnvelope(BaseModel):
    """On-disk record written to data/app_runs/<run_id>/result.json.

    Matches contracts/result_json.schema.json. Exactly one of `error` and
    `pipeline_result` is non-null for any written envelope.
    """

    run_id: str = Field(pattern=r"^\d{8}-\d{6}-[0-9a-f]{6}$")
    created_at: str
    speech_transcript: str | None
    reference_transcript: str | None
    error: str | None
    pipeline_result: dict | None

    model_config = {"extra": "forbid"}


class AppConfig(BaseModel):
    """Launcher-only deployment config."""

    host: str = "0.0.0.0"
    port: int = 7860
    runs_root: Path = Path("data/app_runs")
    config_path: Path = Path("configs/resynthesis.yaml")

    @field_validator("port")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {v}")
        return v

    @field_validator("runs_root", "config_path", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v) if not isinstance(v, Path) else v
