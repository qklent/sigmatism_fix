"""Pydantic v2 config schema for sigmatism_fix.

Validates all experiment config fields at startup. Key rules:
  - ``precision`` is REQUIRED with no default — missing → ValidationError.
  - ``device`` defaults to "cuda" if torch.cuda.is_available(), else "cpu".
  - All numeric / string constraints are enforced via Pydantic field types.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, Field, field_validator, model_validator

# ── Section schemas ───────────────────────────────────────────────────────────


class SpeakerSplitConfig(BaseModel):
    """Speaker-disjoint train/val split settings."""

    enabled: bool = False
    val_speakers: list[str] | None = None
    num_val_speakers: int = 2

    @field_validator("num_val_speakers")
    @classmethod
    def _num_val_speakers_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("num_val_speakers must be >= 1")
        return v

    @field_validator("val_speakers")
    @classmethod
    def _val_speakers_non_empty(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("val_speakers must be non-empty if provided")
        return v


class DataConfig(BaseModel):
    """Audio data / dataset settings."""

    dataset_name: str
    sample_rate: int = 22050
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    mask_padding_ms: float = 10.0
    max_audio_seconds: float | None = None
    split_train: str = "train"
    split_val: str = "validation"
    num_workers: int = 4
    persistent_workers: bool = True
    speaker_split: SpeakerSplitConfig = SpeakerSplitConfig()

    # Log-mel normalization: converts linear mel to log domain and normalizes
    # to [0, 1] range.  This compresses dynamic range so high-frequency
    # sibilant bands contribute equally to the loss.
    use_log_mel: bool = True
    log_mel_floor: float = 1e-5  # added before log to avoid -inf
    mel_norm_min: float = -11.5  # approximate log(1e-5)
    mel_norm_max: float = 2.0    # approximate log of loud speech mel values


class ModelConfig(BaseModel):
    """Model architecture settings."""

    type: str  # Registry name, e.g. "UNet"
    in_channels: int = 2
    out_channels: int = 1


class TrainingConfig(BaseModel):
    """Training hyper-parameters."""

    epochs: int
    batch_size: int
    lr: float
    precision: Literal["bf16", "fp32"]
    checkpoint_every_n_epochs: int


class LossConfig(BaseModel):
    """Loss function settings."""

    type: str
    masked_l1_weight: float = 1.0
    multiscale_spectral_weight: float = 0.5
    unmasked_l1_weight: float = 0.01


class WandbConfig(BaseModel):
    """Weights & Biases logging settings."""

    project: str
    entity: str = ""


class SanityCheckConfig(BaseModel):
    """Sanity check settings for single-batch overfit test."""

    enabled: bool = True
    overfit_steps: int = Field(default=50, gt=0)
    save_audio: bool = True


class ValidationAudioConfig(BaseModel):
    """Settings for logging validation audio samples to W&B."""

    enabled: bool = True
    num_samples: int = Field(default=10, gt=0)
    log_every_n_epochs: int = Field(default=1, gt=0)


class VocoderConfig(BaseModel):
    """Vocoder selection and weight download settings."""

    type: str = "hifigan"
    checkpoint_repo: str = "jaketae/hifigan-lj-v1"
    checkpoint_filename: str = "pytorch_model.bin"


class OmniVoiceConfig(BaseModel):
    """OmniVoice TTS model configuration for resynthesis."""

    model_config = {"protected_namespaces": ()}

    model_name: str = "k2-fsa/OmniVoice"
    dtype: Literal["float16", "bfloat16"] = "float16"
    num_step: int = Field(default=32, gt=0)
    speed: float = Field(default=1.0, gt=0)
    sample_rate: int = 24000


class ResynthesisConfig(BaseModel):
    """Resynthesis pipeline configuration for word-level splicing."""

    crossfade_ms: float = Field(default=10.0, ge=0)
    max_overlap_ms: float = Field(default=50.0, ge=0)
    guard_ms: float = Field(default=0.0, ge=0)
    aligner: Literal["mfa", "gigaam"] = "mfa"
    provide_ref_text: bool = True
    output_sample_rate: int | None = None


# ── Top-level schema ─────────────────────────────────────────────────────────


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class ExperimentConfig(BaseModel):
    """Full experiment config — the single source of truth for a run."""

    seed: int = 42
    precision: Literal["bf16", "fp32"]  # REQUIRED, no default
    device: str = Field(default_factory=_default_device)
    sanity_check: SanityCheckConfig = SanityCheckConfig()
    val_audio: ValidationAudioConfig = ValidationAudioConfig()
    vocoder: VocoderConfig = VocoderConfig()

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig
    wandb: WandbConfig

    # Optional — only required for resynthesis pipeline
    omnivoice: OmniVoiceConfig | None = None
    resynthesis: ResynthesisConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _pre_validate(cls, values: dict) -> dict:  # type: ignore[override]
        """Pre-process config values before validation."""
        # Reject missing precision with a clear message.
        if "precision" not in values or values["precision"] is None:
            raise ValueError("precision is REQUIRED — set it to 'bf16' or 'fp32' in your config")
        # Backwards compatibility: sanity_check: true/false → SanityCheckConfig.
        sc = values.get("sanity_check")
        if isinstance(sc, bool):
            values["sanity_check"] = {"enabled": sc}
        return values

    model_config = {"extra": "ignore"}
