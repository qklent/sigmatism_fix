"""Config loading and validation for sigmatism_fix.

Loads YAML experiment configs via OmegaConf, merges with default.yaml,
and validates using the Pydantic schema.

Strict rules enforced at startup:
  - ``precision`` field MUST be present; absence raises ``ValidationError``.
  - ``device`` defaults to "cuda" if available, else "cpu".
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.config.schema import ExperimentConfig

# Location of the built-in default config, relative to repo root.
_DEFAULTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from a YAML file.

    1. Load ``configs/default.yaml`` as base (if it exists).
    2. Merge with the user-provided config at *path* using OmegaConf.
    3. Validate using the Pydantic :class:`ExperimentConfig` schema.
    4. Return the validated config as a Pydantic model.

    Parameters
    ----------
    path:
        Path to the YAML config file.

    Returns
    -------
    ExperimentConfig
        Fully validated experiment config.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    pydantic.ValidationError
        If validation fails (e.g. missing ``precision``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Step 1 & 2: load and merge
    if _DEFAULTS_PATH.exists():
        base = OmegaConf.load(_DEFAULTS_PATH)
    else:
        base = OmegaConf.create({})

    override = OmegaConf.load(path)
    merged = OmegaConf.merge(base, override)

    # Convert to plain dict (resolve interpolations)
    cfg_dict: dict = OmegaConf.to_container(merged, resolve=True)  # type: ignore[assignment]

    # Step 3 & 4: validate and return
    return validate_config(cfg_dict)


def validate_config(cfg: dict) -> ExperimentConfig:
    """Validate a config dict against the Pydantic schema.

    Parameters
    ----------
    cfg:
        A plain dict (usually from OmegaConf).

    Returns
    -------
    ExperimentConfig
        Validated config.

    Raises
    ------
    pydantic.ValidationError
        If any required field is missing or has an invalid value.
    """
    return ExperimentConfig.model_validate(cfg)
