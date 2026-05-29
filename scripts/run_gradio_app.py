"""Launcher for the Gradio hard-S correction demo (feature 015).

Loads configs/resynthesis.yaml, constructs a single ResynthesisPipeline, and
serves the UI at <host>:<port>. share=True is hardcoded off.

Usage:
    HF_TOKEN=... python scripts/run_gradio_app.py
    python scripts/run_gradio_app.py --port 7861 --runs-root /tmp/app_runs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.apps.gradio_app.app import build_interface
from src.apps.gradio_app.runner import GradioRunner
from src.apps.gradio_app.schema import AppConfig
from src.config.schema import OmniVoiceConfig, ResynthesisConfig
from src.pipelines.resynthesis_pipeline import ResynthesisPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> AppConfig:
    parser = argparse.ArgumentParser(description="Hard-S correction Gradio demo")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (FR-017)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (FR-017)")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("data/app_runs"),
        help="Root directory for per-run artifact dirs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/resynthesis.yaml"),
        help="Pipeline config (OmegaConf YAML)",
    )
    ns = parser.parse_args()
    return AppConfig(
        host=ns.host,
        port=ns.port,
        runs_root=ns.runs_root,
        config_path=ns.config,
    )


def _build_pipeline(cfg_path: Path) -> ResynthesisPipeline:
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    raw = OmegaConf.load(cfg_path)
    omnivoice_cfg = OmniVoiceConfig(**OmegaConf.to_container(raw.get("omnivoice", {}), resolve=True))
    resynthesis_cfg = ResynthesisConfig(
        **OmegaConf.to_container(raw.get("resynthesis", {}), resolve=True)
    )
    device = str(raw.get("device", "cuda:0"))
    pipeline = ResynthesisPipeline(omnivoice_cfg, resynthesis_cfg, device=device)
    logger.info("pipeline loaded: id=%s", id(pipeline))
    return pipeline


def main() -> None:
    app_config = _parse_args()
    app_config.runs_root.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(app_config.config_path)
    runner = GradioRunner(app_config, pipeline)
    demo = build_interface(runner)

    logger.info(
        "Launching Gradio demo on http://%s:%d (share=False)",
        app_config.host,
        app_config.port,
    )
    demo.launch(
        server_name=app_config.host,
        server_port=app_config.port,
        share=False,
    )


if __name__ == "__main__":
    main()
