"""OmniVoice TTS adapter for zero-shot voice cloning resynthesis.

Follows the HiFi-GAN adapter pattern: config-driven, eval mode, no gradients.
"""

from __future__ import annotations

import logging

import torch

from src.config.schema import OmniVoiceConfig

logger = logging.getLogger(__name__)


class OmniVoiceAdapter:
    """Wraps OmniVoice model for text-to-speech with voice cloning."""

    def __init__(self, cfg: OmniVoiceConfig, device: str = "cuda") -> None:
        self.cfg = cfg
        self.device = device

        from omnivoice import OmniVoice

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = dtype_map[cfg.dtype]

        logger.info("Loading OmniVoice from %s on %s (%s)...", cfg.model_name, device, cfg.dtype)
        self.model = OmniVoice.from_pretrained(
            cfg.model_name, device_map=device, dtype=dtype
        )
        self.model.eval()
        self.model.requires_grad_(False)
        logger.info("OmniVoice loaded on %s", device)

    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Synthesize speech from text using voice cloning.

        Args:
            text: Text to synthesize.
            ref_audio_path: Path to reference audio for voice cloning.
            ref_text: Optional transcript of reference audio.

        Returns:
            Tuple of (waveform tensor shape (1, T), sample_rate).
        """
        kwargs: dict = {
            "text": text,
            "ref_audio": ref_audio_path,
            "num_step": self.cfg.num_step,
            "speed": self.cfg.speed,
        }
        if ref_text is not None:
            kwargs["ref_text"] = ref_text

        with torch.no_grad():
            result = self.model.generate(**kwargs)

        # OmniVoice may return list[Tensor|ndarray] or Tensor|ndarray at 24kHz
        waveform = result[0] if isinstance(result, list) else result
        if not isinstance(waveform, torch.Tensor):
            import numpy as np
            waveform = torch.from_numpy(np.asarray(waveform))
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return waveform, self.cfg.sample_rate
