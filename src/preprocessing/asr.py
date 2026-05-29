"""GigaAM-v3 ASR wrapper for Russian speech transcription."""

import logging
import tempfile
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


class GigaAMTranscriber:
    """Transcribe Russian audio using GigaAM-v3 model from HuggingFace."""

    def __init__(self, device: str | torch.device = "cpu", revision: str = "e2e_rnnt"):
        self.device = torch.device(device)
        self.revision = revision
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return

        from transformers import AutoModel, AutoConfig

        logger.info("Loading GigaAM-v3 model (revision=%s)...", self.revision)

        # GigaAM uses torchaudio.MelSpectrogram in __init__, which crashes
        # under transformers>=5's meta-device initialization. We construct the
        # model on CPU manually, then load the pretrained weights.
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download
        import safetensors.torch

        config = AutoConfig.from_pretrained(
            "ai-sage/GigaAM-v3", revision=self.revision, trust_remote_code=True,
        )
        # Build model on CPU (no meta device)
        model_cls = config.__class__.model_type  # noqa: just need the class
        # Get the actual model class from the dynamic module
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        cls_ref = get_class_from_dynamic_module(
            "modeling_gigaam.GigaAMModel",
            "ai-sage/GigaAM-v3",
            revision=self.revision,
        )
        model = cls_ref(config)

        # Load weights
        weight_file = hf_hub_download(
            "ai-sage/GigaAM-v3", "pytorch_model.bin", revision=self.revision,
        )
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        self._model = model
        self._model.float()
        self._model.to(self.device)
        self._model.eval()
        logger.info("GigaAM-v3 model loaded on %s", self.device)

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file to Russian text.

        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC). Any sample rate.

        Returns:
            Transcribed Russian text.
        """
        self._load_model()

        # Resample to 16kHz if needed (GigaAM requirement)
        path_to_use = str(audio_path)
        waveform, sr = torchaudio.load(path_to_use)
        if sr != 16000:
            logger.debug("Resampling %s from %dHz to 16kHz for ASR", audio_path, sr)
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(tmp.name, waveform, 16000)
            path_to_use = tmp.name

        try:
            with torch.no_grad():
                result = self._model.transcribe(path_to_use)
        finally:
            if path_to_use != str(audio_path):
                Path(path_to_use).unlink(missing_ok=True)

        if isinstance(result, list):
            return result[0] if result else ""
        return str(result)

    @torch.inference_mode()
    def transcribe_with_timings(
        self, audio_path: str
    ) -> tuple[str, list[tuple[int, int]]]:
        """Transcribe and capture (frame_idx, token_id) per emitted token.

        Mirrors GigaAM-v3's `RNNTGreedyDecoding._greedy_decode` but records the
        encoder timestep at which each non-blank token fires. Frame stride is
        40 ms (10 ms mel hop * subsampling_factor=4).

        Returns:
            (text, timed_tokens) where timed_tokens is a list of (t, token_id);
            multiply t by 0.04 to get seconds.
        """
        self._load_model()
        # transformers' GigaAMModel wraps GigaAMASR at self.model
        asr = self._model.model
        wav, length = asr.prepare_wav(audio_path)
        encoded, encoded_len = asr.forward(wav, length)
        encoded = encoded.transpose(1, 2)  # [B, T, F]

        head = asr.head
        decoding = asr.decoding
        blank_id = decoding.blank_id
        max_symbols = decoding.max_symbols

        inseq = encoded[0, :, :].unsqueeze(1)  # [T, 1, F]
        seqlen = int(encoded_len[0].item())

        timed_tokens: list[tuple[int, int]] = []
        dec_state = None
        last_label = None
        for t in range(seqlen):
            f = inseq[t, :, :].unsqueeze(1)
            not_blank = True
            new_symbols = 0
            while not_blank and new_symbols < max_symbols:
                g, hidden = head.decoder.predict(last_label, dec_state)
                k = head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item()
                if k == blank_id:
                    not_blank = False
                else:
                    timed_tokens.append((t, int(k)))
                    dec_state = hidden
                    last_label = torch.tensor([[k]], device=inseq.device)
                    new_symbols += 1

        text = decoding.tokenizer.decode([k for _, k in timed_tokens])
        return text, timed_tokens

    @property
    def model(self):
        """Expose the underlying GigaAM model (lazy-loaded on first call)."""
        self._load_model()
        return self._model
