import pytest
import torch


@pytest.fixture(autouse=True)
def _mock_mfa_aligner(monkeypatch, request):
    """Globally mock the slow MFAAligner so tests don't pay the ~150 s per
    alignment cost (and don't need the `mfa` binary on PATH).

    Patches the alignment-running methods on MFAAligner to return canned
    success data, and the install-check / model-download hooks to no-op
    so the constructor doesn't raise. Pure-Python helpers
    (`parse_mfa_json*`, dataclasses) are left untouched so the unit tests
    in `test_forced_aligner.py` still exercise the real parser.

    Does NOT touch GigaAMAligner — that one is already fast.

    Opt out per test with `@pytest.mark.real_mfa`.
    """
    if "real_mfa" in request.keywords:
        return

    from src.preprocessing.forced_aligner import (
        MFAAligner,
        PhonemeAlignment,
        WordAlignment,
    )

    fake_phonemes = [
        PhonemeAlignment(phoneme="t", start=0.0, end=0.1),
        PhonemeAlignment(phoneme="e", start=0.1, end=0.2),
        PhonemeAlignment(phoneme="sʲ", start=0.2, end=0.3),
        PhonemeAlignment(phoneme="t", start=0.3, end=0.4),
    ]
    fake_words = [
        WordAlignment(word="тест", start=0.0, end=0.4, phonemes=list(fake_phonemes)),
    ]

    monkeypatch.setattr(MFAAligner, "_check_mfa_installed", lambda self: True)
    monkeypatch.setattr(MFAAligner, "_ensure_model_downloaded", lambda self: None)
    monkeypatch.setattr(
        MFAAligner,
        "align_from_file_full",
        lambda self, audio_path, text: (list(fake_phonemes), list(fake_words)),
    )
    monkeypatch.setattr(
        MFAAligner,
        "align_from_file",
        lambda self, audio_path, text: list(fake_phonemes),
    )
    monkeypatch.setattr(
        MFAAligner,
        "align_full",
        lambda self, audio, text, sample_rate=16000: (list(fake_phonemes), list(fake_words)),
    )
    monkeypatch.setattr(
        MFAAligner,
        "align",
        lambda self, audio, text, sample_rate=16000: list(fake_phonemes),
    )
    monkeypatch.setattr(
        MFAAligner,
        "align_batch",
        lambda self, items, sample_rate=16000: {
            utt_id: (list(fake_phonemes), list(fake_words)) for utt_id, _a, _t in items
        },
    )
    monkeypatch.setattr(
        MFAAligner,
        "run_alignment",
        lambda self, audio_path, text, output_format="json": (self.temp_dir / "fake.json"),
    )


@pytest.fixture
def dummy_waveform():
    """1 second of mono audio at 22050 Hz."""
    return torch.randn(22050)


@pytest.fixture
def dummy_mel():
    """Dummy mel-spectrogram: shape (1, 80, 64)."""
    return torch.randn(1, 80, 64).abs()  # positive values for mel


@pytest.fixture
def dummy_timestamps():
    """Example hard_s_timestamps: two segments."""
    return [[0.1, 0.2], [0.5, 0.6]]


@pytest.fixture
def minimal_config_dict():
    """Minimal valid config dict for ExperimentConfig."""
    return {
        "seed": 42,
        "precision": "fp32",
        "device": "cpu",
        "sanity_check": False,
        "data": {
            "dataset_name": "test/dataset",
            "sample_rate": 22050,
            "n_mels": 80,
            "n_fft": 1024,
            "hop_length": 256,
            "mask_padding_ms": 10.0,
            "split_train": "train",
            "split_val": "validation",
        },
        "model": {
            "type": "UNet",
            "in_channels": 2,
            "out_channels": 1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "lr": 1e-4,
            "precision": "fp32",
            "checkpoint_every_n_epochs": 1,
        },
        "loss": {
            "type": "SpectralL1",
            "masked_l1_weight": 1.0,
            "multiscale_spectral_weight": 0.5,
            "unmasked_l1_weight": 0.01,
        },
        "wandb": {
            "project": "test",
            "entity": "",
        },
    }
