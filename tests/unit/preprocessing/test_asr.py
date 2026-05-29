"""Unit tests for GigaAM-v3 ASR wrapper."""

from unittest.mock import MagicMock, patch

import torch

from src.preprocessing.asr import GigaAMTranscriber


def test_transcriber_init_defaults():
    transcriber = GigaAMTranscriber()
    assert transcriber.device == torch.device("cpu")
    assert transcriber.revision == "e2e_rnnt"
    assert transcriber._model is None


def test_transcriber_init_custom_device():
    transcriber = GigaAMTranscriber(device="cpu", revision="v2")
    assert transcriber.device == torch.device("cpu")
    assert transcriber.revision == "v2"


@patch("src.preprocessing.asr.torchaudio.load", return_value=(torch.zeros(1, 16000), 16000))
@patch("src.preprocessing.asr.GigaAMTranscriber._load_model")
def test_transcribe_returns_string(mock_load, mock_audio_load):
    transcriber = GigaAMTranscriber(device="cpu")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = "Привет мир"
    transcriber._model = mock_model

    result = transcriber.transcribe("/fake/audio.wav")
    assert isinstance(result, str)
    assert result == "Привет мир"
    mock_model.transcribe.assert_called_once_with("/fake/audio.wav")


@patch("src.preprocessing.asr.torchaudio.load", return_value=(torch.zeros(1, 16000), 16000))
@patch("src.preprocessing.asr.GigaAMTranscriber._load_model")
def test_transcribe_handles_list_return(mock_load, mock_audio_load):
    transcriber = GigaAMTranscriber(device="cpu")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ["Текст один"]
    transcriber._model = mock_model

    result = transcriber.transcribe("/fake/audio.wav")
    assert result == "Текст один"


@patch("src.preprocessing.asr.torchaudio.load", return_value=(torch.zeros(1, 16000), 16000))
@patch("src.preprocessing.asr.GigaAMTranscriber._load_model")
def test_transcribe_handles_empty_list(mock_load, mock_audio_load):
    transcriber = GigaAMTranscriber(device="cpu")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = []
    transcriber._model = mock_model

    result = transcriber.transcribe("/fake/audio.wav")
    assert result == ""


class _FakeGigaAMConfig:
    """Stand-in for the GigaAMConfig class so `config.__class__.model_type` resolves."""

    model_type = "gigaam"


@patch("torch.load")
@patch("huggingface_hub.hf_hub_download")
@patch("transformers.dynamic_module_utils.get_class_from_dynamic_module")
@patch("transformers.AutoConfig.from_pretrained")
def test_load_model(mock_config_fp, mock_get_class, mock_hf_download, mock_torch_load):
    fake_config = _FakeGigaAMConfig()
    mock_config_fp.return_value = fake_config

    mock_model = MagicMock()
    mock_cls_ref = MagicMock(return_value=mock_model)
    mock_get_class.return_value = mock_cls_ref

    mock_hf_download.return_value = "/tmp/fake_weights.bin"
    mock_torch_load.return_value = {}

    transcriber = GigaAMTranscriber(device="cpu", revision="e2e_rnnt")
    transcriber._load_model()

    mock_config_fp.assert_called_once_with(
        "ai-sage/GigaAM-v3", revision="e2e_rnnt", trust_remote_code=True
    )
    mock_get_class.assert_called_once_with(
        "modeling_gigaam.GigaAMModel", "ai-sage/GigaAM-v3", revision="e2e_rnnt"
    )
    mock_cls_ref.assert_called_once_with(fake_config)
    mock_hf_download.assert_called_once_with(
        "ai-sage/GigaAM-v3", "pytorch_model.bin", revision="e2e_rnnt"
    )
    mock_torch_load.assert_called_once_with(
        "/tmp/fake_weights.bin", map_location="cpu", weights_only=True
    )
    mock_model.load_state_dict.assert_called_once_with({}, strict=False)
    mock_model.float.assert_called_once()
    mock_model.to.assert_called_once_with(torch.device("cpu"))
    mock_model.eval.assert_called_once()
    assert transcriber._model is mock_model


@patch("src.preprocessing.asr.torchaudio.save")
@patch("src.preprocessing.asr.torchaudio.functional.resample")
@patch(
    "src.preprocessing.asr.torchaudio.load",
    return_value=(torch.zeros(1, 8000), 8000),
)
@patch("src.preprocessing.asr.GigaAMTranscriber._load_model")
def test_transcribe_resamples_non_16k_audio(_load, _audio_load, mock_resample, mock_save):
    """8 kHz input triggers the resample + tempfile branch."""
    mock_resample.return_value = torch.zeros(1, 16000)

    transcriber = GigaAMTranscriber(device="cpu")
    mock_model = MagicMock()
    mock_model.transcribe.return_value = "ok"
    transcriber._model = mock_model

    result = transcriber.transcribe("/fake/audio.wav")
    assert result == "ok"
    mock_resample.assert_called_once()
    mock_save.assert_called_once()
    # transcribe was called on the temp resampled path, not the original.
    called_path = mock_model.transcribe.call_args[0][0]
    assert called_path != "/fake/audio.wav"


def test_model_property_triggers_load():
    transcriber = GigaAMTranscriber(device="cpu")
    transcriber._model = "already-loaded-sentinel"
    # property returns the loaded model without re-loading
    assert transcriber.model == "already-loaded-sentinel"


@patch("src.preprocessing.asr.torchaudio.save")
@patch("src.preprocessing.asr.torchaudio.functional.resample")
@patch(
    "src.preprocessing.asr.torchaudio.load",
    return_value=(torch.zeros(1, 8000), 8000),
)
@patch("src.preprocessing.asr.GigaAMTranscriber._load_model")
def test_transcribe_cleans_up_tempfile_after_resample(
    _load, _audio_load, mock_resample, mock_save, tmp_path
):
    """The tempfile created for resampled audio is removed after transcription."""
    mock_resample.return_value = torch.zeros(1, 16000)

    transcriber = GigaAMTranscriber(device="cpu")
    captured = {}

    mock_model = MagicMock()

    def _capture_path(path):
        captured["path"] = path
        return "ok"

    mock_model.transcribe.side_effect = _capture_path
    transcriber._model = mock_model

    transcriber.transcribe("/fake/audio.wav")
    assert "path" in captured
    from pathlib import Path as P

    # tempfile should be cleaned up afterwards.
    assert not P(captured["path"]).exists()


def test_transcribe_with_timings_emits_and_decodes_tokens():
    """transcribe_with_timings: drive RNN-T greedy decode manually with mocks.

    Builds a fake GigaAM internals tree (head.decoder, head.joint, decoding) that
    emits two tokens at frames 0 and 1, then a blank.
    """
    transcriber = GigaAMTranscriber(device="cpu")

    # Fake encoder output [B=1, T=3, F=4]
    encoded = torch.zeros(1, 4, 3)  # asr.forward returns (encoded, encoded_len)
    encoded_len = torch.tensor([3])

    asr = MagicMock()
    asr.prepare_wav.return_value = (torch.zeros(1, 16000), torch.tensor([16000]))
    asr.forward.return_value = (encoded, encoded_len)

    # joint returns a logits tensor; per frame, the inner loop runs until blank
    # so emissions need: t=0 → tok5, blank; t=1 → tok7, blank; t=2 → blank.
    blank_id = 0
    tok5 = torch.tensor([[[[-1.0, 1.0, -1.0, -1.0, -1.0, 3.0, -1.0, 1.0]]]])  # argmax=5
    tok7 = torch.tensor([[[[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]]]])  # argmax=7
    blank = torch.tensor([[[[9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]])  # argmax=0
    emissions = iter([tok5, blank, tok7, blank, blank])

    asr.head.joint.joint = MagicMock(side_effect=lambda f, g: next(emissions))
    asr.head.decoder.predict = MagicMock(return_value=(torch.zeros(1, 1), MagicMock()))

    asr.decoding.blank_id = blank_id
    asr.decoding.max_symbols = 3
    asr.decoding.tokenizer.decode = MagicMock(return_value="decoded text")

    # GigaAMModel wraps GigaAMASR at .model
    fake_model = MagicMock()
    fake_model.model = asr
    transcriber._model = fake_model

    with patch.object(transcriber, "_load_model"):
        text, timed = transcriber.transcribe_with_timings("/fake/audio.wav")

    assert text == "decoded text"
    # Two tokens emitted before the blank — frames 0 and 1
    assert len(timed) == 2
    assert timed[0] == (0, 5)
    assert timed[1] == (1, 7)


def test_transcribe_with_timings_breaks_on_max_symbols():
    """When max_symbols is hit, the inner loop exits even if no blank arrives."""
    transcriber = GigaAMTranscriber(device="cpu")

    encoded = torch.zeros(1, 4, 1)
    asr = MagicMock()
    asr.prepare_wav.return_value = (torch.zeros(1, 16000), torch.tensor([16000]))
    asr.forward.return_value = (encoded, torch.tensor([1]))

    # Always emits token_id 1 (never blank)
    def _join(f, g):
        return torch.tensor([[[[-1.0, 5.0]]]])  # argmax=1

    asr.head.joint.joint = MagicMock(side_effect=_join)
    asr.head.decoder.predict = MagicMock(return_value=(torch.zeros(1, 1), MagicMock()))
    asr.decoding.blank_id = 0
    asr.decoding.max_symbols = 2  # cap at 2 emissions per frame
    asr.decoding.tokenizer.decode = MagicMock(return_value="ok")

    fake_model = MagicMock()
    fake_model.model = asr
    transcriber._model = fake_model

    with patch.object(transcriber, "_load_model"):
        _, timed = transcriber.transcribe_with_timings("/fake/audio.wav")
    # Single frame, max 2 emissions → exactly 2 tokens
    assert len(timed) == 2
