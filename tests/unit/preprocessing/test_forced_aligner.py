"""Smoke tests for inherited MFA forced aligner module."""

import json
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.preprocessing.forced_aligner import MFAAligner, PhonemeAlignment


def test_phoneme_alignment_dataclass():
    pa = PhonemeAlignment(phoneme="s̪", start=0.5, end=0.7)
    assert pa.phoneme == "s̪"
    assert pa.start == 0.5
    assert pa.end == 0.7


def test_phoneme_alignment_to_dict():
    pa = PhonemeAlignment(phoneme="a", start=1.0, end=1.5)
    d = pa.to_dict()
    assert d == {"phoneme": "a", "start": 1.0, "end": 1.5}


def test_parse_mfa_json():
    """parse_mfa_json extracts phonemes from MFA JSON output."""
    mfa_output = {
        "tiers": {
            "phones": {
                "entries": [
                    [0.0, 0.1, "sil"],
                    [0.1, 0.3, "p"],
                    [0.3, 0.5, "r"],
                    [0.5, 0.6, ""],
                    [0.6, 0.8, "s̪"],
                ]
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mfa_output, f)
        json_path = Path(f.name)

    with patch("shutil.which", return_value="/usr/bin/mfa"), \
         patch.object(MFAAligner, "_ensure_model_downloaded"):
        aligner = MFAAligner()
        result = aligner.parse_mfa_json(json_path)

    json_path.unlink()

    # Should skip "sil" and empty phonemes
    assert len(result) == 3
    assert result[0].phoneme == "p"
    assert result[2].phoneme == "s̪"


def test_parse_mfa_json_empty_tiers():
    """parse_mfa_json handles missing tiers gracefully."""
    mfa_output = {"tiers": {}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mfa_output, f)
        json_path = Path(f.name)

    with patch("shutil.which", return_value="/usr/bin/mfa"), \
         patch.object(MFAAligner, "_ensure_model_downloaded"):
        aligner = MFAAligner()
        result = aligner.parse_mfa_json(json_path)

    json_path.unlink()
    assert result == []


def test_parse_mfa_json_full_associates_phonemes_to_words():
    """parse_mfa_json_full returns (phonemes, words) with each word holding its phonemes."""
    from src.preprocessing.forced_aligner import WordAlignment

    mfa_output = {
        "tiers": {
            "phones": {
                "entries": [
                    [0.0, 0.1, "sil"],
                    [0.1, 0.2, "p"],
                    [0.2, 0.4, "r"],
                    [0.4, 0.6, "s̪"],
                ]
            },
            "words": {
                "entries": [
                    [0.1, 0.4, "pr"],
                    [0.4, 0.6, "s"],
                    [0.6, 0.7, "<eps>"],
                ]
            },
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mfa_output, f)
        json_path = Path(f.name)

    with patch("shutil.which", return_value="/usr/bin/mfa"), \
         patch.object(MFAAligner, "_ensure_model_downloaded"):
        aligner = MFAAligner()
        phonemes, words = aligner.parse_mfa_json_full(json_path)

    json_path.unlink()

    assert len(phonemes) == 3  # sil dropped
    assert [w.word for w in words] == ["pr", "s"]  # <eps> dropped
    # First word should contain phonemes p, r
    pr = next(w for w in words if w.word == "pr")
    assert [p.phoneme for p in pr.phonemes] == ["p", "r"]
    assert isinstance(pr, WordAlignment)


def test_word_alignment_to_dict():
    from src.preprocessing.forced_aligner import WordAlignment

    pa = PhonemeAlignment(phoneme="s̪", start=0.1, end=0.2)
    wa = WordAlignment(word="саша", start=0.0, end=0.3, phonemes=[pa])
    d = wa.to_dict()
    assert d["word"] == "саша"
    assert d["phonemes"][0]["phoneme"] == "s̪"


def test_parse_mfa_json_full_missing_phones_tier():
    """parse_mfa_json_full handles missing 'phones' tier (still returns words)."""
    mfa_output = {
        "tiers": {
            "words": {"entries": [[0.0, 0.5, "hello"]]}
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mfa_output, f)
        json_path = Path(f.name)

    with patch("shutil.which", return_value="/usr/bin/mfa"), \
         patch.object(MFAAligner, "_ensure_model_downloaded"):
        aligner = MFAAligner()
        phonemes, words = aligner.parse_mfa_json_full(json_path)

    json_path.unlink()
    assert phonemes == []
    assert len(words) == 1
    assert words[0].word == "hello"


# -- Methods that the global MFA mock normally hides --------------------------
# These tests opt out of the global mock and drive the real implementations
# with subprocess mocked, so each branch shows up in coverage.

pytestmark_real_mfa = pytest.mark.real_mfa


def _real_aligner(tmp_path: Path) -> MFAAligner:
    with patch("shutil.which", return_value="/usr/bin/mfa"), \
         patch.object(MFAAligner, "_ensure_model_downloaded"):
        return MFAAligner(temp_dir=tmp_path)


@pytest.mark.real_mfa
def test_check_mfa_installed_returns_false_without_binary(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="not found"):
        MFAAligner()


@pytest.mark.real_mfa
def test_ensure_model_downloaded_downloads_when_missing(tmp_path: Path):
    """Models missing from `mfa model list` are downloaded."""
    aligner = _real_aligner(tmp_path)

    # Now drive _ensure_model_downloaded with subprocess mocked.
    fake_list = MagicMock(stdout="some_other_model\n")
    fake_download = MagicMock()

    def _fake_run(cmd, *args, **kwargs):
        if cmd[:3] == ["mfa", "model", "list"]:
            return fake_list
        return fake_download

    with patch("subprocess.run", side_effect=_fake_run) as mock_run:
        aligner._ensure_model_downloaded()
    # Should have invoked download twice — once for acoustic, once for dictionary.
    download_calls = [c for c in mock_run.call_args_list if "download" in c.args[0]]
    assert len(download_calls) == 2


@pytest.mark.real_mfa
def test_ensure_model_downloaded_swallows_subprocess_errors(tmp_path: Path, caplog):
    aligner = _real_aligner(tmp_path)
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "mfa"),
    ):
        with caplog.at_level("WARNING"):
            aligner._ensure_model_downloaded()
    assert any("Could not verify" in r.message for r in caplog.records)


@pytest.mark.real_mfa
def test_prepare_mfa_input_writes_files(tmp_path: Path):
    aligner = _real_aligner(tmp_path)
    src_wav = tmp_path / "src.wav"
    src_wav.write_bytes(b"FAKE")
    out_dir = tmp_path / "input"
    audio_out, text_out = aligner.prepare_mfa_input(src_wav, "  hello\n", out_dir)
    assert audio_out.exists()
    assert text_out.exists()
    assert text_out.read_text(encoding="utf-8") == "hello"


@pytest.mark.real_mfa
def test_run_alignment_returns_json_output_path(tmp_path: Path):
    aligner = _real_aligner(tmp_path)
    src_wav = tmp_path / "src.wav"
    src_wav.write_bytes(b"FAKE")

    def _fake_run(cmd, *args, **kwargs):
        # Simulate MFA producing utterance.json
        output_dir = Path(cmd[5])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "utterance.json").write_text("{}")
        return MagicMock(stdout="ok")

    with patch("subprocess.run", side_effect=_fake_run):
        out = aligner.run_alignment(src_wav, "test", output_format="json")
    assert out.name == "utterance.json"


@pytest.mark.real_mfa
def test_run_alignment_raises_on_subprocess_failure(tmp_path: Path):
    aligner = _real_aligner(tmp_path)
    src_wav = tmp_path / "src.wav"
    src_wav.write_bytes(b"FAKE")

    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "mfa", output="", stderr="boom"),
    ):
        with pytest.raises(RuntimeError, match="MFA alignment failed"):
            aligner.run_alignment(src_wav, "test")


@pytest.mark.real_mfa
def test_run_alignment_raises_when_output_missing(tmp_path: Path):
    aligner = _real_aligner(tmp_path)
    src_wav = tmp_path / "src.wav"
    src_wav.write_bytes(b"FAKE")

    def _fake_run(cmd, *args, **kwargs):
        # Don't produce the expected utterance.json
        return MagicMock(stdout="ok")

    with patch("subprocess.run", side_effect=_fake_run):
        with pytest.raises(RuntimeError, match="did not produce output"):
            aligner.run_alignment(src_wav, "test")


@pytest.mark.real_mfa
def test_align_from_file_delegates_to_run_alignment(tmp_path: Path):
    aligner = _real_aligner(tmp_path)

    json_path = tmp_path / "x.json"
    json_path.write_text(json.dumps({"tiers": {"phones": {"entries": [[0.0, 0.1, "p"]]}}}))

    with patch.object(aligner, "run_alignment", return_value=json_path):
        phonemes = aligner.align_from_file("/x.wav", "test")
    assert len(phonemes) == 1
    assert phonemes[0].phoneme == "p"


@pytest.mark.real_mfa
def test_align_writes_temp_audio_and_delegates(tmp_path: Path):
    aligner = _real_aligner(tmp_path)

    json_path = tmp_path / "x.json"
    json_path.write_text(json.dumps({"tiers": {"phones": {"entries": [[0.0, 0.1, "p"]]}}}))

    audio = np.zeros(16000, dtype=np.float32)
    with patch.object(aligner, "run_alignment", return_value=json_path):
        phonemes = aligner.align(audio, "test", sample_rate=16000)
    assert phonemes[0].phoneme == "p"


@pytest.mark.real_mfa
def test_align_full_returns_words(tmp_path: Path):
    aligner = _real_aligner(tmp_path)
    json_path = tmp_path / "x.json"
    json_path.write_text(json.dumps({
        "tiers": {
            "phones": {"entries": [[0.0, 0.1, "p"]]},
            "words": {"entries": [[0.0, 0.1, "p"]]},
        }
    }))
    audio = np.zeros(16000, dtype=np.float32)
    with patch.object(aligner, "run_alignment", return_value=json_path):
        phonemes, words = aligner.align_full(audio, "p")
    assert len(words) == 1
    assert words[0].word == "p"


@pytest.mark.real_mfa
def test_align_batch_collects_present_outputs(tmp_path: Path):
    aligner = _real_aligner(tmp_path)
    out_dir = aligner.temp_dir / "batch_output"

    def _fake_run(cmd, *args, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        # Only produce output for utt1, skip utt2
        (out_dir / "utt1.json").write_text(json.dumps({
            "tiers": {"phones": {"entries": [[0.0, 0.1, "p"]]}}
        }))
        return MagicMock(stdout="ok")

    items = [
        ("utt1", np.zeros(16000, dtype=np.float32), "text"),
        ("utt2", np.zeros(16000, dtype=np.float32), "text"),
    ]
    with patch("subprocess.run", side_effect=_fake_run):
        results = aligner.align_batch(items)
    assert "utt1" in results
    assert "utt2" not in results


@pytest.mark.real_mfa
def test_align_batch_raises_on_subprocess_failure(tmp_path: Path):
    aligner = _real_aligner(tmp_path)
    items = [("utt1", np.zeros(16000, dtype=np.float32), "text")]
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "mfa", stderr="oops"),
    ):
        with pytest.raises(RuntimeError, match="batch alignment failed"):
            aligner.align_batch(items)


@pytest.mark.real_mfa
def test_cleanup_removes_temp_dir(tmp_path: Path):
    aligner = _real_aligner(tmp_path / "stuff")
    aligner.temp_dir.mkdir(parents=True, exist_ok=True)
    (aligner.temp_dir / "file").write_text("x")
    aligner.cleanup()
    assert not aligner.temp_dir.exists()
