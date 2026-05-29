"""Word-level alignment using GigaAM-v3 RNN-T emission timestamps.

Replaces MFAAligner for inference paths that don't need phoneme-level
boundaries. Uses the encoder timestep of each emitted SentencePiece token
to recover word starts (40 ms frame stride). Phoneme list is left empty —
phoneme-based hard-S detection must be done on text via the phonemizer.

This runs in a single forward pass through GigaAM-v3 and is ~3 orders of
magnitude faster than MFA on short clips.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.preprocessing.forced_aligner import PhonemeAlignment, WordAlignment

logger = logging.getLogger(__name__)

FRAME_STRIDE_S = 0.04  # 10 ms mel hop * subsampling_factor=4


class GigaAMAligner:
    """Word-level aligner backed by GigaAM-v3 RNN-T emission timestamps.

    Surface mirrors `MFAAligner` for the methods called by the resynthesis
    pipeline. `align_from_file_full` returns ``(phonemes=[], words)`` — the
    phoneme list is intentionally empty since RNN-T emits subword units, not
    phonemes. Callers that need phoneme-level info should use the MFA backend.
    """

    def __init__(
        self,
        transcriber=None,
        device: str = "cuda",
        late_bias_ms: float = 80.0,
    ):
        from src.preprocessing.asr import GigaAMTranscriber

        self.transcriber = transcriber or GigaAMTranscriber(device=device)
        self.late_bias_s = late_bias_ms / 1000.0

    def align_from_file_full(
        self, audio_path: str | Path, text: str
    ) -> tuple[list[PhonemeAlignment], list[WordAlignment]]:
        """Align audio to transcript, returning ([], words).

        Strategy: run RNN-T greedy decode capturing (frame, token) pairs.
        Group SentencePiece tokens by ``▁``-prefix into emitted words; map
        emitted words to transcript words by index. If counts mismatch we
        fall back to a positional best-effort match.
        """
        audio_path = str(audio_path)
        text_pred, timed_tokens = self.transcriber.transcribe_with_timings(audio_path)

        # GigaAMModel wraps GigaAMASR at .model
        asr = self.transcriber.model.model
        sp_model = asr.decoding.tokenizer.model

        # Group consecutive tokens into words by ▁-prefix
        emitted_words: list[tuple[str, int, int]] = []  # (word_text, start_t, end_t_exclusive)
        cur_pieces: list[str] = []
        cur_start_t: int | None = None
        for t, tok_id in timed_tokens:
            piece = sp_model.id_to_piece(int(tok_id))
            if piece.startswith("▁"):
                if cur_pieces and cur_start_t is not None:
                    emitted_words.append(
                        ("".join(cur_pieces).replace("▁", ""), cur_start_t, t)
                    )
                cur_pieces = [piece]
                cur_start_t = t
            else:
                if cur_start_t is None:  # leading non-▁ piece
                    cur_start_t = t
                cur_pieces.append(piece)
        if cur_pieces and cur_start_t is not None:
            # End frame for the final word: one past the last token's frame.
            last_t = timed_tokens[-1][0] if timed_tokens else cur_start_t
            emitted_words.append(
                ("".join(cur_pieces).replace("▁", ""), cur_start_t, last_t + 1)
            )

        # Map emitted words → transcript words by position. The transcript is
        # the source of truth for word identity; emission times are the source
        # of truth for boundaries.
        transcript_words = text.split()
        words: list[WordAlignment] = []
        n = min(len(emitted_words), len(transcript_words))
        if len(emitted_words) != len(transcript_words):
            logger.warning(
                "GigaAM word count mismatch: emitted=%d transcript=%d "
                "(emitted='%s' transcript='%s'). Using positional match for first %d.",
                len(emitted_words),
                len(transcript_words),
                " ".join(w for w, _, _ in emitted_words),
                text,
                n,
            )
        for i in range(n):
            transcript_word = transcript_words[i]
            _, start_t, end_t = emitted_words[i]
            # Subtract late-emission bias from both endpoints. RNN-T emits a
            # token only after enough acoustic evidence accumulates — typically
            # ~80 ms late for fricative onsets like /s/. Without this shift,
            # `guard_ms` alone can't recover the true onset and the original
            # /s/ tail bleeds into the spliced output.
            start_s = max(0.0, start_t * FRAME_STRIDE_S - self.late_bias_s)
            end_s = max(start_s, end_t * FRAME_STRIDE_S - self.late_bias_s)
            words.append(
                WordAlignment(
                    word=transcript_word,
                    start=start_s,
                    end=end_s,
                    phonemes=[],
                )
            )

        return [], words
