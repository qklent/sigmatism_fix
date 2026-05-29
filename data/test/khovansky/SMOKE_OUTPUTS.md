# Хованский reference — smoke-test outputs

Outputs from `python -m src.resynthesize --save-raw-tts` on
`data/test/check_sigmatism/audio.wav` (transcript:
"Всем привет. Сегодня я покажу, как надо пить сок и есть салат.", two hard-S
words: `сок`, `салат`) using two different reference builds from the same
source interview (1:24:32–1:25:11 of https://youtu.be/sEpXl-AbvrA).

| File | Reference used | Blocked letters in reference | Notes |
|------|----------------|------------------------------|-------|
| `corrected_khov.raw_tts.wav` | `khovansky_no_s_reference.wav` (superseded, not in repo) | с/з/ц/ч/ш/щ/ж (all sibilants) | Raw TTS; S is extremely long/smeared because the reference had zero sibilant evidence for the speaker encoder. |
| `corrected_khov.wav` | same | same | Spliced into original; sounds clean because `max_overlap_ms` clips the long S tail down to the original word's duration. |
| `corrected_khov_v2.raw_tts.wav` | `khovansky_no_hard_s_reference.wav` | с/з/ц only (palatal ш/щ/ж/ч allowed) | Raw TTS; S is much shorter — model now has per-speaker palatal-sibilant evidence. |
| `corrected_khov_v2.wav` | same | same | Spliced into original; cleanest final result. |

Use the v2 files as the reference-quality target when tuning.
