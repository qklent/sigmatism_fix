
from src.data.masking import generate_mask

SR = 22050
HOP = 256


def _frame_idx(time_sec: float) -> int:
    """Convert time in seconds to frame index."""
    return int(time_sec * SR / HOP)


def test_mask_shape():
    """Output shape should be (1, 1, num_frames)."""
    num_frames = 100
    mask = generate_mask([[0.1, 0.2]], num_frames=num_frames, sample_rate=SR, hop_length=HOP, padding_ms=0.0)
    assert mask.shape == (1, 1, num_frames)


def test_mask_covers_correct_frames():
    """Given known timestamps and no padding, specific frames should be masked."""
    num_frames = 200
    mask = generate_mask(
        [[0.1, 0.2]], num_frames=num_frames, sample_rate=SR, hop_length=HOP, padding_ms=0.0
    )
    start = _frame_idx(0.1)
    end = _frame_idx(0.2)

    flat = mask.squeeze()
    # Frames in the segment should be 1.0
    for i in range(start, end):
        assert flat[i].item() == 1.0, f"Frame {i} should be masked"
    # A frame well outside the segment should be 0.0
    assert flat[0].item() == 0.0, "Frame 0 should not be masked"


def test_padding_extends_mask():
    """With padding > 0, the mask should cover more frames than with padding=0."""
    num_frames = 200
    timestamps = [[0.1, 0.2]]

    mask_no_pad = generate_mask(timestamps, num_frames=num_frames, sample_rate=SR, hop_length=HOP, padding_ms=0.0)
    mask_with_pad = generate_mask(timestamps, num_frames=num_frames, sample_rate=SR, hop_length=HOP, padding_ms=50.0)

    count_no_pad = mask_no_pad.sum().item()
    count_with_pad = mask_with_pad.sum().item()

    assert count_with_pad > count_no_pad, "Padding should extend the masked region"


def test_multiple_segments():
    """Two non-overlapping segments should produce two distinct masked regions."""
    num_frames = 500
    mask = generate_mask(
        [[0.1, 0.15], [0.5, 0.55]],
        num_frames=num_frames, sample_rate=SR, hop_length=HOP, padding_ms=0.0,
    )
    flat = mask.squeeze()

    # There should be an unmasked gap between the two segments
    mid_frame = _frame_idx(0.3)
    assert flat[mid_frame].item() == 0.0, "Gap between segments should be unmasked"

    # Both segments should have masked frames
    seg1_frame = _frame_idx(0.12)
    seg2_frame = _frame_idx(0.52)
    assert flat[seg1_frame].item() == 1.0, "First segment should be masked"
    assert flat[seg2_frame].item() == 1.0, "Second segment should be masked"


def test_edge_case_start_of_clip():
    """Timestamp starting at 0.0 should work without error."""
    num_frames = 100
    mask = generate_mask(
        [[0.0, 0.05]], num_frames=num_frames, sample_rate=SR, hop_length=HOP, padding_ms=0.0
    )
    assert mask.shape == (1, 1, num_frames)
    assert mask.sum().item() > 0, "Mask should have some masked frames"


def test_edge_case_end_of_clip():
    """Timestamp ending at the clip boundary should work without error."""
    num_frames = 100
    clip_duration = num_frames * HOP / SR
    mask = generate_mask(
        [[clip_duration - 0.05, clip_duration]],
        num_frames=num_frames, sample_rate=SR, hop_length=HOP, padding_ms=0.0,
    )
    assert mask.shape == (1, 1, num_frames)
    assert mask.sum().item() > 0, "Mask should have some masked frames at clip end"


def test_adjacent_segments():
    """Adjacent segments (end of one = start of next) should produce a merged mask."""
    num_frames = 500
    mask = generate_mask(
        [[0.1, 0.2], [0.2, 0.3]],
        num_frames=num_frames, sample_rate=SR, hop_length=HOP, padding_ms=0.0,
    )
    flat = mask.squeeze()

    # The boundary frame should be masked (no gap between adjacent segments)
    boundary_frame = _frame_idx(0.2)
    assert flat[boundary_frame].item() == 1.0, "Boundary between adjacent segments should be masked"

    # The full range from start of first to end of second should be masked
    start = _frame_idx(0.1)
    end = _frame_idx(0.3)
    for i in range(start, end):
        assert flat[i].item() == 1.0, f"Frame {i} in merged region should be masked"
