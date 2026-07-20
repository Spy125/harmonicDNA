"""Tests for chord detection (no audio files needed)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from harmonicdna.chord_detector import (
    detect_chords, chords_to_sequence, ChordLabel, _TEMPLATES, _CHORD_NAMES
)


def _make_chroma(chord_name: str, n_frames: int = 5) -> np.ndarray:
    """Build a synthetic chromagram where every frame is a known chord."""
    idx  = _CHORD_NAMES.index(chord_name)
    tmpl = _TEMPLATES[idx]
    return np.tile(tmpl[:, None], (1, n_frames))  # (12, n_frames)


class TestChordDetector:
    def test_detects_cmaj(self):
        chroma = _make_chroma("Cmaj", n_frames=4)
        labels = detect_chords(chroma)
        assert all(lbl.name == "Cmaj" for lbl in labels)

    def test_detects_amin(self):
        chroma = _make_chroma("Amin", n_frames=3)
        labels = detect_chords(chroma)
        assert all(lbl.name == "Amin" for lbl in labels)

    def test_frame_index_set(self):
        chroma = _make_chroma("Gmaj", n_frames=5)
        labels = detect_chords(chroma)
        assert labels[0].frame == 0
        assert labels[-1].frame == 4

    def test_confidence_in_range(self):
        chroma = _make_chroma("Fmaj", n_frames=3)
        for lbl in detect_chords(chroma):
            assert 0.0 <= lbl.confidence <= 1.0

    def test_silent_frame_skipped(self):
        chroma = np.zeros((12, 5))
        labels = detect_chords(chroma)
        assert len(labels) == 0

    def test_min_confidence_filter(self):
        chroma = _make_chroma("Cmaj", n_frames=3)
        # very high threshold - should filter out everything
        labels = detect_chords(chroma, min_confidence=1.1)
        assert len(labels) == 0

    def test_chords_to_sequence_deduplicates(self):
        labels = [
            ChordLabel("Cmaj", 0, 0.9),
            ChordLabel("Cmaj", 1, 0.9),
            ChordLabel("Gmaj", 2, 0.85),
            ChordLabel("Gmaj", 3, 0.85),
            ChordLabel("Cmaj", 4, 0.9),
        ]
        seq = chords_to_sequence(labels)
        assert seq == ["Cmaj", "Gmaj", "Cmaj"]

    def test_templates_shape(self):
        assert _TEMPLATES.shape == (24, 12)
