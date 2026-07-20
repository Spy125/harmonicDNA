"""Tests for Smith-Waterman chord aligner."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from harmonicdna.aligner import align, AlignmentResult


class TestAlign:
    def test_identical_sequences_high_score(self):
        seq = ["Cmaj", "Gmaj", "Amin", "Fmaj"]
        r   = align(seq, seq)
        assert r.score > 0
        assert r.identity == pytest.approx(1.0)

    def test_empty_sequences(self):
        r = align([], [])
        assert r.score == 0.0
        assert r.seq_a_aligned == []

    def test_no_match_returns_zero(self):
        # sequences with completely different chords
        r = align(["Cmaj", "Gmaj"], ["Dmin", "Emin"])
        # mismatch score is negative, so SW should return 0
        assert r.score == 0.0

    def test_partial_match(self):
        a = ["Cmaj", "Gmaj", "Amin", "Fmaj"]
        b = ["Dmin", "Cmaj", "Gmaj", "Emin"]
        r = align(a, b)
        # at least "Cmaj", "Gmaj" should align
        assert r.score > 0
        assert len(r.seq_a_aligned) >= 2

    def test_alignment_lengths_match(self):
        a = ["Cmaj", "Gmaj", "Amin"]
        b = ["Cmaj", "Fmaj", "Amin"]
        r = align(a, b)
        assert len(r.seq_a_aligned) == len(r.seq_b_aligned)

    def test_identity_in_range(self):
        a = ["Cmaj", "Gmaj", "Amin", "Fmaj"]
        b = ["Cmaj", "Gmaj", "Dmin", "Fmaj"]
        r = align(a, b)
        assert 0.0 <= r.identity <= 1.0
