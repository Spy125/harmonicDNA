"""Tests for similarity scoring."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from harmonicdna.aligner import align, AlignmentResult
from harmonicdna.scoring import similarity_score, self_align_score, SimilarityScore


def _identical_result(seq):
    return align(seq, seq)


class TestSimilarityScore:
    def test_identical_gives_max(self):
        seq   = ["Cmaj", "Gmaj", "Amin", "Fmaj"]
        r     = _identical_result(seq)
        max_s = self_align_score(seq)
        s     = similarity_score(r, max_s)
        assert s.normalised == pytest.approx(1.0)
        assert s.verdict == "identical"

    def test_zero_score_is_distant(self):
        r = AlignmentResult(score=0.0, seq_a_aligned=[], seq_b_aligned=[],
                            start_a=0, start_b=0, identity=0.0)
        s = similarity_score(r, max_possible=10.0)
        assert s.verdict == "distant"
        assert s.normalised == 0.0

    def test_normalised_in_range(self):
        seq_a = ["Cmaj", "Gmaj"]
        seq_b = ["Cmaj", "Amin"]
        r     = align(seq_a, seq_b)
        max_s = self_align_score(seq_a)
        s     = similarity_score(r, max_s)
        assert 0.0 <= s.normalised <= 1.0

    def test_self_align_score_positive(self):
        assert self_align_score(["Cmaj", "Gmaj"]) > 0

    def test_self_align_empty(self):
        assert self_align_score([]) == 0.0
