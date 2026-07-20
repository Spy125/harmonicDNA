"""Smith-Waterman local alignment for chord sequences."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    score: float
    seq_a_aligned: list[str]
    seq_b_aligned: list[str]
    start_a: int
    start_b: int
    identity: float   # fraction of aligned positions that match


def _score_fn(a: str, b: str, match: float = 2.0, mismatch: float = -1.0) -> float:
    return match if a == b else mismatch


def align(
    seq_a: list[str],
    seq_b: list[str],
    match: float = 2.0,
    mismatch: float = -1.0,
    gap: float = -1.0,
) -> AlignmentResult:
    """Smith-Waterman local alignment of two chord sequences."""
    n, m = len(seq_a), len(seq_b)
    H    = np.zeros((n + 1, m + 1))

    # fill scoring matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s   = _score_fn(seq_a[i - 1], seq_b[j - 1], match, mismatch)
            H[i, j] = max(
                0,
                H[i - 1, j - 1] + s,
                H[i - 1, j]     + gap,
                H[i,     j - 1] + gap,
            )

    # find best score position
    best_score = float(H.max())
    if best_score == 0:
        return AlignmentResult(
            score=0.0, seq_a_aligned=[], seq_b_aligned=[],
            start_a=0, start_b=0, identity=0.0
        )

    idx      = np.unravel_index(np.argmax(H), H.shape)
    i, j     = int(idx[0]), int(idx[1])
    end_a, end_b = i, j

    # traceback
    aligned_a, aligned_b = [], []
    while i > 0 and j > 0 and H[i, j] > 0:
        s = _score_fn(seq_a[i - 1], seq_b[j - 1], match, mismatch)
        if H[i, j] == H[i - 1, j - 1] + s:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(seq_b[j - 1])
            i -= 1; j -= 1
        elif H[i, j] == H[i - 1, j] + gap:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append("-")
            i -= 1
        else:
            aligned_a.append("-")
            aligned_b.append(seq_b[j - 1])
            j -= 1

    aligned_a.reverse(); aligned_b.reverse()
    matches   = sum(a == b for a, b in zip(aligned_a, aligned_b))
    length    = max(len(aligned_a), 1)

    return AlignmentResult(
        score         = best_score,
        seq_a_aligned = aligned_a,
        seq_b_aligned = aligned_b,
        start_a       = i,
        start_b       = j,
        identity      = round(matches / length, 3),
    )
