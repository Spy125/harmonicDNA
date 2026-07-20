"""Detect chord labels from a chromagram using template matching."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# 12 pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
_PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Major chord intervals (root, major 3rd, perfect 5th) as semitone offsets
_MAJOR_OFFSETS = [0, 4, 7]
# Minor chord intervals
_MINOR_OFFSETS = [0, 3, 7]


def _make_templates() -> tuple[list[str], np.ndarray]:
    """Build 24 chord templates (12 major + 12 minor), shape (24, 12)."""
    names     = []
    templates = []
    for i, root in enumerate(_PITCH_CLASSES):
        # major
        t = np.zeros(12)
        for offset in _MAJOR_OFFSETS:
            t[(i + offset) % 12] = 1.0
        names.append(f"{root}maj")
        templates.append(t)
        # minor
        t = np.zeros(12)
        for offset in _MINOR_OFFSETS:
            t[(i + offset) % 12] = 1.0
        names.append(f"{root}min")
        templates.append(t)
    return names, np.array(templates)   # (24, 12)


_CHORD_NAMES, _TEMPLATES = _make_templates()


@dataclass
class ChordLabel:
    name: str      # e.g. "Gmaj", "Amin"
    frame: int
    confidence: float


def detect_chords(chroma: np.ndarray, min_confidence: float = 0.5) -> list[ChordLabel]:
    """
    Match each chroma frame against the 24 chord templates.
    chroma shape: (12, T)
    Returns list of ChordLabel, one per frame (only those above min_confidence).
    """
    labels = []
    for t in range(chroma.shape[1]):
        frame = chroma[:, t]
        norm  = np.linalg.norm(frame)
        if norm < 1e-6:
            continue
        frame_n = frame / norm

        # cosine similarity against each template
        sims    = _TEMPLATES @ frame_n
        best    = int(np.argmax(sims))
        conf    = float(sims[best])
        if conf >= min_confidence:
            labels.append(ChordLabel(
                name=_CHORD_NAMES[best], frame=t, confidence=round(conf, 3)
            ))
    return labels


def chords_to_sequence(labels: list[ChordLabel]) -> list[str]:
    """Collapse consecutive identical chords into a unique sequence."""
    seq  = []
    prev = None
    for lbl in labels:
        if lbl.name != prev:
            seq.append(lbl.name)
            prev = lbl.name
    return seq
