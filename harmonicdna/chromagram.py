"""Compute chromagram (12-bin pitch class energy) from an audio file."""

from __future__ import annotations

import numpy as np


def compute_chromagram(
    path: str,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> np.ndarray:
    """Return chromagram matrix shape (12, T) using librosa."""
    import librosa
    y, sr  = librosa.load(path, sr=None, mono=True)
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
    )
    return chroma  # shape (12, T)


def chroma_to_frames(chroma: np.ndarray) -> list[np.ndarray]:
    """Split (12, T) matrix into a list of T frame vectors."""
    return [chroma[:, t] for t in range(chroma.shape[1])]
