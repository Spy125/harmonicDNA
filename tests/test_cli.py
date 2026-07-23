"""CLI smoke tests.

These invoke the actual Typer commands against a short synthesised audio clip,
so a drift between the documented command names and the code fails a test rather
than only surfacing when a user runs the README example.
"""

import numpy as np
import pytest
import scipy.io.wavfile as wav
from typer.testing import CliRunner

from harmonicdna.cli import app

runner = CliRunner()

_SR = 22050
# Triads for a short C - G - Am - F progression.
_CHORDS = {
    "C": [261.63, 329.63, 392.00],
    "G": [392.00, 493.88, 587.33],
    "A": [220.00, 261.63, 329.63],
    "F": [349.23, 440.00, 523.25],
}


def _write_progression(path, names):
    parts = []
    for name in names:
        t = np.linspace(0, 1.0, int(_SR * 1.0), False)
        tone = sum(np.sin(2 * np.pi * f * t) for f in _CHORDS[name]) / 3
        parts.append(tone)
    signal = np.concatenate(parts)
    wav.write(str(path), _SR, (signal * 0.3 * 32767).astype(np.int16))


@pytest.fixture
def songs(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _write_progression(a, ["C", "G", "A", "F"])
    _write_progression(b, ["C", "G", "A", "F"])
    return a, b


def test_chords_command(songs, tmp_path):
    a, _ = songs
    result = runner.invoke(app, ["chords", str(a)])
    assert result.exit_code == 0
    # The synthesised progression should be recovered.
    assert "Cmaj" in result.stdout and "Fmaj" in result.stdout


def test_compare_identical_is_high(songs):
    a, b = songs
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 0
    assert "identical" in result.stdout.lower()
