# HarmonicDNA

Applies the Smith-Waterman local sequence alignment algorithm - normally used in bioinformatics to find similar regions in DNA strands - to chord progressions extracted from audio files. The result is a similarity score and a visual alignment showing which passages are most harmonically alike.

The idea is that chord sequences, like genetic sequences, can be compared for local similarity rather than requiring a global match. Two songs might share a bridge or chorus even if they are structurally different overall.

---

## How it works

1. Audio is loaded and a beat-synchronous chromagram is extracted using librosa
2. Chroma vectors are matched against 24 chord templates (12 roots x major/minor) via cosine similarity
3. The resulting chord sequence is run through Smith-Waterman alignment against a second song's sequence
4. A scoring matrix rewards same chords, related chords (parallel, relative, subdominant/dominant), and penalises gaps
5. Traceback recovers the highest-scoring local alignment

---

## Usage

```bash
pip install -r requirements.txt

# Compare two audio files
python -m harmonicdna.cli compare song_a.mp3 song_b.mp3

# Show detected chords only
python -m harmonicdna.cli chords song_a.mp3
```

---

## Scoring matrix

| Relationship | Score |
|---|---|
| Same chord | +2.0 |
| Parallel major/minor (same root) | +1.0 |
| Relative major/minor | +0.5 |
| Subdominant or dominant | +0.3 |
| Unrelated | -1.0 |
| Gap penalty | -0.5 |

---

## Testing

Install the dependencies and run the suite:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt pytest   # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest -v
```

Exercise the CLI directly with `python -m harmonicdna.cli --help`.

---

## Project structure

```
harmonicdna/
├── harmonicdna/
│   ├── chromagram.py       # beat-synchronous chroma extraction
│   ├── chord_detector.py   # template matching, smoothing, deduplication
│   ├── scoring.py          # 24x24 chord similarity matrix
│   ├── aligner.py          # Smith-Waterman DP + traceback
│   ├── visualiser.py       # HTML alignment table
│   └── cli.py
└── tests/
    ├── test_aligner.py
    ├── test_scoring.py
    └── test_chord_detector.py
```

---

## Stack

Python 3.10, librosa, NumPy, SciPy, Typer, Rich

Supports MP3, WAV, FLAC and any format librosa can decode.
