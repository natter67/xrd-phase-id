"""
Analyse your own measured pattern.

    python scripts/analyze_xy.py my_scan.xy            # two columns: 2θ  counts
    python scripts/analyze_xy.py my_scan.xy --json out.json

Requirements on the data (the pipeline's physics assumes them):
  * Cu Kα radiation (Kα1 + Kα2, no monochromator removing Kα2),
  * Bragg–Brentano geometry, counts (not counts/s) so Poisson statistics hold,
  * instrument broadening roughly matching the Caglioti U, V, W in
    xrd_phase_id/constants.py — measure yours on LaB6 and edit them, or the
    crystallite sizes will be wrong.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from xrd_phase_id.pipeline import analyze  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", help="write the full result as JSON")
    a = ap.parse_args()
    data = np.loadtxt(a.path, comments=("#", "'", "!"))
    x, y = data[:, 0], data[:, 1]
    res = analyze(x, y)
    print(res.summary())
    if a.json:
        with open(a.json, "w") as f:
            f.write(res.to_json(indent=1))


if __name__ == "__main__":
    main()
