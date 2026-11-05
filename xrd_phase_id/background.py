"""
Step 1 — Background subtraction with the SNIP algorithm.

Why is there a background at all?
  * Air scatter of the primary beam (strong at low 2θ),
  * X-ray fluorescence from the sample — a big deal for HEAs: Cu Kα
    (8.05 keV) sits just above the K absorption edges of Fe (7.11 keV),
    Co (7.71 keV) and Cr (5.99 keV), so these elements fluoresce and raise
    the background a lot unless the detector discriminates energies,
  * Diffuse scattering (thermal, short-range order, amorphous oxide films),
  * Sample holder.

Background varies slowly with 2θ; Bragg peaks are narrow.  SNIP
(Statistics-sensitive Non-linear Iterative Peak-clipping; Ryan et al.,
Nucl. Instrum. Methods B34, 396 (1988)) exploits exactly that:

    for window p = 1 … m:
        y[i] <- min( y[i],  (y[i-p] + y[i+p]) / 2 )

At each point, if the value is higher than the average of two neighbours p
points away, it is "clipped" down to that average.  Peaks narrower than the
final window m get shaved off; slowly varying background survives.

Before clipping we apply the "LLS" transform  v = log(log(√(y+1)+1)+1).
It compresses the dynamic range so that strong peaks and weak background are
treated more evenly, and it makes the procedure robust to Poisson noise.

Choosing the window m is the one real decision: it must be larger than the
widest peak's *half-base-width*, but smaller than the background features.
Very broad peaks from nanocrystalline material (D < 10 nm, FWHM > 1°)
start to be partially eaten by SNIP.  The peak fit later adds a local
linear background term to absorb any residual.
"""

from __future__ import annotations

import numpy as np


def _lls(y):
    return np.log(np.log(np.sqrt(np.maximum(y, 0) + 1) + 1) + 1)


def _lls_inv(v):
    return (np.exp(np.exp(v) - 1) - 1) ** 2 - 1


def snip_background(y: np.ndarray, step_deg: float, window_deg: float = 2.5) -> np.ndarray:
    """
    Estimate the background of a pattern with SNIP.

    Parameters
    ----------
    y : counts
    step_deg : 2θ step size of the scan
    window_deg : maximum half-window (degrees 2θ); see module notes.
    """
    m = max(2, int(round(window_deg / step_deg)))
    # Clipping always follows the *lower* envelope of the noise, which would
    # bias the background low.  Lightly smoothing first (a 0.2° moving
    # average) removes most of that bias without touching the background
    # shape.
    k = max(3, int(round(0.2 / step_deg)) | 1)
    ys = np.convolve(np.pad(y.astype(float), k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    v = _lls(ys)
    n = v.size
    # Pad by reflection so the clipping is defined at the ends of the scan.
    vp = np.concatenate([v[m:0:-1], v, v[-2:-m - 2:-1]])
    # Decreasing-window variant (m -> 1) gives a smoother result than
    # increasing windows for the same m.
    for p in range(m, 0, -1):
        mid = vp[p:-p]
        avg = 0.5 * (vp[:-2 * p] + vp[2 * p:])
        vp[p:-p] = np.minimum(mid, avg)
    bg = _lls_inv(vp[m:m + n])
    return np.maximum(bg, 0.0)


def refine_background(x: np.ndarray, y: np.ndarray, peak_model: np.ndarray,
                      window_deg: float = 1.5) -> np.ndarray:
    """
    Second-pass background: smooth (data − fitted peaks).

    Why a second pass?  SNIP clips everything that rises above the average
    of its neighbours — including the top of a genuinely *concave* background
    feature such as an amorphous hump.  The clipping error is roughly
    ½ · B''(2θ) · p² for window p, i.e. the estimate sags below the true
    background wherever the background curves downward (we measured up to
    ~13 % on the benchmark).  A low background makes noise look like peaks.

    Once the Bragg peaks have been fitted we can subtract them and are left
    with "background + noise + whatever the fit missed".  That needs no
    clipping, just smoothing:
      1. running *median* over `window_deg` — robust to narrow leftovers
         (weak peaks that were not fitted) and to noise spikes,
      2. running *mean* over the same window — removes the median's
         staircase artefacts.
    Background and peaks are really one joint estimation problem; this
    alternating scheme (background → peaks → background → peaks) is a cheap
    way to approximate solving it jointly.
    """
    step = float(np.median(np.diff(x)))
    k = max(3, int(round(window_deg / step)) | 1)
    r = y - peak_model
    rp = np.pad(r, k // 2, mode="reflect")
    # running median via a strided window view (fine for a few thousand points)
    win = np.lib.stride_tricks.sliding_window_view(rp, k)
    med = np.median(win, axis=1)
    mp = np.pad(med, k // 2, mode="reflect")
    bg = np.convolve(mp, np.ones(k) / k, mode="valid")
    return np.maximum(bg, 0.0)
