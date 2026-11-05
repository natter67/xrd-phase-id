"""
Steps 2 and 3 — Peak finding and pseudo-Voigt profile fitting.

Peak finding
------------
After background subtraction the net signal is "peaks + noise".  To decide
what counts as a peak we need to know how big the noise is.  For photon
counting it is known *a priori*: counts are Poisson distributed, so the
standard deviation of a point with N counts is √N.  We

  1. smooth the net signal with a Savitzky–Golay filter (fits a local
     quadratic; preserves peak height and position better than a moving
     average),
  2. compute the noise level of the *smoothed* data, σ_s = g · √N, where
     g = √(Σ c_i²) is the noise-reduction factor of the filter coefficients,
  3. accept local maxima whose height and prominence both exceed
     DETECTION_SIGMA × σ_s.

The threshold sets the limit of detection (LOD).  A minor phase whose
strongest peak is below ~4σ above background is invisible to *any*
algorithm — this is a physical limit, not a software one, and the benchmark
shows that's where most minor-phase misses come from.

Kα2 handling: at high angle a sharp peak's Kα2 twin may be resolved and
picked up as a separate "peak".  We drop candidates that sit at the
Kα2 position of a stronger peak with roughly half its height.

Profile fitting
---------------
Each peak is modelled as a *pseudo-Voigt* doublet:

    pV(x) = A · [ η · L(x; x0, H) + (1 − η) · G(x; x0, H) ]

    G = (2/H) √(ln2/π) · exp(−4 ln2 (x−x0)²/H²)     unit-area Gaussian
    L = (2/(πH)) / (1 + 4 (x−x0)²/H²)               unit-area Lorentzian

with the same FWHM H for both parts and a mixing parameter η ∈ [0, 1].
pV is a cheap, very good approximation to the Voigt function (the true
convolution of Gaussian and Lorentzian broadening).  The fitted η tells us
how Lorentzian the peak is, which we later use to separate size broadening
(Lorentzian) from instrumental + strain broadening (Gaussian).

The Kα2 component is tied to the Kα1 one: position from Bragg's law,
same H and η, and half the area.  So each peak has only 4 free
parameters (x0, A, H, η), and fitting the doublet gives the *Kα1* position
directly — which is what we use for d-spacings.

Neighbouring peaks whose tails overlap are fitted *together* in one cluster
with a local linear background.  We fit by weighted least squares, with
weights 1/σ_i = 1/√N_i (Poisson), so that χ²_red ≈ 1 means the model
describes the data to within counting noise.

If the residuals still contain a significant bump (an unresolved shoulder,
e.g. a weak BCC (110) hiding under an FCC (111)), we try adding a peak there
and keep it only if it lowers the Bayesian Information Criterion
    BIC = χ² + k ln n.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths, savgol_coeffs, savgol_filter

from .simulate import ka2_position
from .constants import KA2_KA1_RATIO

DETECTION_SIGMA = 4.0  # detection threshold in units of the smoothed noise σ
SG_WINDOW = 7  # Savitzky–Golay window (points)
SG_ORDER = 2
MAX_FWHM_DEG = 2.5

_LN2 = np.log(2.0)


# -----------------------------------------------------------------------------
# Profile functions
# -----------------------------------------------------------------------------
def pseudo_voigt(x, x0, area, fwhm, eta):
    """Unit-area pseudo-Voigt scaled by `area`."""
    dx2 = ((x - x0) / fwhm) ** 2
    g = (2.0 / fwhm) * np.sqrt(_LN2 / np.pi) * np.exp(-4 * _LN2 * dx2)
    lor = (2.0 / (np.pi * fwhm)) / (1.0 + 4.0 * dx2)
    return area * (eta * lor + (1.0 - eta) * g)


def pv_doublet(x, x0, area, fwhm, eta):
    """Kα1 + Kα2 pseudo-Voigt pair; `area` and `x0` refer to the Kα1 line."""
    return pseudo_voigt(x, x0, area, fwhm, eta) + pseudo_voigt(
        x, ka2_position(x0), KA2_KA1_RATIO * area, fwhm, eta
    )


def pv_peak_height(area, fwhm, eta):
    """Maximum of a single pV component."""
    return area * (eta * 2 / (np.pi * fwhm) + (1 - eta) * (2 / fwhm) * np.sqrt(_LN2 / np.pi))


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class Peak:
    center: float  # Kα1 2θ (degrees)
    center_err: float
    area: float  # Kα1 integrated intensity (counts·deg)
    area_err: float
    fwhm: float  # degrees 2θ
    fwhm_err: float
    eta: float  # Lorentzian fraction of the pV
    eta_err: float
    height: float  # Kα1 peak height (counts)
    snr: float  # height / background noise
    cluster: int
    cluster_redchi2: float
    overlapped: bool = False  # another fitted peak within ~1 FWHM
    added_from_residual: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class Candidate:
    x: float
    height: float
    fwhm: float


# -----------------------------------------------------------------------------
# Peak finding
# -----------------------------------------------------------------------------
def find_candidates(x, y, bg):
    """Locate candidate peaks in background-subtracted data (see module notes)."""
    step = float(np.median(np.diff(x)))
    net = y - bg
    net_s = savgol_filter(net, SG_WINDOW, SG_ORDER)
    g = np.sqrt(np.sum(savgol_coeffs(SG_WINDOW, SG_ORDER) ** 2))  # noise reduction factor
    # Noise is set by the *total* counts at each point (signal + background),
    # but for the detection threshold we use the background level: this is
    # the noise a weak peak has to stand out against.
    sigma_s = g * np.sqrt(np.maximum(bg, 1.0))
    thr = DETECTION_SIGMA * sigma_s
    idx, _ = find_peaks(net_s, height=thr, prominence=thr)
    if idx.size == 0:
        return [], sigma_s
    widths = peak_widths(net_s, idx, rel_height=0.5)[0] * step
    cands = [Candidate(float(x[i]), float(net_s[i]), float(max(w, 2 * step)))
             for i, w in zip(idx, widths)]

    # --- remove resolved Kα2 twins ---------------------------------------
    cands.sort(key=lambda c: -c.height)
    kept: list[Candidate] = []
    for c in cands:
        is_twin = False
        for p in kept:
            dx = abs(c.x - ka2_position(p.x))
            if dx < max(0.06, 0.35 * p.fwhm) and c.height < 0.8 * p.height and c.x > p.x:
                is_twin = True
                break
        if not is_twin:
            kept.append(c)
    kept.sort(key=lambda c: c.x)
    return kept, sigma_s


# -----------------------------------------------------------------------------
# Profile fitting
# -----------------------------------------------------------------------------
def _clusters(cands, x_min, x_max):
    """Group candidates whose fitting windows overlap."""
    wins = []
    for c in cands:
        half = max(4.0 * c.fwhm, 0.5)
        lo, hi = c.x - half, ka2_position(c.x) + half
        wins.append([max(lo, x_min), min(hi, x_max), [c]])
    merged = []
    for w in wins:
        if merged and w[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], w[1])
            merged[-1][2].extend(w[2])
        else:
            merged.append(w)
    return merged


def _model(params, x, n_peaks, xc):
    b0, b1 = params[0], params[1]
    out = b0 + b1 * (x - xc)
    for k in range(n_peaks):
        x0, A, H, eta = params[2 + 4 * k: 6 + 4 * k]
        out = out + pv_doublet(x, x0, A, H, eta)
    return out


def _pv_and_derivs(x, c, A, H, eta):
    """
    One pV component and its analytic partial derivatives with respect to
    (centre, area, FWHM, η).  Supplying these to the optimiser (instead of
    letting it estimate them by finite differences) makes fitting ~10×
    faster for clusters with many peaks, e.g. sigma phase.
    With u = (x − c)/H:
        ∂G/∂c = G · 8 ln2 · u / H          ∂L/∂c = L · 8u / (H (1 + 4u²))
        ∂G/∂H = G · (8 ln2 u² − 1) / H     ∂L/∂H = L · (8u²/(1 + 4u²) − 1) / H
    """
    u = (x - c) / H
    G = (2.0 / H) * np.sqrt(_LN2 / np.pi) * np.exp(-4 * _LN2 * u**2)
    q = 1.0 + 4.0 * u**2
    L = (2.0 / (np.pi * H)) / q
    prof = eta * L + (1 - eta) * G
    dG_dc, dL_dc = G * 8 * _LN2 * u / H, L * 8 * u / (H * q)
    dG_dH, dL_dH = G * (8 * _LN2 * u**2 - 1) / H, L * (8 * u**2 / q - 1) / H
    return (A * prof,
            A * (eta * dL_dc + (1 - eta) * dG_dc),
            prof,
            A * (eta * dL_dH + (1 - eta) * dG_dH),
            A * (L - G))


def _jacobian(params, x, n_peaks, xc, sig):
    """∂(residual)/∂(params) for the cluster model; residual = (data − model)/σ."""
    J = np.empty((x.size, params.size))
    J[:, 0] = 1.0
    J[:, 1] = x - xc
    for k in range(n_peaks):
        x0, A, H, eta = params[2 + 4 * k: 6 + 4 * k]
        x2 = ka2_position(x0)
        # Differentiate sinθ2 = (λ2/λ1) sinθ1:  dθ2/dθ1 = (λ2/λ1) cosθ1/cosθ2 = tanθ2/tanθ1
        dx2 = np.tan(np.radians(x2 / 2)) / np.tan(np.radians(x0 / 2))
        _, c1, a1, h1, e1 = _pv_and_derivs(x, x0, A, H, eta)
        _, c2, a2, h2, e2 = _pv_and_derivs(x, x2, KA2_KA1_RATIO * A, H, eta)
        J[:, 2 + 4 * k] = c1 + c2 * dx2
        J[:, 3 + 4 * k] = a1 + KA2_KA1_RATIO * a2
        J[:, 4 + 4 * k] = h1 + h2
        J[:, 5 + 4 * k] = e1 + e2
    return -J / sig[:, None]


def _fit_cluster(x, net, sig, peaks_init, step):
    """Weighted least-squares fit of one cluster.  Returns (params, cov, chi2, n)."""
    xc = float(np.mean(x))
    p0 = [0.0, 0.0]
    lo = [-np.inf, -np.inf]
    hi = [np.inf, np.inf]
    for (x0, h, H) in peaks_init:
        eta0 = 0.5
        A0 = max(h, 1.0) * H * (eta0 * np.pi / 2 + (1 - eta0) * 1.0645) / (1 + KA2_KA1_RATIO * 0.7)
        p0 += [x0, A0, H, eta0]
        dx = max(0.5 * H, 3 * step)
        lo += [x0 - dx, 0.0, max(1.5 * step, 0.02), 0.0]
        # FWHM capped at MAX_FWHM_DEG: anything broader is background, not a
        # Bragg peak (even 8 nm crystallites give only ~1.6° at 2θ = 90°), and
        # allowing it lets "peaks" soak up background errors.
        hi += [x0 + dx, np.inf, MAX_FWHM_DEG, 1.0]
    p0 = np.clip(p0, np.array(lo) + 1e-12, np.array(hi) - 1e-12)
    n_peaks = len(peaks_init)

    def resid(p):
        return (net - _model(p, x, n_peaks, xc)) / sig

    res = least_squares(resid, p0, jac=lambda p: _jacobian(p, x, n_peaks, xc, sig),
                        bounds=(lo, hi), x_scale="jac", max_nfev=400)
    chi2 = float(np.sum(res.fun**2))
    dof = max(1, x.size - res.x.size)
    # Parameter covariance from the Jacobian:  C = (JᵀJ)⁻¹ · χ²_red
    # (scaling by χ²_red inflates errors when the model is imperfect).
    J = res.jac
    try:
        cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2 / dof)
    except np.linalg.LinAlgError:
        cov = np.full((res.x.size, res.x.size), np.nan)
    return res.x, cov, chi2, x.size, xc


def fit_peaks(x, y, bg, cands, sigma_s, max_additions=2):
    """
    Fit all candidate peaks with pseudo-Voigt doublets.  Returns a list of
    Peak objects sorted by 2θ.
    """
    step = float(np.median(np.diff(x)))
    net = y - bg
    sig = np.sqrt(np.maximum(y, 1.0))  # Poisson σ of every data point
    peaks: list[Peak] = []

    for ci, (lo, hi, members) in enumerate(_clusters(cands, x[0], x[-1])):
        m = (x >= lo) & (x <= hi)
        xs, ns, ss = x[m], net[m], sig[m]
        init = [(c.x, c.height, c.fwhm) for c in members]
        added = [False] * len(init)
        params, cov, chi2, n, xc = _fit_cluster(xs, ns, ss, init, step)

        # --- residual-driven search for hidden peaks ------------------------
        for _ in range(max_additions):
            r = (ns - _model(params, xs, len(init), xc)) / ss
            r_s = np.convolve(r, np.ones(5) / 5, mode="same") * np.sqrt(5)  # σ≈1 if pure noise
            j = int(np.argmax(r_s))
            if r_s[j] < 4.0:
                break
            H_med = float(np.median(params[4::4][: len(init)]))
            # restart from the current solution (x0, height, FWHM) + new peak
            trial = []
            for k in range(len(init)):
                x0, A, H, eta = params[2 + 4 * k: 6 + 4 * k]
                trial.append((x0, pv_peak_height(A, H, eta) * (1 + KA2_KA1_RATIO * 0.7), H))
            trial.append((float(xs[j]), float(r[j] * ss[j]), H_med))
            p2, c2, chi2_2, _, _ = _fit_cluster(xs, ns, ss, trial, step)
            # BIC test: 4 extra parameters must buy Δχ² > 4 ln n
            if chi2 - chi2_2 > 4 * np.log(n):
                init, params, cov, chi2 = trial, p2, c2, chi2_2
                added.append(True)
            else:
                break

        dof = max(1, n - params.size)
        redchi2 = chi2 / dof
        errs = np.sqrt(np.abs(np.diag(cov)))
        for k in range(len(init)):
            x0, A, H, eta = params[2 + 4 * k: 6 + 4 * k]
            ex0, eA, eH, eeta = errs[2 + 4 * k: 6 + 4 * k]
            h = pv_peak_height(A, H, eta)
            noise = float(np.sqrt(max(np.interp(x0, x, bg), 1.0)))
            peaks.append(Peak(float(x0), float(ex0), float(A), float(eA), float(H), float(eH),
                              float(eta), float(eeta), float(h), float(h / noise), ci,
                              float(redchi2), added_from_residual=added[k]))

    # Drop statistically insignificant fits (area < 3σ or low SNR)
    peaks = [p for p in peaks if p.area > 3 * p.area_err and p.snr > 3.0]
    peaks.sort(key=lambda p: p.center)

    # Overlap flag: two peaks closer than the larger of their FWHMs cannot be
    # cleanly separated (a Rayleigh-like resolution criterion).  Their areas,
    # widths and even positions are correlated and less trustworthy.
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        if b.center - a.center < max(a.fwhm, b.fwhm):
            a.overlapped = b.overlapped = True
    return peaks


def full_model(x, peaks):
    """Sum of all fitted doublets (without the local linear terms)."""
    out = np.zeros_like(x)
    for p in peaks:
        out += pv_doublet(x, p.center, p.area, p.fwhm, p.eta)
    return out
