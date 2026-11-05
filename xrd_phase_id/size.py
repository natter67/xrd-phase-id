"""
Step 6 — Crystallite size (Scherrer) and microstrain.

The Scherrer equation
---------------------
A crystal with only N lattice planes in a given direction does not give a
perfectly sharp Bragg reflection: the destructive interference that kills
intensity slightly away from the Bragg angle is incomplete when N is small.
The resulting broadening is

    β = K λ / (D cos θ)

  β : peak breadth in *radians of 2θ*, AFTER removing instrumental broadening
  D : volume-weighted crystallite (coherent-domain) size  — NOT the grain
      size you see in a micrograph; one grain can contain many domains
  K : shape factor ≈ 0.9 for FWHM
  λ : wavelength (Kα1, since we fitted the doublet)

Three practical caveats, all handled or flagged here:

1. Instrumental broadening must be removed first.  We know it from the
   Caglioti function (calibrated on a LaB6 standard).

2. Strain broadening is NOT size broadening.  Microstrain gives
   β = 4 ε tan θ — a different angular dependence and, physically, a more
   Gaussian shape.  Plugging total breadth into Scherrer silently
   reports strain as "small crystallites".

   We use the pseudo-Voigt mixing parameter η to split each fitted peak
   into Lorentzian and Gaussian FWHMs (Thompson–Cox–Hastings relations).
   Size broadening is taken from the Lorentzian part, strain from the
   Gaussian part after removing instrumental (Gaussian) width.  This is
   a "single-line" Voigt method (de Keijser et al., J. Appl. Cryst. 15, 308
   (1982)).  For comparison we also report the *naive* Scherrer size from
   the total instrument-corrected FWHM.

3. The method has a resolution window.  Above ~150–200 nm the size
   broadening is a small fraction of the instrumental width and is lost in
   the noise; we report a lower bound instead of a number.  Below ~3 nm the
   Scherrer approximation itself breaks down.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq

from .constants import CU_KA1, SCHERRER_K
from .simulate import instrumental_fwhm

SIZE_UPPER_LIMIT_NM = 200.0

# Thompson–Cox–Hastings: η as a function of q = H_L / H
_q_tab = np.linspace(0, 1, 2001)
_eta_tab = 1.36603 * _q_tab - 0.47719 * _q_tab**2 + 0.11116 * _q_tab**3


def split_voigt_widths(fwhm, eta):
    """
    Invert the TCH pseudo-Voigt relations to get the Lorentzian (H_L) and
    Gaussian (H_G) FWHMs of the equivalent Voigt profile:

        η = 1.36603 q − 0.47719 q² + 0.11116 q³ ,   q = H_L / H
        H⁵ = H_G⁵ + 2.69269 H_G⁴H_L + 2.42843 H_G³H_L² + 4.47163 H_G²H_L³
             + 0.07842 H_G H_L⁴ + H_L⁵
    """
    eta = float(np.clip(eta, 0, 1))
    q = float(np.interp(eta, _eta_tab, _q_tab))
    HL = q * fwhm

    def g(HG):
        return (HG**5 + 2.69269 * HG**4 * HL + 2.42843 * HG**3 * HL**2
                + 4.47163 * HG**2 * HL**3 + 0.07842 * HG * HL**4 + HL**5) - fwhm**5

    HG = 0.0 if g(0.0) >= 0 else brentq(g, 0.0, fwhm)
    return HL, HG


def scherrer_nm(beta_deg, two_theta_deg, wavelength=CU_KA1, K=SCHERRER_K):
    """D (nm) from a breadth in degrees 2θ."""
    beta = np.radians(beta_deg)
    th = np.radians(two_theta_deg / 2)
    with np.errstate(divide="ignore"):
        return K * (wavelength / 10) / (beta * np.cos(th))


@dataclass
class SizeResult:
    size_nm: float | None  # Lorentzian (size-only) Scherrer estimate, median of peaks
    size_iqr_nm: float | None  # spread between peaks
    size_naive_nm: float | None  # Scherrer from total instrument-corrected FWHM
    strain: float | None  # microstrain from the Gaussian part
    n_peaks: int
    lower_bound_only: bool = False

    def to_dict(self):
        return asdict(self)


def crystallite_size(peaks) -> SizeResult:
    """Scherrer size and strain for one phase from its (non-overlapped) peaks."""
    D_L, D_naive, eps = [], [], []
    for p in peaks:
        H_inst = float(instrumental_fwhm(p.center))
        HL, HG = split_voigt_widths(p.fwhm, p.eta)
        # Naive: treat everything except instrument as size; Gaussian-style
        # quadrature subtraction is the common textbook correction.
        beta_naive = np.sqrt(max(p.fwhm**2 - H_inst**2, 1e-8))
        D_naive.append(scherrer_nm(beta_naive, p.center))
        # Lorentzian part -> size.  (Our instrument model is purely Gaussian;
        # with a real instrument you would also subtract its Lorentzian part.)
        D_L.append(scherrer_nm(max(HL, 1e-6), p.center))
        HG_sample = np.sqrt(max(HG**2 - H_inst**2, 0.0))
        th = np.radians(p.center / 2)
        eps.append(np.radians(HG_sample) / (4 * np.tan(th)))
    if not D_L:
        return SizeResult(None, None, None, None, 0)
    D_L, D_naive = np.array(D_L), np.array(D_naive)
    med = float(np.median(D_L))
    lower = med > SIZE_UPPER_LIMIT_NM
    iqr = float(np.subtract(*np.percentile(D_L, [75, 25]))) if D_L.size > 1 else None
    return SizeResult(
        size_nm=min(med, SIZE_UPPER_LIMIT_NM),
        size_iqr_nm=iqr,
        size_naive_nm=float(min(np.median(D_naive), SIZE_UPPER_LIMIT_NM)),
        strain=float(np.median(eps)),
        n_peaks=len(peaks),
        lower_bound_only=lower,
    )
