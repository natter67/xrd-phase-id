"""
Step 5 — Precise lattice parameters by Nelson–Riley extrapolation.

From one peak, Bragg's law gives the d-spacing and, for a cubic crystal,

    d_hkl = λ / (2 sin θ)          a_hkl = d_hkl · √(h² + k² + l²)

Ideally every peak gives the same a.  In practice they don't, because of
*systematic* errors that depend on θ:

* specimen displacement  (surface off the focusing circle) Δd/d ∝ cos²θ/sinθ
* sample transparency / absorption (X-rays penetrate the sample)
* axial divergence, flat-specimen error, ...

Nelson & Riley (Proc. Phys. Soc. 57, 160 (1945)) showed that for
Debye–Scherrer/diffractometer geometry the combined error in a is close to
linear in

    f(θ) = ½ · ( cos²θ / sinθ  +  cos²θ / θ )         (θ in radians)

and *all* these errors vanish as θ -> 90° (f -> 0).  So we plot a_hkl
against f(θ), fit a straight line, and read off the intercept at f = 0.
That intercept is a far better estimate of the true a than the plain mean
of the a_hkl values — the benchmark quantifies how much better.

Which peaks to use?  Only peaks that are not overlapped by another phase:
an overlapped peak's position is pulled by its neighbour.  High-angle peaks
are the most valuable: (i) their f(θ) is small (close to the intercept) and
(ii) a given position error δθ produces a much smaller error in d, since
Δd/d = −cotθ · Δθ.

For the tetragonal sigma phase there is no single "a_hkl" per peak, so we
fit a, c and the displacement s simultaneously by least squares:
    2θ_obs = 2θ_calc(a, c) − 2 s cosθ / R
This is the same physics as Nelson–Riley (displacement ∝ cos θ in 2θ),
just written for two lattice parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from .constants import CU_KA1, GONIOMETER_RADIUS_MM


def nelson_riley(two_theta_deg):
    th = np.radians(np.asarray(two_theta_deg) / 2)
    return 0.5 * (np.cos(th) ** 2 / np.sin(th) + np.cos(th) ** 2 / th)


@dataclass
class LatticeResult:
    a: float
    a_err: float
    c: float | None = None
    c_err: float | None = None
    a_mean: float | None = None  # plain average of per-peak a (for comparison)
    slope: float | None = None  # NR slope (Å per unit f)
    n_peaks: int = 0
    method: str = ""
    # data for plotting
    f: list | None = None
    a_hkl: list | None = None
    a_hkl_err: list | None = None

    def to_dict(self):
        return asdict(self)


def cubic_lattice(two_theta, two_theta_err, hkls, wavelength=CU_KA1) -> LatticeResult:
    """Nelson–Riley extrapolation for a cubic phase."""
    tt = np.asarray(two_theta, float)
    th = np.radians(tt / 2)
    m = np.array([h * h + k * k + l * l for h, k, l in hkls], float)
    a_hkl = wavelength / (2 * np.sin(th)) * np.sqrt(m)
    # error propagation: δa/a = cotθ · δθ   (δθ = δ(2θ)/2, in radians)
    err = a_hkl / np.tan(th) * np.radians(np.maximum(np.asarray(two_theta_err, float), 0.002) / 2)
    f = nelson_riley(tt)
    n = tt.size
    a_mean = float(np.average(a_hkl, weights=1 / err**2))
    if n < 3 or np.ptp(f) < 0.3:
        # With < 3 points (or all peaks at similar angle) a straight-line fit
        # is not meaningful.  Fall back to the weighted mean and say so.
        return LatticeResult(a_mean, float(np.std(a_hkl) if n > 1 else err[0]), a_mean=a_mean,
                             n_peaks=n, method="weighted-mean (too few peaks for NR)",
                             f=f.tolist(), a_hkl=a_hkl.tolist(), a_hkl_err=err.tolist())
    # Weighted linear regression a_hkl = a0 + b f.
    # Per-peak statistical errors from the fit are often smaller than the
    # real scatter (residual systematic errors), so we scale the parameter
    # covariance by the reduced χ² when that exceeds 1.
    w = 1 / err**2
    A = np.vstack([np.ones(n), f]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a0, b = cov @ (A.T @ (w * a_hkl))
    res = a_hkl - (a0 + b * f)
    chi2r = float(np.sum(w * res**2) / max(1, n - 2))
    a0_err = float(np.sqrt(cov[0, 0] * max(1.0, chi2r)))
    return LatticeResult(float(a0), a0_err, a_mean=a_mean, slope=float(b), n_peaks=n,
                         method="Nelson-Riley", f=f.tolist(), a_hkl=a_hkl.tolist(),
                         a_hkl_err=err.tolist())


def tetragonal_lattice(two_theta, two_theta_err, hkls, a0, c0, wavelength=CU_KA1) -> LatticeResult:
    """Least-squares a, c (+ displacement) for a tetragonal phase."""
    tt = np.asarray(two_theta, float)
    err = np.maximum(np.asarray(two_theta_err, float), 0.005)
    H = np.array([h * h + k * k for h, k, l in hkls], float)
    L = np.array([l * l for h, k, l in hkls], float)

    def calc(p):
        a, c, s = p
        inv_d2 = H / a**2 + L / c**2
        sin_t = np.clip(wavelength * np.sqrt(inv_d2) / 2, 0, 1)
        t2 = 2 * np.degrees(np.arcsin(sin_t))
        return t2 + np.degrees(-2 * s * np.cos(np.radians(t2 / 2)) / GONIOMETER_RADIUS_MM)

    n = tt.size
    if n < 4:
        return LatticeResult(a0, np.nan, c0, np.nan, n_peaks=n, method="not refined (too few peaks)")
    res = least_squares(lambda p: (tt - calc(p)) / err, [a0, c0, 0.0],
                        bounds=([a0 * 0.97, c0 * 0.97, -0.5], [a0 * 1.03, c0 * 1.03, 0.5]))
    J = res.jac
    chi2r = float(np.sum(res.fun**2) / max(1, n - 3))
    cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2r)
    e = np.sqrt(np.abs(np.diag(cov)))
    return LatticeResult(float(res.x[0]), float(e[0]), float(res.x[1]), float(e[1]), n_peaks=n,
                         method="LSQ a,c + displacement")
