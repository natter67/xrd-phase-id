"""
Synthetic XRD patterns that look like real lab data.

pymatgen's XRDCalculator gives us an *ideal* stick pattern: infinitely sharp
lines at the Bragg angles with kinematical intensities (|F|² × multiplicity
× Lorentz-polarisation, optionally × Debye–Waller).  A measured pattern
differs from that in many ways, and each one is something the analysis
pipeline must cope with:

1. Kα1/Kα2 doublet       – each reflection appears twice (see constants.py).
2. Peak broadening        – instrument + small crystallites + microstrain.
3. Specimen displacement  – sample surface not exactly on the goniometer
                            axis -> systematic, angle-dependent peak shift.
4. Background             – air scatter (rises at low angle), fluorescence
                            (Fe/Co/Cr fluoresce strongly under Cu radiation!),
                            amorphous oxide/holder hump.
5. Counting noise         – photon counts are Poisson-distributed.
6. Preferred orientation  – rolled/cast samples have texture, so relative
                            intensities deviate from the powder ideal.
7. Mixtures               – several phases, each with its own lattice
                            parameter, crystallite size and amount.

Broadening model
----------------
We generate each peak as a true *Voigt* profile, i.e. a convolution of a
Gaussian and a Lorentzian, because the physical broadening sources have
different shapes:

* Instrumental (Caglioti):  mostly Gaussian here.
      FWHM_inst² = U tan²θ + V tanθ + W
* Crystallite size (Scherrer): approximately Lorentzian.
      β_size  = K λ / (D cos θ)         (radians, FWHM)
  Smaller crystallites -> fewer lattice planes diffracting coherently ->
  broader peaks.  Note the 1/cosθ dependence.
* Microstrain (Stokes–Wilson): approximately Gaussian.
      β_strain = 4 ε tan θ             (radians, FWHM)
  A spread of d-spacings Δd/d = ε smears the peak; note the tanθ dependence
  — this different angle dependence is what lets Williamson–Hall analysis
  separate size from strain.

Gaussian widths add in quadrature, Lorentzian widths add linearly, and the
Voigt function combines the two.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from scipy.special import voigt_profile

from . import phases as ph
from .constants import (
    CAGLIOTI_U,
    CAGLIOTI_V,
    CAGLIOTI_W,
    CU_KA1,
    CU_KA2,
    GONIOMETER_RADIUS_MM,
    KA2_KA1_RATIO,
    SCHERRER_K,
    TWO_THETA_MAX,
    TWO_THETA_MIN,
    TWO_THETA_STEP,
)

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # Gaussian FWHM -> σ


def instrumental_fwhm(two_theta_deg, U=CAGLIOTI_U, V=CAGLIOTI_V, W=CAGLIOTI_W):
    """Caglioti instrumental FWHM in degrees 2θ."""
    t = np.tan(np.radians(np.asarray(two_theta_deg) / 2))
    return np.sqrt(np.maximum(U * t**2 + V * t + W, 1e-8))


def ka2_position(two_theta_ka1_deg):
    """
    Where the Kα2 twin of a Kα1 peak appears.  Same d-spacing, slightly longer
    wavelength, so from Bragg's law  sinθ2 = (λ2/λ1) sinθ1.
    """
    s1 = np.sin(np.radians(np.asarray(two_theta_ka1_deg) / 2))
    s2 = np.clip(s1 * CU_KA2 / CU_KA1, -1, 1)
    return 2 * np.degrees(np.arcsin(s2))


def displacement_shift(two_theta_deg, displacement_mm, radius_mm=GONIOMETER_RADIUS_MM):
    """
    Peak shift (degrees 2θ) caused by a sample surface sitting `s` mm below
    the focusing circle in Bragg–Brentano geometry:

        Δ2θ = −2 s cosθ / R      (radians)

    This is the dominant systematic error in lab lattice-parameter work.  It
    vanishes as θ -> 90° (cos θ -> 0), which is exactly why we extrapolate
    lattice parameters to θ = 90° with the Nelson–Riley function.
    """
    th = np.radians(np.asarray(two_theta_deg) / 2)
    return np.degrees(-2 * displacement_mm * np.cos(th) / radius_mm)


@dataclass
class PhaseSpec:
    """Ground truth for one phase in a synthetic sample."""

    name: str  # "FCC", "BCC", "B2", "sigma"
    a: float
    c: float | None = None
    fraction: float = 1.0  # volume fraction
    size_nm: float = 50.0  # crystallite size D
    strain: float = 0.0  # microstrain ε (Δd/d)
    order: float = 1.0  # B2 long-range order parameter S
    composition: str = "default"  # "default" or "refractory" (BCC only)

    def structure(self):
        if self.name == "FCC":
            return ph.make_fcc(self.a)
        if self.name == "BCC":
            comp = ph.REFRACTORY_BCC_COMPOSITION if self.composition == "refractory" else None
            return ph.make_bcc(self.a, comp)
        if self.name == "B2":
            return ph.make_b2(self.a, self.order)
        if self.name == "sigma":
            return ph.make_sigma(self.a, self.c)
        raise ValueError(self.name)


@dataclass
class SampleSpec:
    """Ground truth for a whole synthetic measurement."""

    sample_id: str
    kind: str  # e.g. "FCC", "FCC+BCC"
    phases: list[PhaseSpec]
    max_counts: float = 20000.0  # counts at the strongest peak
    background_level: float = 0.05  # flat background as a fraction of max
    displacement_mm: float = 0.0
    texture: float = 0.0  # σ of log-normal intensity perturbation
    debye_waller_B: float = 0.5  # isotropic B factor, Å²
    seed: int = 0
    notes: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def _phase_reflections(spec: PhaseSpec, B: float):
    """
    Use pymatgen's XRDCalculator to get Kα1 reflection positions and
    intensities for one phase.

    We ask for *unscaled* intensities and divide by V_cell²:  the diffracted
    power from a phase is proportional to (its volume fraction) × |F|²/V²
    (for the same total irradiated volume, a bigger unit cell means fewer
    cells).  This is the basis of quantitative (Rietveld/RIR) phase analysis.
    """
    st = spec.structure()
    els = {str(el) for el in st.composition.elements}
    calc = XRDCalculator(wavelength=CU_KA1, debye_waller_factors={e: B for e in els})
    pat = calc.get_pattern(st, scaled=False, two_theta_range=(TWO_THETA_MIN - 2, TWO_THETA_MAX + 2))
    inten = np.array(pat.y) / st.volume**2
    hkls = [tuple(h[0]["hkl"]) for h in pat.hkls]
    return np.array(pat.x), inten, hkls


def simulate(sample: SampleSpec, two_theta=None):
    """
    Build the full synthetic pattern for `sample`.

    Returns (two_theta, counts, truth) where `truth` holds the noise-free
    components, useful for checking each pipeline stage.
    """
    rng = np.random.default_rng(sample.seed)
    if two_theta is None:
        two_theta = np.arange(TWO_THETA_MIN, TWO_THETA_MAX + 1e-9, TWO_THETA_STEP)
    x = two_theta
    signal = np.zeros_like(x)
    sticks = []

    for spec in sample.phases:
        tt, inten, hkls = _phase_reflections(spec, sample.debye_waller_B)
        inten = inten * spec.fraction
        # Preferred orientation / grain statistics: each reflection's
        # intensity is perturbed by a random factor (log-normal so it stays
        # positive).  Real texture is correlated by crystal direction; this
        # simpler model is still enough to stress the matching step.
        if sample.texture > 0:
            inten = inten * rng.lognormal(0.0, sample.texture, size=inten.size)

        for t0, I0, hkl in zip(tt, inten, hkls):
            # 1) systematic shift from specimen displacement
            t1 = t0 + displacement_shift(t0, sample.displacement_mm)
            th = np.radians(t1 / 2)
            # 2) broadening components, converted from radians to degrees 2θ
            fw_inst = instrumental_fwhm(t1)
            fw_size = np.degrees(SCHERRER_K * (CU_KA1 / 10) / (spec.size_nm * np.cos(th)))
            fw_strain = np.degrees(4 * spec.strain * np.tan(th))
            fw_gauss = np.sqrt(fw_inst**2 + fw_strain**2)  # Gaussian widths: quadrature
            fw_lorentz = fw_size  # Lorentzian width
            # 3) Kα1 and Kα2 components
            for pos, w in ((t1, 1.0), (ka2_position(t1), KA2_KA1_RATIO)):
                prof = voigt_profile(x - pos, fw_gauss * FWHM_TO_SIGMA, fw_lorentz / 2)
                signal += I0 * w * prof
            sticks.append(dict(phase=spec.name, hkl=hkl, two_theta=t1, intensity=I0,
                               fwhm_gauss=fw_gauss, fwhm_lorentz=fw_lorentz))

    # Scale so the strongest point reaches `max_counts` (sets signal-to-noise)
    signal *= sample.max_counts / signal.max()

    # --- background --------------------------------------------------------
    bl = sample.background_level * sample.max_counts
    air = bl * 1.5 * np.exp(-(x - x[0]) / rng.uniform(6, 15))  # low-angle air scatter
    hump_c = rng.uniform(22, 35)  # amorphous oxide/holder
    hump = bl * rng.uniform(0.0, 0.8) * np.exp(-0.5 * ((x - hump_c) / rng.uniform(4, 8)) ** 2)
    slope = bl * rng.uniform(-0.3, 0.3) * (x - x.mean()) / (x[-1] - x[0])
    background = bl + air + hump + slope
    background = np.maximum(background, 0.2 * bl)

    expected = signal + background
    counts = rng.poisson(expected).astype(float)  # counting statistics

    truth = dict(signal=signal, background=background, sticks=sticks)
    return x, counts, truth


# -----------------------------------------------------------------------------
# Random sample generator for the benchmark
# -----------------------------------------------------------------------------
#  Sample types and how many of each go into the 200-sample benchmark.
BENCHMARK_MIX = {
    "FCC": 30,
    "BCC": 25,
    "B2": 25,
    "sigma": 10,
    "FCC+BCC": 30,
    "FCC+B2": 25,
    "FCC+sigma": 30,
    "BCC+B2": 25,
}


def _random_phase(name, rng, fraction):
    size = float(np.exp(rng.uniform(np.log(8), np.log(150))))  # log-uniform 8–150 nm
    strain = float(rng.choice([0.0, rng.uniform(0.0005, 0.003)], p=[0.5, 0.5]))
    if name == "FCC":
        return PhaseSpec("FCC", a=float(rng.uniform(3.565, 3.640)), fraction=fraction,
                         size_nm=size, strain=strain)
    if name == "BCC":
        if rng.random() < 0.25:  # refractory BCC HEA
            return PhaseSpec("BCC", a=float(rng.uniform(3.18, 3.40)), fraction=fraction,
                             size_nm=size, strain=strain, composition="refractory")
        return PhaseSpec("BCC", a=float(rng.uniform(2.865, 2.925)), fraction=fraction,
                         size_nm=size, strain=strain)
    if name == "B2":
        return PhaseSpec("B2", a=float(rng.uniform(2.865, 2.915)), fraction=fraction,
                         size_nm=size, strain=strain, order=float(rng.uniform(0.4, 1.0)))
    if name == "sigma":
        a = float(rng.uniform(8.75, 8.92))
        c = a * float(rng.uniform(0.514, 0.522))
        return PhaseSpec("sigma", a=a, c=c, fraction=fraction, size_nm=size, strain=strain)
    raise ValueError(name)


def random_sample(kind: str, seed: int, sample_id: str) -> SampleSpec:
    """Draw one random sample of the given kind with realistic parameter spreads."""
    rng = np.random.default_rng(seed)
    names = kind.split("+")
    if len(names) == 1:
        fracs = [1.0]
    else:
        minor = float(rng.uniform(0.03, 0.5))  # minor phase from 3 % to 50 %
        fracs = [1 - minor, minor]
    phases_ = [_random_phase(n, rng, f) for n, f in zip(names, fracs)]
    if kind == "BCC+B2":
        # The classic AlCoCrFeNi spinodal microstructure: disordered BCC and
        # ordered B2 with almost the same lattice parameter (misfit < 1 %).
        bcc, b2 = phases_
        bcc.composition = "default"
        bcc.a = float(b2.a * (1 + rng.uniform(-0.01, 0.01)))
    return SampleSpec(
        sample_id=sample_id,
        kind=kind,
        phases=phases_,
        max_counts=float(np.exp(rng.uniform(np.log(1500), np.log(60000)))),
        background_level=float(rng.uniform(0.02, 0.15)),
        displacement_mm=float(rng.uniform(-0.15, 0.15)),
        texture=float(rng.uniform(0.0, 0.25)),
        debye_waller_B=float(rng.uniform(0.3, 0.8)),
        seed=seed + 100000,
    )


def benchmark_samples(seed: int = 2026) -> list[SampleSpec]:
    samples = []
    i = 0
    for kind, n in BENCHMARK_MIX.items():
        for _ in range(n):
            samples.append(random_sample(kind, seed + i, f"S{i:03d}"))
            i += 1
    return samples
