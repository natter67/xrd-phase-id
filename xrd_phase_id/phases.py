"""
Crystal structures of the phases most often found in high-entropy alloys
(HEAs), and "reference patterns" (stick patterns) computed from them with
pymatgen's XRDCalculator.

Background: what phases form in HEAs?
--------------------------------------
HEAs are alloys of ~5 elements in near-equal amounts.  The high
configurational entropy tends to stabilise simple *solid solutions*, where
every lattice site is randomly occupied by any of the elements:

* FCC (face-centred cubic, space group Fm-3m, prototype Cu).
  Example: the Cantor alloy CoCrFeMnNi, CoCrFeNi.  a ≈ 3.57–3.62 Å.
  Allowed reflections: h,k,l all odd or all even -> (111), (200), (220),
  (311), (222), ...

* BCC (body-centred cubic, Im-3m, prototype W).
  3d-metal HEAs with lots of Al (e.g. AlCoCrFeNi), a ≈ 2.86–2.92 Å; and
  refractory HEAs (NbMoTaW, HfNbTaTiZr), a ≈ 3.2–3.4 Å.
  Allowed reflections: h+k+l even -> (110), (200), (211), (220), ...

* B2 (ordered BCC, Pm-3m, prototype CsCl / NiAl).
  Same atomic positions as BCC, but the corner site (0,0,0) is preferentially
  occupied by one kind of atom (e.g. Al) and the body centre (½,½,½) by
  another (Ni, Co, Fe).  Because the two sites now scatter differently,
  the BCC extinction rule h+k+l = even is broken and *superlattice*
  reflections (100), (111), (210), ... appear.  Their intensity is
  proportional to (f_A − f_B)^2 · S^2, where S is the long-range order
  parameter.  If the two sublattices have similar scattering factors, or
  order is only partial, the superlattice peaks are weak, and B2 becomes
  hard to tell from BCC.  This is the single most common ambiguity in
  Al-containing HEAs, and the pipeline flags it explicitly.

* Sigma (σ, P4_2/mnm, prototype FeCr, 30 atoms per tetragonal cell).
  A brittle topologically close-packed intermetallic that precipitates in
  Cr/Mo/V-rich HEAs after ageing at 600–900 °C.  a ≈ 8.8 Å, c ≈ 4.55 Å.
  Its large cell gives many closely-spaced reflections clustered around
  2θ ≈ 40–50° (Cu Kα), right where the strongest FCC/BCC peaks sit,
  so sigma + FCC mixtures produce heavy peak overlap.

How the X-ray scattering of a *disordered* site is modelled
------------------------------------------------------------
pymatgen accepts partially-occupied sites, e.g. {"Co":0.25,"Cr":0.25,...}.
The atomic scattering factor of such a site is the occupancy-weighted
average f = Σ c_i f_i  (the "virtual crystal" / average-atom approximation).
That is exactly what a laboratory XRD pattern "sees" for a random solid
solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

from .constants import CU_KA1

# -----------------------------------------------------------------------------
# Representative compositions (atomic fractions) for each phase.
# -----------------------------------------------------------------------------
FCC_COMPOSITION = {"Co": 0.25, "Cr": 0.25, "Fe": 0.25, "Ni": 0.25}  # CoCrFeNi
BCC_COMPOSITION = {"Al": 0.2, "Co": 0.2, "Cr": 0.2, "Fe": 0.2, "Ni": 0.2}
REFRACTORY_BCC_COMPOSITION = {"Nb": 0.25, "Mo": 0.25, "Ta": 0.25, "W": 0.25}
# B2 sublattices: "A" (corner) is Al-rich, "B" (body centre) is (Ni,Co,Fe)-rich
B2_A_ELEMENTS = {"Al": 1.0}
B2_B_ELEMENTS = {"Ni": 0.5, "Co": 0.25, "Fe": 0.25}
# Sigma is Cr-rich; Cr/Fe/Co/Ni have similar scattering factors, so the
# exact site preference barely changes the XRD pattern.
SIGMA_COMPOSITION = {"Cr": 0.5, "Fe": 0.2, "Co": 0.15, "Ni": 0.15}

# Sigma phase atomic positions (FeCr prototype, Yakel, Acta Cryst. B39, 20
# (1983)).  Five Wyckoff sites generate the 30 atoms of the unit cell.
SIGMA_WYCKOFF = [
    [0.0, 0.0, 0.0],  # 2a
    [0.3981, 0.3981, 0.0],  # 4f
    [0.4632, 0.1316, 0.0],  # 8i (1)
    [0.7376, 0.0653, 0.0],  # 8i (2)
    [0.1823, 0.1823, 0.2524],  # 8j
]
SIGMA_C_OVER_A = 4.56 / 8.80  # ~0.518, fairly constant across sigma phases


# -----------------------------------------------------------------------------
# Structure builders
# -----------------------------------------------------------------------------
def make_fcc(a: float, composition: dict | None = None) -> Structure:
    """Random FCC solid solution: one Wyckoff site 4a at (0,0,0)."""
    comp = composition or FCC_COMPOSITION
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [comp], [[0, 0, 0]])


def make_bcc(a: float, composition: dict | None = None) -> Structure:
    """Random BCC solid solution: one Wyckoff site 2a at (0,0,0)."""
    comp = composition or BCC_COMPOSITION
    return Structure.from_spacegroup("Im-3m", Lattice.cubic(a), [comp], [[0, 0, 0]])


def make_b2(a: float, order: float = 1.0) -> Structure:
    """
    B2 (CsCl-type) ordered BCC with long-range order parameter S = `order`.

    S = 1: perfect order (all A atoms on the corner site).
    S = 0: completely random -> identical to BCC (superlattice peaks vanish).

    For a 50/50 A/B alloy the site occupancies are
        corner site:      A: (1+S)/2 ,  B: (1−S)/2
        body-centre site: A: (1−S)/2 ,  B: (1+S)/2
    """
    S = float(np.clip(order, 0.0, 1.0))
    site_a, site_b = {}, {}
    for el, x in B2_A_ELEMENTS.items():
        site_a[el] = site_a.get(el, 0) + x * (1 + S) / 2
        site_b[el] = site_b.get(el, 0) + x * (1 - S) / 2
    for el, x in B2_B_ELEMENTS.items():
        site_a[el] = site_a.get(el, 0) + x * (1 - S) / 2
        site_b[el] = site_b.get(el, 0) + x * (1 + S) / 2
    site_a = {k: v for k, v in site_a.items() if v > 1e-6}
    site_b = {k: v for k, v in site_b.items() if v > 1e-6}
    return Structure.from_spacegroup(
        "Pm-3m", Lattice.cubic(a), [site_a, site_b], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


def make_sigma(a: float, c: float | None = None, composition: dict | None = None) -> Structure:
    """Tetragonal sigma phase (P4_2/mnm, 30 atoms/cell)."""
    c = c if c is not None else a * SIGMA_C_OVER_A
    comp = composition or SIGMA_COMPOSITION
    return Structure.from_spacegroup(
        "P4_2/mnm", Lattice.tetragonal(a, c), [comp] * 5, SIGMA_WYCKOFF
    )


# -----------------------------------------------------------------------------
# Lorentz-polarisation factor
# -----------------------------------------------------------------------------
def lorentz_polarization(two_theta_deg):
    """
    Lorentz-polarisation (LP) factor for an unpolarised lab source,
    no monochromator (the same expression pymatgen uses):

        LP(θ) = (1 + cos² 2θ) / (sin² θ · cos θ)

    * (1 + cos² 2θ)/2 is the polarisation factor: X-rays scattered at 90°
      lose the component of polarisation that lies in the scattering plane.
    * 1/(sin²θ cosθ) is the Lorentz factor: a geometric term reflecting how
      long each reciprocal-lattice point spends in the diffraction condition
      and what fraction of the Debye ring the detector intercepts.

    It makes low-angle peaks much stronger than high-angle ones.
    """
    th = np.radians(np.asarray(two_theta_deg) / 2.0)
    return (1 + np.cos(2 * th) ** 2) / (np.sin(th) ** 2 * np.cos(th))


# -----------------------------------------------------------------------------
# Reference (stick) patterns for phase matching
# -----------------------------------------------------------------------------
@dataclass
class ReferencePhase:
    """
    A phase in the search library.

    The phase-matching step does not know the lattice parameter in advance
    (in an HEA it depends on composition), so we store the pattern at one
    reference lattice parameter and *rescale* it.  For an isotropic scaling
    of the cell by factor s (a -> s·a, c -> s·c), every d-spacing scales by s
    too, because d_hkl is a length in the crystal.  The structure-factor part
    of the intensity is nearly unchanged (atom positions are fractional), so
    we only need to recompute the angle-dependent LP factor.
    """

    name: str
    crystal_system: str  # "cubic" or "tetragonal"
    a_ref: float
    c_ref: float | None
    a_range: tuple[float, float]  # lattice-parameter search window (Å)
    d_ref: np.ndarray = field(repr=False)  # d-spacings at a_ref (Å)
    hkl: list = field(repr=False)  # representative (h,k,l) of each reflection
    struct_intensity: np.ndarray = field(repr=False)  # I / LP  (∝ m |F|²)
    superlattice: np.ndarray = field(repr=False)  # True for B2 order peaks

    def d_spacings(self, scale: float, c_scale: float = 1.0):
        """
        d-spacings for a = scale·a_ref and (tetragonal only) c = scale·c_scale·c_ref.

        Cubic:       1/d² = (h² + k² + l²) / a²
        Tetragonal:  1/d² = (h² + k²) / a²  +  l² / c²
        Allowing c/a to vary matters for sigma: its c/a ratio changes with
        composition by ~±1 %, which moves (00l)-type reflections by up to
        ~0.5° and would make a fixed-c/a reference miss them.
        """
        if self.crystal_system == "cubic" or c_scale == 1.0:
            return self.d_ref * scale
        hkl = np.array(self.hkl, float)
        a = self.a_ref * scale
        c = self.c_ref * scale * c_scale
        return 1.0 / np.sqrt((hkl[:, 0] ** 2 + hkl[:, 1] ** 2) / a**2 + hkl[:, 2] ** 2 / c**2)

    def pattern(self, scale: float, two_theta_range=(20.0, 100.0), wavelength=CU_KA1,
                c_scale: float = 1.0):
        """
        Stick pattern (2θ positions, relative intensities 0–100, indices into
        the reflection list) for a cell scaled by `scale` relative to a_ref.
        """
        d = self.d_spacings(scale, c_scale)
        s = wavelength / (2 * d)
        ok = s < 1.0  # sinθ must be < 1 for the reflection to exist
        tt = np.full_like(d, np.nan)
        tt[ok] = 2 * np.degrees(np.arcsin(s[ok]))
        inside = ok & (tt >= two_theta_range[0]) & (tt <= two_theta_range[1])
        idx = np.where(inside)[0]
        inten = self.struct_intensity[idx] * lorentz_polarization(tt[idx])
        if inten.size:
            inten = 100 * inten / inten.max()
        return tt[idx], inten, idx


def _build_reference(name, structure, crystal_system, a_ref, c_ref, a_range, superlattice_rule=None):
    """Run pymatgen once over a wide 2θ window and store d, hkl, I/LP."""
    calc = XRDCalculator(wavelength=CU_KA1)
    # A wide window (10–150°) so that rescaling never pushes a reflection
    # that "should" be in 20–100° outside the stored list.
    pat = calc.get_pattern(structure, scaled=False, two_theta_range=(10, 150))
    tt = np.array(pat.x)
    inten = np.array(pat.y)
    d = np.array(pat.d_hkls)
    hkls = [tuple(h[0]["hkl"]) for h in pat.hkls]
    struct_i = inten / lorentz_polarization(tt)
    # pymatgen reports hexagonal-style 4-index hkl only for hexagonal cells;
    # for our cubic/tetragonal cells hkl is 3-index.
    sl = np.array([superlattice_rule(h) if superlattice_rule else False for h in hkls])
    keep = inten > 1e-3 * inten.max()  # drop numerically-zero reflections
    return ReferencePhase(
        name=name,
        crystal_system=crystal_system,
        a_ref=a_ref,
        c_ref=c_ref,
        a_range=a_range,
        d_ref=d[keep],
        hkl=[h for h, k in zip(hkls, keep) if k],
        struct_intensity=struct_i[keep],
        superlattice=sl[keep],
    )


def _is_b2_superlattice(hkl):
    """In B2, reflections with h+k+l odd are forbidden for BCC -> 'order' peaks."""
    return (sum(abs(x) for x in hkl) % 2) == 1


_LIBRARY_CACHE: dict | None = None


def reference_library() -> dict[str, ReferencePhase]:
    """
    The candidate phases the pipeline searches over, with search windows
    for the lattice parameter that cover typical HEA values.

    Note the BCC window is deliberately wide (2.84–3.42 Å) to include both
    Al-containing 3d HEAs and refractory HEAs.  A wider window means more
    chances of an accidental match, which the scoring has to guard against.
    """
    global _LIBRARY_CACHE
    if _LIBRARY_CACHE is None:
        _LIBRARY_CACHE = {
            "FCC": _build_reference("FCC", make_fcc(3.59), "cubic", 3.59, None, (3.52, 3.70)),
            "BCC": _build_reference("BCC", make_bcc(2.88), "cubic", 2.88, None, (2.84, 3.42)),
            "B2": _build_reference(
                "B2", make_b2(2.88, 1.0), "cubic", 2.88, None, (2.84, 2.98), _is_b2_superlattice
            ),
            "sigma": _build_reference(
                "sigma", make_sigma(8.80), "tetragonal", 8.80, 8.80 * SIGMA_C_OVER_A, (8.65, 9.00)
            ),
        }
    return _LIBRARY_CACHE
