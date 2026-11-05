"""
Physical constants and instrument defaults used throughout the package.

A lab X-ray tube with a copper anode does not emit a single wavelength.
The characteristic "K-alpha" line is really a *doublet*:

    Kα1 : an electron drops from the L3 shell to the K shell  -> 1.540593 Å
    Kα2 : an electron drops from the L2 shell to the K shell  -> 1.544414 Å

Kα1 is about twice as intense as Kα2 (the L3 shell holds 4 electrons, L2
holds 2).  The two wavelengths are only 0.25 % apart, so at low angles the
two diffraction peaks overlap completely, but at high angles (2θ > ~60°)
you can see a "shoulder" or even a fully split second peak on the
high-angle side of every reflection.  Any serious peak-fitting code has to
model this doublet, otherwise it will mis-locate peaks and over-estimate
peak widths (and therefore under-estimate crystallite size).

pymatgen's "CuKa" wavelength (1.54184 Å) is a weighted average of the two
lines.  We do NOT use it: we simulate and fit the doublet explicitly.
"""

# --- Cu K-alpha doublet ------------------------------------------------------
CU_KA1 = 1.540593  # Å  (Hölzer et al., Phys. Rev. A 56, 4554 (1997))
CU_KA2 = 1.544414  # Å
KA2_KA1_RATIO = 0.5  # intensity ratio I(Kα2)/I(Kα1); 0.497–0.51 in practice

# --- Scherrer constant -------------------------------------------------------
# K depends on crystallite shape and on how "width" is defined.  For
# FWHM-based widths and roughly spherical/equiaxed crystallites K ≈ 0.89–0.94;
# 0.9 is the conventional textbook value.
SCHERRER_K = 0.9

# --- Default measurement range and step --------------------------------------
TWO_THETA_MIN = 20.0  # degrees
TWO_THETA_MAX = 100.0  # degrees
TWO_THETA_STEP = 0.02  # degrees, a typical Bragg–Brentano lab scan step

# --- Instrument (Bragg–Brentano geometry) -------------------------------------
# Goniometer radius R: needed to convert a physical sample-height error
# (specimen displacement, in mm) into a peak shift in 2θ.
GONIOMETER_RADIUS_MM = 240.0

# Instrumental peak width follows the Caglioti relation
#     FWHM_inst(θ)^2 = U tan^2θ + V tanθ + W      (degrees^2)
# These values describe a decent lab diffractometer (~0.05–0.08° FWHM).  In
# practice you measure them once on a line-profile standard such as NIST
# SRM 660 (LaB6), whose crystallites are so large and strain-free that all of
# its broadening is instrumental.
CAGLIOTI_U = 0.0040
CAGLIOTI_V = -0.0020
CAGLIOTI_W = 0.0030
