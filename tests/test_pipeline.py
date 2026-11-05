"""
Basic checks of each pipeline stage against known ground truth.
Run with:  python -m pytest -q
"""

import numpy as np
import pytest

from xrd_phase_id.background import snip_background
from xrd_phase_id.lattice import cubic_lattice
from xrd_phase_id.peaks import _jacobian, _model, find_candidates, fit_peaks
from xrd_phase_id.phases import make_b2, make_bcc, reference_library
from xrd_phase_id.pipeline import analyze
from xrd_phase_id.simulate import PhaseSpec, SampleSpec, ka2_position, simulate
from xrd_phase_id.size import split_voigt_widths


def _sample(phases, **kw):
    kw.setdefault("max_counts", 20000)
    kw.setdefault("seed", 3)
    return SampleSpec("t", "+".join(p.name for p in phases), phases, **kw)


def test_ka2_is_on_high_angle_side_and_grows_with_angle():
    d30 = ka2_position(30.0) - 30.0
    d90 = ka2_position(90.0) - 90.0
    assert 0 < d30 < d90
    assert d90 == pytest.approx(0.26, abs=0.03)  # textbook value near 90°


def test_b2_with_zero_order_is_bcc():
    lib_b2 = reference_library()["B2"]
    assert lib_b2.superlattice.any()
    # S = 0 B2 has identical site occupancies to BCC: superlattice must vanish
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    calc = XRDCalculator(wavelength=1.540593)
    p = calc.get_pattern(make_b2(2.88, 0.0), two_theta_range=(20, 100))
    hkls = [tuple(h[0]["hkl"]) for h in p.hkls]
    assert all(sum(h) % 2 == 0 for h, y in zip(hkls, p.y) if y > 0.1)
    assert len(calc.get_pattern(make_bcc(2.88), two_theta_range=(20, 100)).x) >= 4


def test_background_recovers_truth():
    x, y, truth = simulate(_sample([PhaseSpec("FCC", 3.6)]))
    bg = snip_background(y, 0.02)
    rel = np.median(np.abs(bg - truth["background"]) / truth["background"])
    assert rel < 0.08


def test_peak_positions_within_hundredths_of_a_degree():
    x, y, truth = simulate(_sample([PhaseSpec("FCC", 3.6, size_nm=60)]))
    bg = snip_background(y, 0.02)
    c, s = find_candidates(x, y, bg)
    peaks = fit_peaks(x, y, bg, c, s)
    true_tt = [st["two_theta"] for st in truth["sticks"]]
    for t in true_tt:
        assert min(abs(p.center - t) for p in peaks) < 0.01
    assert len(peaks) == len(true_tt)  # no Kα2 ghosts


def test_analytic_jacobian_matches_finite_differences():
    x = np.linspace(80, 83, 200)
    sig = np.full_like(x, 3.0)
    p = np.array([1.0, 0.1, 81.0, 500, 0.3, 0.4])
    J = _jacobian(p, x, 1, 81.5, sig)
    for i in range(p.size):
        e = np.zeros_like(p)
        e[i] = 1e-6 * max(1, abs(p[i]))
        num = -(_model(p + e, x, 1, 81.5) - _model(p - e, x, 1, 81.5)) / (2 * e[i]) / sig
        assert np.allclose(J[:, i], num, atol=1e-4 * np.abs(num).max() + 1e-9)


def test_nelson_riley_removes_displacement_bias():
    # perfect FCC lines shifted by specimen displacement
    from xrd_phase_id.simulate import displacement_shift
    a = 3.6
    hkls = [(1, 1, 1), (2, 0, 0), (2, 2, 0), (3, 1, 1), (2, 2, 2)]
    tt = [2 * np.degrees(np.arcsin(1.540593 * np.sqrt(sum(i * i for i in h)) / (2 * a)))
          for h in hkls]
    tt = np.array(tt) + displacement_shift(np.array(tt), 0.15)
    res = cubic_lattice(tt, [0.002] * 5, hkls)
    assert abs(res.a - a) < 3e-4
    assert abs(res.a_mean - a) > abs(res.a - a)


def test_voigt_split_limits():
    HL, HG = split_voigt_widths(0.3, 1.0)
    assert HL == pytest.approx(0.3, rel=1e-3) and HG < 1e-3
    HL, HG = split_voigt_widths(0.3, 0.0)
    assert HL < 1e-6 and HG == pytest.approx(0.3, rel=1e-3)


@pytest.mark.parametrize("phase", [PhaseSpec("FCC", 3.59), PhaseSpec("BCC", 3.25),
                                   PhaseSpec("B2", 2.89, order=1.0)])
def test_clean_single_phase_is_auto_accepted(phase):
    x, y, _ = simulate(_sample([phase], max_counts=40000))
    res = analyze(x, y)
    assert res.phase_set == (phase.name,)
    assert res.status == "AUTO_ACCEPT"
    assert abs(res.phases[0].a - phase.a) < 2e-3


def test_unknown_phase_is_flagged_not_forced():
    # FCC + a strong extra peak that no library phase explains
    x, y, truth = simulate(_sample([PhaseSpec("FCC", 3.59)], max_counts=20000))
    y = y + np.random.default_rng(0).poisson(6000 * np.exp(-0.5 * ((x - 36.0) / 0.1) ** 2))
    res = analyze(x, y)
    assert res.status == "NEEDS_REVIEW"
    assert any(f.code == "UNEXPLAINED_PEAKS" for f in res.flags)
