"""
Step 4 — Phase identification ("search–match") with confidence scores.

Classic search–match (e.g. Hanawalt) compares the observed d-spacings with
a database of fixed reference patterns.  That is fragile for HEAs because
the lattice parameter of each phase depends on composition, which is not
known in advance.  So instead, for each candidate phase we:

1. **Scan** the lattice parameter over a physically plausible window and
   find values where the predicted stick pattern lines up with the observed
   peaks (a "soft" overlap score, so the scan surface is smooth).

2. **Refine** each promising value by least squares on matched peaks.  The
   predicted positions include a displacement-like term δ·cosθ, so the
   matching is not thrown off by sample-height error.

3. **Build hypotheses** — every single phase, and every pair of different
   phases — and score each with a figure of merit (FOM) that rewards
   explaining the observed intensity and penalises predicted peaks that
   should have been visible but weren't:

       FOM = (fraction of observed peak area explained)
             − 0.5 × (intensity-weighted fraction of *expected-visible*
                      predicted peaks that are missing)
             − 0.02 × (number of phases − 1)            (Occam penalty)

   "Expected-visible" uses the detection limit (LOD) measured from the
   noise in *this* pattern: a weak reflection of a 3 % minor phase is not
   expected to be seen, so its absence is not held against the hypothesis.

   A phase may only be part of a hypothesis if it has evidence of its own:
   at least two matched peaks, at least one that no other phase in the
   hypothesis can explain, and — for B2 — at least one superlattice peak.
   (Without a superlattice peak, B2 is just BCC.)

4. **Confidence and flags.**  Instead of forcing an answer, the best
   hypothesis gets a confidence score in [0, 1] and a list of flags.
   "Blocking" flags send the sample to human review.

Why a margin term?  If the best and second-best hypotheses (with
*different* phase sets) score almost the same, the data simply don't
discriminate between them — and that is precisely when an automated system
should ask a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np

from .phases import ReferencePhase, reference_library
from .peaks import Peak

# Weights in the figure of merit and confidence (see module docstring)
MISSING_WEIGHT = 0.5
OCCAM_PENALTY = 0.02
MAX_SHIFT_DEG = 0.15  # bound on the δ·cosθ displacement-like term


def match_tolerance(fwhm):
    """Position tolerance (deg 2θ) for calling a predicted line 'matched'."""
    return np.maximum(0.07, 0.3 * np.asarray(fwhm))


@dataclass
class PhaseCandidate:
    name: str
    scale: float  # a / a_ref
    delta: float  # displacement-like shift coefficient (deg)
    scan_score: float
    c_scale: float = 1.0  # (c/a) / (c/a)_ref, tetragonal phases only

    def a(self, ref: ReferencePhase):
        return ref.a_ref * self.scale


@dataclass
class PhaseMatch:
    """How one phase of a hypothesis explains the observed peaks."""

    name: str
    candidate: PhaseCandidate
    pred_tt: np.ndarray
    pred_I: np.ndarray
    pred_idx: np.ndarray  # indices into ReferencePhase reflection list
    pred_to_obs: np.ndarray  # matched observed peak per predicted line, or -1
    matched_obs: set = field(default_factory=set)
    unique_obs: set = field(default_factory=set)
    shared_obs: set = field(default_factory=set)
    scale_k: float = 0.0  # observed area per unit predicted intensity
    missing_frac: float = 0.0
    missing_lines: list = field(default_factory=list)
    superlattice_matched: int = 0
    order_estimate: float | None = None  # apparent B2 order parameter
    valid: bool = True
    why_invalid: str = ""


@dataclass
class Hypothesis:
    phases: list[PhaseMatch]
    explained: float
    fom: float
    unexplained_obs: list[int]
    explained_obs: set = field(default_factory=set)

    @property
    def names(self):
        return tuple(sorted(p.name for p in self.phases))

    @property
    def valid(self):
        return all(p.valid for p in self.phases)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _predict(ref: ReferencePhase, scale, delta, tt_range, min_I=0.5, c_scale=1.0):
    tt, I, idx = ref.pattern(scale, tt_range, c_scale=c_scale)
    keep = I >= min_I
    tt, I, idx = tt[keep], I[keep], idx[keep]
    tt = tt + delta * np.cos(np.radians(tt / 2))
    return tt, I, idx


def _nearest(pred_tt, obs_x, obs_tol):
    """Index of the nearest observed peak within its tolerance, else -1."""
    if obs_x.size == 0:
        return np.full(pred_tt.size, -1)
    D = np.abs(pred_tt[:, None] - obs_x[None, :])
    j = np.argmin(D, axis=1)
    ok = D[np.arange(pred_tt.size), j] <= obs_tol[j]
    return np.where(ok, j, -1)


# -----------------------------------------------------------------------------
# 1–2: scan and refine
# -----------------------------------------------------------------------------
def scan_phase(ref: ReferencePhase, peaks: list[Peak], tt_range, n_best=3) -> list[PhaseCandidate]:
    obs_x = np.array([p.center for p in peaks])
    obs_A = np.array([p.area for p in peaks])
    obs_tol = match_tolerance([p.fwhm for p in peaks])
    if obs_x.size == 0:
        return []
    total = obs_A.sum()
    lo, hi = ref.a_range
    scales = np.arange(lo / ref.a_ref, hi / ref.a_ref, 0.0003)
    # For tetragonal phases also scan c/a (±1.2 %), keeping the best c/a for
    # every value of a.
    c_grid = np.linspace(0.988, 1.012, 7) if ref.crystal_system == "tetragonal" else [1.0]
    F = np.zeros(scales.size)
    C = np.ones(scales.size)
    for i, s in enumerate(scales):
        for cs in c_grid:
            tt, I, _ = _predict(ref, s, 0.0, tt_range, min_I=2.0, c_scale=cs)
            if tt.size == 0:
                continue
            D = np.abs(tt[:, None] - obs_x[None, :])
            g = np.exp(-0.5 * (D / (1.5 * obs_tol[None, :])) ** 2)  # soft matching
            w = np.sqrt(I)
            # fraction of the predicted pattern that is found ...
            frac_pred = np.sum(w * g.max(axis=1)) / w.sum()
            # ... times (root of) the fraction of observed intensity explained
            frac_obs = np.sum(obs_A * g.max(axis=0)) / total
            f = frac_pred * np.sqrt(frac_obs)
            if f > F[i]:
                F[i], C[i] = f, cs

    # local maxima of the scan curve, best first, well separated
    order = np.argsort(-F)
    chosen = []
    for i in order:
        if F[i] <= 0.05:
            break
        if 0 < i < F.size - 1 and not (F[i] >= F[i - 1] and F[i] >= F[i + 1]):
            continue
        if all(abs(scales[i] - scales[j]) > 0.004 for j in chosen):
            chosen.append(i)
        if len(chosen) >= n_best:
            break
    cands = [refine_candidate(ref, PhaseCandidate(ref.name, float(scales[i]), 0.0, float(F[i]),
                                                  float(C[i])), peaks, tt_range) for i in chosen]
    return cands


def refine_candidate(ref, cand: PhaseCandidate, peaks, tt_range, n_iter=3) -> PhaseCandidate:
    """
    Least-squares refinement of (scale, δ[, c/a]) on matched peaks.

    Linearised model: Bragg's law gives Δ(2θ) = −2 tanθ · Δd/d (radians).
    * Scaling the whole cell by s changes every d by Δln s, so
          ∂(2θ)/∂ ln s = −2 tanθ.
    * For a tetragonal cell, changing c alone changes d by
          Δ ln d = (l² d² / c²) · Δ ln c,   so  ∂(2θ)/∂ ln c = −2 tanθ · l²d²/c².
    * The displacement-like term contributes ∂(2θ)/∂δ = cosθ.
    """
    obs_x = np.array([p.center for p in peaks])
    obs_tol = match_tolerance([p.fwhm for p in peaks])
    obs_e = np.array([max(p.center_err, 0.003) for p in peaks])
    s, d, cs = cand.scale, cand.delta, cand.c_scale
    tetra = ref.crystal_system == "tetragonal"
    hkl = np.array(ref.hkl, float)
    for _ in range(n_iter):
        tt, I, idx = _predict(ref, s, d, tt_range, min_I=1.0, c_scale=cs)
        m = _nearest(tt, obs_x, obs_tol)
        ok = m >= 0
        if ok.sum() < 2:
            break
        th = np.radians(tt[ok] / 2)
        r = obs_x[m[ok]] - tt[ok]
        cols = [-2 * np.degrees(np.tan(th)), np.cos(th)]
        if tetra:
            dd = ref.d_spacings(s, cs)[idx[ok]]
            c = ref.c_ref * s * cs
            cols.append(-2 * np.degrees(np.tan(th)) * hkl[idx[ok], 2] ** 2 * dd**2 / c**2)
        J = np.vstack(cols).T
        w = 1 / obs_e[m[ok]] + np.sqrt(I[ok])  # emphasise strong lines
        n_par = 1 if ok.sum() < 3 else (2 if (not tetra or ok.sum() < 5) else 3)
        J = J[:, :n_par]  # too few lines: refine fewer parameters
        sol, *_ = np.linalg.lstsq(J * w[:, None], r * w, rcond=None)
        s *= np.exp(sol[0])
        if n_par > 1:
            d = float(np.clip(d + sol[1], -MAX_SHIFT_DEG, MAX_SHIFT_DEG))
        if n_par > 2:
            cs = float(np.clip(cs * np.exp(sol[2]), 0.98, 1.02))
    return PhaseCandidate(cand.name, float(s), float(d), cand.scan_score, float(cs))


# -----------------------------------------------------------------------------
# 3: hypothesis evaluation
# -----------------------------------------------------------------------------
def evaluate(cands: list[PhaseCandidate], peaks: list[Peak], lod_area, lib, tt_range) -> Hypothesis:
    """
    Score a hypothesis (a set of phase candidates).

    lod_area(two_theta, fwhm) -> smallest detectable peak area at that angle.
    """
    obs_x = np.array([p.center for p in peaks])
    obs_A = np.array([p.area for p in peaks])
    obs_snr = np.array([p.snr for p in peaks])
    obs_fw = np.array([p.fwhm for p in peaks])
    obs_tol = match_tolerance(obs_fw)
    total = obs_A.sum()

    matches = []
    owner: dict[int, set] = {}
    for c in cands:
        ref = lib[c.name]
        tt, I, idx = _predict(ref, c.scale, c.delta, tt_range, c_scale=c.c_scale)
        m = _nearest(tt, obs_x, obs_tol)
        pm = PhaseMatch(c.name, c, tt, I, idx, m)
        pm.matched_obs = set(int(k) for k in m[m >= 0])
        for k in pm.matched_obs:
            owner.setdefault(k, set()).add(len(matches))
        matches.append(pm)

    for i, pm in enumerate(matches):
        ref = lib[pm.name]
        pm.unique_obs = {k for k in pm.matched_obs if owner[k] == {i}}
        pm.shared_obs = pm.matched_obs - pm.unique_obs
        sl = ref.superlattice[pm.pred_idx]

        # Scale factor k: observed area per unit of predicted intensity.
        # Use fundamental (non-superlattice) lines, unique ones if possible.
        def ratios(mask_lines, prefer_unique=True):
            out = []
            for k in set(pm.pred_to_obs[mask_lines & (pm.pred_to_obs >= 0)]):
                if prefer_unique and k not in pm.unique_obs:
                    continue
                Ip = pm.pred_I[(pm.pred_to_obs == k) & mask_lines].sum()
                out.append(obs_A[k] / Ip)
            return out

        r = ratios(~sl) or ratios(~sl, prefer_unique=False)
        pm.scale_k = float(np.median(r)) if r else 0.0

        # B2: the superlattice/fundamental intensity ratio scales with S²,
        # so the apparent order parameter is S ≈ √(k_superlattice / k_fund).
        if sl.any():
            r_sl = ratios(sl, prefer_unique=False)
            pm.superlattice_matched = len(r_sl)
            k_sl = float(np.median(r_sl)) if r_sl else 0.0
            pm.order_estimate = float(np.sqrt(min(k_sl / pm.scale_k, 1.0))) if pm.scale_k > 0 else 0.0
        else:
            k_sl = pm.scale_k

        # Missing lines: those predicted above ~2× LOD that were not found.
        fw_ref = np.median(obs_fw[list(pm.matched_obs)]) if pm.matched_obs else 0.2
        th_ref = np.radians(np.median(obs_x[list(pm.matched_obs)]) / 2) if pm.matched_obs else 0.4
        exp_I, miss_I = 0.0, 0.0
        for j in range(pm.pred_tt.size):
            k_line = k_sl if sl[j] else pm.scale_k
            if sl[j] and pm.superlattice_matched == 0:
                continue  # no evidence of order at all -> handled by validity
            A_pred = k_line * pm.pred_I[j]
            fw = fw_ref * np.cos(th_ref) / np.cos(np.radians(pm.pred_tt[j] / 2))
            if A_pred > 2.0 * lod_area(pm.pred_tt[j], fw):
                exp_I += A_pred
                if pm.pred_to_obs[j] < 0:
                    miss_I += A_pred
                    pm.missing_lines.append(tuple(lib[pm.name].hkl[pm.pred_idx[j]]))
        pm.missing_frac = miss_I / exp_I if exp_I > 0 else 0.0

        # Validity: every phase needs evidence of its own.
        strong_unique = [k for k in pm.unique_obs if obs_snr[k] >= 5]
        if len(pm.matched_obs) < 2:
            pm.valid, pm.why_invalid = False, "fewer than 2 matched peaks"
        elif not strong_unique:
            pm.valid, pm.why_invalid = False, "no peak that only this phase explains"
        elif sl.any() and pm.superlattice_matched == 0:
            pm.valid, pm.why_invalid = False, "no superlattice peak (indistinguishable from BCC)"

    explained_obs = set(owner)
    explained = float(obs_A[list(explained_obs)].sum() / total) if total > 0 else 0.0
    mean_missing = float(np.mean([pm.missing_frac for pm in matches])) if matches else 1.0
    fom = explained - MISSING_WEIGHT * mean_missing - OCCAM_PENALTY * (len(matches) - 1)
    unexpl = [k for k in range(len(peaks)) if k not in explained_obs]
    hyp = Hypothesis(matches, explained, fom, unexpl, explained_obs)
    if not hyp.valid:
        hyp.fom = -np.inf
    return hyp


def enumerate_hypotheses(peaks, lod_area, two_theta_range, max_phases=2):
    lib = reference_library()
    cands = {name: scan_phase(ref, peaks, two_theta_range) for name, ref in lib.items()}
    hyps = []
    names = [n for n in lib if cands[n]]
    for r in range(1, max_phases + 1):
        for combo in combinations(names, r):
            for choice in product(*[cands[n] for n in combo]):
                hyps.append(evaluate(list(choice), peaks, lod_area, lib, two_theta_range))
    hyps.sort(key=lambda h: -h.fom)
    return hyps, cands
