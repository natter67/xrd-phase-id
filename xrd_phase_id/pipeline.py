"""
The full analysis pipeline:

    raw counts
      │ 1. SNIP background subtraction                (background.py)
      │ 2. peak finding vs. Poisson noise              (peaks.py)
      │ 3. pseudo-Voigt Kα1/Kα2 doublet fitting        (peaks.py)
      │ 4. phase search–match over lattice parameter   (matching.py)
      │ 5. Nelson–Riley lattice parameter              (lattice.py)
      │ 6. Scherrer crystallite size / microstrain     (size.py)
      ▼ 7. confidence score + review flags             (this file)

The most important design decision is in step 7: the pipeline is allowed —
and expected — to say "I'm not sure".  Every result carries a confidence in
[0, 1] and a list of flags.  A *review* flag means "a human should look at
this pattern"; an *info* flag is a caveat that does not stop the result
being used.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np

from .background import refine_background, snip_background
from .lattice import LatticeResult, cubic_lattice, tetragonal_lattice
from .matching import Hypothesis, enumerate_hypotheses
from .peaks import DETECTION_SIGMA, Peak, find_candidates, fit_peaks, full_model
from .phases import reference_library
from .constants import CU_KA1, SCHERRER_K
from .simulate import instrumental_fwhm
from .size import SizeResult, crystallite_size

AUTO_ACCEPT_CONFIDENCE = 0.6
MARGIN_SCALE = 0.08  # FOM difference giving ~63 % of full margin credit
MIN_SHARE_SENSITIVITY = 0.10  # flag patterns that could miss a >10 % minor phase
SENSITIVITY_SIZE_NM = 20.0  # assumed crystallite size of a hypothetical hidden phase
UNEXPLAINED_SHARE = 0.025  # unexplained peak area above this -> review


def _evidence(snrs):
    """
    Evidence that a phase is really present, from the peaks only it explains.

    Each isolated peak contributes min(1, (SNR − 3)/10): a peak at SNR 13 or
    more counts fully, one barely above the noise counts little.  Two
    strong isolated peaks give ~0.8, three give ~0.9.
    """
    e = sum(float(np.clip((s - 3) / 10, 0, 1)) for s in snrs)
    return 1 - np.exp(-e / 1.2)


@dataclass
class Flag:
    code: str
    severity: str  # "review" or "info"
    message: str


@dataclass
class PhaseResult:
    name: str
    confidence: float
    a: float
    a_err: float
    c: float | None
    c_err: float | None
    lattice_method: str
    a_mean_uncorrected: float | None
    n_matched_peaks: int
    n_unique_peaks: int
    intensity_share: float  # fraction of observed peak area assigned (not wt%!)
    size: SizeResult | None
    order_parameter: float | None = None  # B2 only (apparent)
    matched_hkl: list = field(default_factory=list)
    lattice_detail: LatticeResult | None = None


@dataclass
class AnalysisResult:
    status: str  # "AUTO_ACCEPT" or "NEEDS_REVIEW"
    confidence: float
    phases: list[PhaseResult]
    flags: list[Flag]
    alternatives: list[dict]
    peaks: list[Peak]
    explained_fraction: float
    confidence_terms: dict

    @property
    def phase_set(self):
        return tuple(sorted(p.name for p in self.phases))

    def to_dict(self):
        d = asdict(self)
        d["phase_set"] = list(self.phase_set)
        return d

    def to_json(self, **kw):
        return json.dumps(self.to_dict(), default=float, **kw)

    def summary(self) -> str:
        lines = [f"Status: {self.status}   confidence = {self.confidence:.2f}",
                 f"Phases: {' + '.join(self.phase_set) or '(none)'}   "
                 f"({self.explained_fraction:.0%} of peak area explained)"]
        for p in self.phases:
            s = f"  {p.name:6s} conf {p.confidence:.2f}  a = {p.a:.4f} ± {p.a_err:.4f} Å"
            if p.c is not None:
                s += f", c = {p.c:.4f} Å"
            s += f"  [{p.lattice_method}, {p.n_matched_peaks} peaks]"
            if p.size and p.size.size_nm:
                lb = ">" if p.size.lower_bound_only else ""
                s += f"  D = {lb}{p.size.size_nm:.0f} nm"
                if p.size.strain is not None:
                    s += f", ε = {p.size.strain:.1e}"
            if p.order_parameter is not None:
                s += f"  S_app = {p.order_parameter:.2f}"
            lines.append(s)
        for f in self.flags:
            lines.append(f"  [{f.severity.upper():6s}] {f.code}: {f.message}")
        for a in self.alternatives[:2]:
            lines.append(f"  alternative: {' + '.join(a['phases'])}  (FOM {a['fom']:.3f} "
                         f"vs best {a['best_fom']:.3f})")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
def analyze(two_theta: np.ndarray, counts: np.ndarray) -> AnalysisResult:
    x, y = np.asarray(two_theta, float), np.asarray(counts, float)
    step = float(np.median(np.diff(x)))
    tt_range = (float(x[0]), float(x[-1]))
    lib = reference_library()
    flags: list[Flag] = []

    # 1–3: background, peaks, profile fits ----------------------------------
    bg = snip_background(y, step)
    cands, sigma_s = find_candidates(x, y, bg)
    peaks = fit_peaks(x, y, bg, cands, sigma_s)
    # second pass: re-estimate background with the fitted peaks removed, then
    # find and fit peaks again on the improved background (see background.py)
    # SNIP's clipping can only pull the estimate *down*, so the true
    # background is (to within noise) never below it.  Taking the maximum
    # protects against the second pass over-subtracting a spurious broad peak.
    bg = np.maximum(bg, refine_background(x, y, full_model(x, peaks)))
    cands, sigma_s = find_candidates(x, y, bg)
    peaks = fit_peaks(x, y, bg, cands, sigma_s)

    def lod_area(tt, fwhm):
        """Smallest detectable Kα1 area at 2θ for a peak of width `fwhm`."""
        return DETECTION_SIGMA * float(np.interp(tt, x, sigma_s)) * fwhm * 1.2

    if len(peaks) < 2:
        flags.append(Flag("NO_PEAKS", "review", f"only {len(peaks)} significant peak(s) found"))
        return AnalysisResult("NEEDS_REVIEW", 0.0, [], flags, [], peaks, 0.0, {})

    # 4: search–match --------------------------------------------------------
    hyps, _ = enumerate_hypotheses(peaks, lod_area, tt_range)
    valid = [h for h in hyps if np.isfinite(h.fom)]
    if not valid:
        flags.append(Flag("NO_VALID_MATCH", "review", "no library phase explains the peaks"))
        return AnalysisResult("NEEDS_REVIEW", 0.0, [], flags, [], peaks, 0.0, {})
    best = valid[0]
    # Alternatives: best-scoring hypothesis for every *other* phase set.
    # For the confidence margin we only compare against "competing"
    # explanations — phase sets that are neither a subset nor a superset of
    # the best one.  (Whether a minor phase is really there, i.e. best vs. a
    # subset, is judged by that phase's own peak evidence instead; otherwise
    # every genuine 5 % minor phase would look "ambiguous" just because it
    # contributes only 5 % of the intensity.)
    alts = []
    seen = {best.names}
    for h in valid[1:]:
        if h.names not in seen:
            seen.add(h.names)
            nested = set(h.names) <= set(best.names) or set(h.names) >= set(best.names)
            alts.append(dict(phases=list(h.names), fom=h.fom, best_fom=best.fom,
                             explained=h.explained, nested=nested, _hyp=h))
    competing = [a for a in alts if not a["nested"]]
    alt_fom = competing[0]["fom"] if competing else -np.inf

    # How decisively does the data prefer `best` over the closest competitor?
    # Two routes to a decisive preference:
    #  (a) a large FOM gap (the competitor explains much less intensity), or
    #  (b) *discriminating peaks*: statistically strong peaks that best explains
    #      and the competitor cannot — e.g. one SNR-30 B2 (100) superlattice
    #      peak is decisive for B2 over BCC even though it carries only 2 % of
    #      the intensity.  Strong peaks that only the competitor explains count
    #      against best.
    if competing:
        alt_h = competing[0]["_hyp"]
        gap = best.fom - alt_h.fom
        only_best = [peaks[k].snr for k in best.explained_obs - alt_h.explained_obs]
        only_alt = [peaks[k].snr for k in alt_h.explained_obs - best.explained_obs]
        c_margin = float(max(1 - np.exp(-max(gap, 0) / MARGIN_SCALE),
                             _evidence(only_best) * (1 - _evidence(only_alt))))
    else:
        gap, c_margin = np.inf, 1.0
    for a in alts:
        a.pop("_hyp")

    # 5–6: lattice parameters and sizes per phase ------------------------------
    phase_results = []
    total_area = sum(p.area for p in peaks)
    for pm in best.phases:
        ref = lib[pm.name]
        # Peaks usable for lattice/size work: matched to this phase only, and
        # to exactly one of its predicted reflections (no hkl ambiguity).
        use = []
        for k in sorted(pm.unique_obs):
            lines = np.where(pm.pred_to_obs == k)[0]
            if lines.size == 1 or ref.crystal_system == "cubic":
                j = lines[np.argmax(pm.pred_I[lines])]
                use.append((peaks[k], tuple(ref.hkl[pm.pred_idx[j]])))
        if ref.crystal_system == "cubic":
            lat = cubic_lattice([p.center for p, _ in use], [p.center_err for p, _ in use],
                                [h for _, h in use])
        else:
            a0 = pm.candidate.a(ref)
            c0 = ref.c_ref * pm.candidate.scale * pm.candidate.c_scale
            lat = tetragonal_lattice([p.center for p, _ in use], [p.center_err for p, _ in use],
                                     [h for _, h in use], a0, c0)
        clean = [p for p, _ in use if not p.overlapped and p.snr >= 8]
        size = crystallite_size(clean if len(clean) >= 2 else [p for p, _ in use])

        share = sum(peaks[k].area for k in pm.matched_obs) / total_area
        # Per-phase confidence: evidence of its own × completeness
        c_phase = _evidence([peaks[k].snr for k in pm.unique_obs]) * (1 - pm.missing_frac)
        phase_results.append(PhaseResult(
            name=pm.name, confidence=float(c_phase), a=lat.a, a_err=lat.a_err, c=lat.c,
            c_err=lat.c_err, lattice_method=lat.method, a_mean_uncorrected=lat.a_mean,
            n_matched_peaks=len(pm.matched_obs), n_unique_peaks=len(pm.unique_obs),
            intensity_share=float(share), size=size, order_parameter=pm.order_estimate,
            matched_hkl=[h for _, h in use], lattice_detail=lat))

        # --- per-phase flags ---------------------------------------------------
        if pm.missing_frac > 0.3:
            flags.append(Flag("MISSING_REFLECTIONS", "review",
                              f"{pm.name}: {pm.missing_frac:.0%} of expected-visible intensity "
                              f"missing {pm.missing_lines[:4]} — texture, or wrong phase?"))
        if len(use) < 3:
            flags.append(Flag("FEW_PEAKS_FOR_LATTICE", "info",
                              f"{pm.name}: only {len(use)} clean peak(s); a = weighted mean, "
                              "no Nelson–Riley extrapolation"))
        if size.lower_bound_only:
            flags.append(Flag("SIZE_AT_RESOLUTION_LIMIT", "info",
                              f"{pm.name}: broadening ≈ instrumental; D > {size.size_nm:.0f} nm"))
        if size.strain is not None and size.strain > 1e-3:
            flags.append(Flag("STRAIN_BROADENING", "info",
                              f"{pm.name}: microstrain ≈ {size.strain:.1e}; naive Scherrer "
                              f"({size.size_naive_nm:.0f} nm) underestimates size"))

        # --- the BCC / B2 question ----------------------------------------------
        if pm.name == "BCC":
            b2 = lib["B2"]
            a_bcc = pm.candidate.a(ref)
            if b2.a_range[0] <= a_bcc <= b2.a_range[1]:
                tt_b2, I_b2, idx_b2 = b2.pattern(a_bcc / b2.a_ref, tt_range)
                sl = b2.superlattice[idx_b2]
                if sl.any():
                    j = np.argmax(np.where(sl, I_b2, -1))
                    fw = np.median([peaks[k].fwhm for k in pm.matched_obs])
                    # Superlattice intensity ∝ f_B2 · S²  (fraction of the BCC-type
                    # intensity coming from B2, times order parameter squared).
                    # The LOD of the strongest superlattice line therefore sets
                    # the largest f·S² that could hide in this pattern.
                    A_full = pm.scale_k * I_b2[j]  # (100) area if 100 % fully-ordered B2
                    fS2 = float(min(1.0, lod_area(tt_b2[j], fw) / max(A_full, 1e-9)))
                    if fS2 > 0.1:
                        flags.append(Flag(
                            "BCC_B2_AMBIGUOUS", "review",
                            f"no superlattice peaks, but any B2 with f·S² < {fS2:.2f} would be "
                            f"invisible here (e.g. {fS2:.0%} fully ordered B2, or all-B2 with "
                            f"S = {np.sqrt(fS2):.2f}); cannot exclude B2"))
                    else:
                        flags.append(Flag("BCC_NOT_B2", "info",
                                          f"superlattice peaks absent: B2 f·S² < {fS2:.2f}"))
        if pm.name == "B2" and pm.order_estimate is not None and pm.order_estimate < 0.6:
            flags.append(Flag(
                "BCC_B2_AMBIGUOUS", "review",
                f"weak superlattice peaks (apparent S = {pm.order_estimate:.2f}): partially "
                "ordered B2, or B2 + disordered BCC with similar a — intensities alone can't tell"))

    # --- sample-level flags ----------------------------------------------------
    unexpl_area = sum(peaks[k].area for k in best.unexplained_obs) / total_area
    strong_unexpl = [peaks[k] for k in best.unexplained_obs if peaks[k].snr > 15]
    if unexpl_area > UNEXPLAINED_SHARE or strong_unexpl:
        where = ", ".join(f"{p.center:.2f}°" for p in
                          sorted((peaks[k] for k in best.unexplained_obs), key=lambda p: -p.area)[:4])
        flags.append(Flag("UNEXPLAINED_PEAKS", "review",
                          f"{unexpl_area:.1%} of peak area unexplained (at {where}) — "
                          "additional/unknown phase?"))
    margin = best.fom - alt_fom
    if competing and c_margin < 0.5:
        flags.append(Flag("AMBIGUOUS_ALTERNATIVE", "review",
                          f"{' + '.join(competing[0]['phases'])} fits almost as well "
                          f"(ΔFOM = {margin:.3f}, no decisive discriminating peaks)"))

    # --- sensitivity: how big a minor phase could hide in this pattern? ---------
    # A second phase is seen only if its strongest line beats the LOD.  If its
    # strongest line carries a fraction q of its total intensity, the
    # smallest detectable intensity share is ≈ LOD / (q · total area).
    # The hidden phase may well be finer-grained than the main one (e.g. fresh
    # precipitates), so we assume its peaks are at least as broad as those of
    # D = SENSITIVITY_SIZE_NM crystallites.  Sigma spreads its intensity over
    # ~20 lines (small q), so it is much harder to detect than FCC or BCC.
    fw_obs = float(np.median([p.fwhm for p in peaks]))
    sensitivity = {}
    for name, ref in lib.items():
        if name in best.names:
            continue
        tt_r, I_r, _ = ref.pattern(1.0, tt_range)
        q = I_r.max() / I_r.sum()
        t_max = float(tt_r[np.argmax(I_r)])
        fw_size = float(np.degrees(SCHERRER_K * CU_KA1 / 10 /
                                   (SENSITIVITY_SIZE_NM * np.cos(np.radians(t_max / 2)))))
        fw = max(fw_obs, float(np.hypot(fw_size, instrumental_fwhm(t_max))))
        sensitivity[name] = float(lod_area(t_max, fw) / (q * total_area))
    worst = max(sensitivity.values()) if sensitivity else 0.0
    if worst > MIN_SHARE_SENSITIVITY:
        flags.append(Flag("LOW_SENSITIVITY", "review",
                          f"a minor {max(sensitivity, key=sensitivity.get)} phase (D ≈ "
                          f"{SENSITIVITY_SIZE_NM:.0f} nm) with < {worst:.0%} of the diffracted "
                          "intensity could be missed; absence of other phases is not established"))
    if len(best.phases) > 1:
        for pm, pr in zip(best.phases, phase_results):
            strong_unique = sum(1 for k in pm.unique_obs if peaks[k].snr >= 5)
            shared_area = sum(peaks[k].area for k in pm.shared_obs)
            own_area = sum(peaks[k].area for k in pm.matched_obs)
            if strong_unique < 2 or (shared_area > 0.5 * own_area and strong_unique < 3):
                flags.append(Flag("OVERLAPPING_PHASES", "review",
                                  f"{pm.name} identified from {strong_unique} isolated peak(s); "
                                  f"{shared_area / max(own_area, 1e-9):.0%} of its intensity is "
                                  "in peaks shared with another phase"))
    bad_fit = [p for p in peaks if p.cluster_redchi2 > 2.0]
    if bad_fit:
        flags.append(Flag("POOR_PROFILE_FIT", "info",
                          f"{len(bad_fit)} peak(s) in clusters with χ²_red > 2 "
                          "(unresolved overlap or asymmetry?)"))

    # 7: overall confidence --------------------------------------------------------
    c_explain = float(np.clip((best.explained - 0.75) / 0.2, 0, 1))
    c_complete = float(1 - max(pm.missing_frac for pm in best.phases))
    c_evidence = float(min(_evidence([peaks[k].snr for k in pm.unique_obs]) for pm in best.phases))
    confidence = c_explain * c_complete * c_margin * c_evidence
    terms = dict(explained=c_explain, completeness=c_complete, margin=c_margin,
                 evidence=c_evidence, raw_margin=float(gap) if competing else None,
                 min_detectable_share=sensitivity)
    if confidence < AUTO_ACCEPT_CONFIDENCE:
        flags.append(Flag("LOW_CONFIDENCE", "review", f"confidence {confidence:.2f} < "
                          f"{AUTO_ACCEPT_CONFIDENCE}"))

    status = "NEEDS_REVIEW" if any(f.severity == "review" for f in flags) else "AUTO_ACCEPT"
    result = AnalysisResult(status, float(confidence), phase_results, flags, alts, peaks,
                            float(best.explained), terms)
    result.background = bg  # final background (kept out of to_dict(); handy for plotting)
    return result
