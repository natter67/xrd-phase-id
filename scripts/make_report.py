"""
Turn results/benchmark.json into figures and results/BENCHMARK.md.

    python scripts/make_report.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from xrd_phase_id.pipeline import analyze  # noqa: E402
from xrd_phase_id.peaks import full_model  # noqa: E402
from xrd_phase_id.phases import reference_library  # noqa: E402
from xrd_phase_id.simulate import PhaseSpec, SampleSpec, simulate  # noqa: E402

RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")

# Fixed phase -> colour mapping (categorical slots in fixed order; the colour
# follows the phase in every figure).
COL = {"FCC": "#2a78d6", "BCC": "#eb6834", "B2": "#1baf7a", "sigma": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#8a8985"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 1.5,
    "legend.frameon": False,
})


def load():
    with open(os.path.join(RES, "benchmark.json")) as f:
        return [r for r in json.load(f) if "error" not in r]


def pct(x):
    return f"{100 * x:.0f}%" if np.isfinite(x) else "–"


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def fig_example():
    """A worked example: FCC + 12 % BCC with overlapping (111)/(110) peaks."""
    s = SampleSpec("example", "FCC+BCC",
                   [PhaseSpec("FCC", 3.598, fraction=0.88, size_nm=35, strain=0.0008),
                    PhaseSpec("BCC", 2.881, fraction=0.12, size_nm=20)],
                   max_counts=12000, background_level=0.08, displacement_mm=0.10,
                   texture=0.1, seed=11)
    x, y, truth = simulate(s)
    res = analyze(x, y)
    bg = res.background
    model = full_model(x, res.peaks) + bg

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                  gridspec_kw=dict(height_ratios=[4, 1], hspace=0.05))
    ax.plot(x, y, color=NEUTRAL, lw=0.8, label="counts")
    ax.plot(x, bg, color=INK2, lw=1.2, ls="--", label="background (SNIP + refinement)")
    ax.plot(x, model, color=INK, lw=1.0, label="pseudo-Voigt fit")
    lib = reference_library()
    ymax = y.max()
    for i, pr in enumerate(res.phases):
        ref = lib[pr.name]
        tt, I, _ = ref.pattern(pr.a / ref.a_ref, (x[0], x[-1]))
        base = -0.06 * ymax * (i + 1)
        ax.vlines(tt[I > 1], base, base + 0.04 * ymax, color=COL[pr.name], lw=2,
                  label=f"{pr.name} a = {pr.a:.4f} Å")
    ax.set_ylabel("counts")
    ax.set_ylim(-0.15 * ymax, 1.05 * ymax)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Worked example — status {res.status}, confidence {res.confidence:.2f}",
                 loc="left")
    # inset: the overlapping FCC(111) / BCC(110) region
    ins = ax.inset_axes([0.42, 0.42, 0.25, 0.5])
    m = (x > 42.5) & (x < 45.5)
    ins.plot(x[m], y[m], color=NEUTRAL, lw=0.8)
    ins.plot(x[m], model[m], color=INK, lw=1.0)
    ins.set_title("FCC(111) + BCC(110)", fontsize=8)
    ins.tick_params(labelsize=7)
    sig = np.sqrt(np.maximum(y, 1))
    axr.plot(x, (y - model) / sig, color=INK2, lw=0.6)
    axr.set_ylabel("resid./σ")
    axr.set_xlabel("2θ (degrees, Cu Kα)")
    axr.set_ylim(-6, 6)
    fig.savefig(os.path.join(FIG, "example_fit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Nelson–Riley plot for the FCC phase of the same sample
    pr = next(p for p in res.phases if p.name == "FCC")
    lat = pr.lattice_detail
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    f = np.array(lat.f)
    ax.errorbar(f, lat.a_hkl, yerr=lat.a_hkl_err, fmt="o", ms=6, color=COL["FCC"],
                ecolor=COL["FCC"], capsize=2, label="a from each peak")
    for fi, ai, h in zip(f, lat.a_hkl, pr.matched_hkl):
        # (222) and (311) are close in f, so put their labels on opposite sides
        dx = -30 if h == (2, 2, 2) else 8
        ax.annotate("(" + "".join(map(str, h)) + ")", (fi, ai), textcoords="offset points",
                    xytext=(dx, -12 if h == (1, 1, 1) else 4), fontsize=8, color=INK2)
    ff = np.linspace(0, f.max() * 1.05, 50)
    ax.plot(ff, lat.a + lat.slope * ff, color=INK, lw=1.2, label="linear fit")
    ax.axhline(3.598, color=INK2, ls=":", lw=1.2, label="true a = 3.5980 Å")
    ax.plot([0], [lat.a], "s", color=INK, ms=6)
    ax.annotate(f"extrapolated a = {lat.a:.4f} Å", (0, lat.a), xytext=(10, 8),
                textcoords="offset points", fontsize=9)
    ax.set_xlabel("Nelson–Riley function  ½(cos²θ/sinθ + cos²θ/θ)")
    ax.set_ylabel("a (Å)")
    ax.set_xlim(left=-0.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"Nelson–Riley extrapolation (mean of peaks: {lat.a_mean:.4f} Å)", loc="left")
    fig.savefig(os.path.join(FIG, "nelson_riley.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res


def fig_calibration(rows):
    conf = np.array([r["confidence"] for r in rows])
    corr = np.array([r["correct"] for r in rows])
    edges = np.array([0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001])
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    centers, acc, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            centers.append((lo + min(hi, 1)) / 2)
            acc.append(corr[m].mean())
            n.append(m.sum())
    widths = np.diff(edges) * 0.85
    ax.bar(centers, acc, width=widths[: len(centers)], color=COL["FCC"], edgecolor="#fcfcfb",
           linewidth=2)
    for c, a, k in zip(centers, acc, n):
        ax.text(c, a + 0.02, f"{a:.0%}\nn={k}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.plot([0, 1], [0, 1], color=INK2, ls=":", lw=1, label="perfect calibration")
    ax.axvline(0.6, color=INK2, lw=1, ls="--")
    ax.text(0.605, 0.04, "auto-accept\nthreshold", fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("confidence score")
    ax.set_ylabel("phase-set accuracy")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Is the confidence score meaningful?", loc="left")
    fig.savefig(os.path.join(FIG, "calibration.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def lattice_errors(rows):
    out = defaultdict(lambda: dict(nr=[], mean=[]))
    for r in rows:
        for p in r["phases"]:
            if "a_true" in p and p["a_mean"] is not None and p["method"] == "Nelson-Riley":
                out[p["name"]]["nr"].append(p["a"] - p["a_true"])
                out[p["name"]]["mean"].append(p["a_mean"] - p["a_true"])
    return out


def fig_lattice(errs):
    names = [n for n in ["FCC", "BCC", "B2"] if errs[n]["nr"]]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rng = np.random.default_rng(0)
    for i, n in enumerate(names):
        for j, (key, lab) in enumerate((("mean", "mean of peaks"), ("nr", "Nelson–Riley"))):
            v = np.abs(np.array(errs[n][key])) * 1000
            xpos = i * 3 + j + rng.uniform(-0.25, 0.25, v.size)
            ax.scatter(xpos, v, s=10, color=COL[n] if key == "nr" else NEUTRAL, alpha=0.7,
                       linewidths=0)
            ax.plot([i * 3 + j - 0.35, i * 3 + j + 0.35], [np.median(v)] * 2, color=INK, lw=2)
    ax.set_xticks([i * 3 + j for i in range(len(names)) for j in range(2)])
    ax.set_xticklabels([f"{n}\nmean" if j == 0 else f"{n}\nN–R" for n in names for j in range(2)],
                       fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("|a − a_true| (mÅ)")
    ax.set_title("Lattice-parameter error: plain mean vs. Nelson–Riley (bar = median)", loc="left")
    fig.savefig(os.path.join(FIG, "lattice_error.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def size_pairs(rows):
    out = []
    for r in rows:
        for p in r["phases"]:
            if "size_true" in p and p["size"] is not None and not p["size_lb"]:
                out.append((p["name"], p["size_true"], p["size"], p["size_naive"],
                            p["strain_true"]))
    return out


def fig_size(pairs):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, idx, title in ((axs[0], 3, "Naive Scherrer (total FWHM)"),
                           (axs[1], 2, "Scherrer on Lorentzian part (Voigt split)")):
        for strained, mk in ((False, "o"), (True, "^")):
            v = [p for p in pairs if (p[4] > 0) == strained]
            ax.scatter([p[1] for p in v], [p[idx] for p in v], s=14, marker=mk,
                       color=COL["BCC"] if strained else COL["FCC"], alpha=0.75, linewidths=0,
                       label="with microstrain" if strained else "strain-free")
        ax.plot([5, 200], [5, 200], color=INK2, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("true crystallite size (nm)")
        ax.set_title(title, loc="left")
    axs[0].set_ylabel("estimated size (nm)")
    axs[1].legend(fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIG, "size_parity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def minor_detection(rows):
    bins = [0.03, 0.08, 0.15, 0.3, 0.5]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if len(r["true_set"]) == 2 and r["sample"]["kind"] != "BCC+B2"
               and lo <= r["sample"]["phases"][1]["fraction"] < hi]
        found = [r["sample"]["phases"][1]["name"] in r["pred_set"] for r in sel]
        out.append((f"{lo:.0%}–{hi:.0%}", len(sel), np.mean(found) if sel else np.nan,
                    np.mean([r["status"] == "NEEDS_REVIEW" for r, f in zip(sel, found) if not f])
                    if sel and not all(found) else np.nan))
    return out


def fig_minor(det):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    labels = [d[0] for d in det]
    ax.bar(labels, [d[2] for d in det], color=COL["BCC"], edgecolor="#fcfcfb", linewidth=2,
           width=0.6)
    for i, d in enumerate(det):
        ax.text(i, d[2] + 0.02, f"{d[2]:.0%}\nn={d[1]}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_xlabel("minor-phase volume fraction")
    ax.set_ylabel("minor phase detected")
    ax.set_title("Detection of the minor phase in two-phase samples", loc="left")
    fig.savefig(os.path.join(FIG, "minor_detection.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
def describe(r):
    s = r["sample"]
    ph = ", ".join(f"{p['name']} {p['fraction']:.0%} a={p['a']:.3f} D={p['size_nm']:.0f}nm"
                   + (f" S={p['order']:.2f}" if p["name"] == "B2" else "") for p in s["phases"])
    return ph, f"{s['max_counts']:.0f}"


def main():
    os.makedirs(FIG, exist_ok=True)
    rows = load()
    N = len(rows)
    auto = [r for r in rows if r["status"] == "AUTO_ACCEPT"]
    rev = [r for r in rows if r["status"] == "NEEDS_REVIEW"]
    silent = [r for r in auto if not r["correct"]]
    caught = [r for r in rev if not r["correct"]]
    acc = np.mean([r["correct"] for r in rows])

    fig_example()
    fig_calibration(rows)
    errs = lattice_errors(rows)
    fig_lattice(errs)
    pairs = size_pairs(rows)
    fig_size(pairs)
    det = minor_detection(rows)
    fig_minor(det)

    L = []
    L.append("# Benchmark report (auto-generated by `scripts/make_report.py`)\n")
    L.append(f"{N} synthetic samples, seed 2026. Runtime median "
             f"{np.median([r['runtime_s'] for r in rows]):.1f} s/sample.\n")
    L.append("## Headline numbers\n")
    L.append("| metric | value |\n|---|---|")
    L.append(f"| phase-set accuracy, all samples (best guess) | {pct(acc)} ({sum(r['correct'] for r in rows)}/{N}) |")
    L.append(f"| auto-accepted | {len(auto)} ({pct(len(auto) / N)}) |")
    L.append(f"| accuracy of auto-accepted results | {pct(np.mean([r['correct'] for r in auto]))} |")
    L.append(f"| **silent failures** (auto-accepted but wrong) | **{len(silent)}** ({pct(len(silent) / N)} of all) |")
    L.append(f"| sent to review | {len(rev)} ({pct(len(rev) / N)}) |")
    L.append(f"| …of which the best guess was wrong (review justified) | {len(caught)} |")
    L.append(f"| …of which the best guess was right (cautious) | {len(rev) - len(caught)} |")
    L.append(f"| wrong answers caught by review | {len(caught)}/{len(caught) + len(silent)} "
             f"({pct(len(caught) / max(1, len(caught) + len(silent)))}) |\n")

    held = os.path.join(RES, "benchmark_seed7.json")
    if os.path.exists(held):
        with open(held) as f:
            hr = [r for r in json.load(f) if "error" not in r]
        L.append("### Development set vs. held-out set\n")
        L.append("Thresholds (review rules, confidence weights) were tuned while looking at the "
                 "development set (seed 2026). The held-out set (seed 7, same sample-type mix, "
                 "never used for tuning) is the honest estimate.\n")
        L.append("| | development (seed 2026) | held-out (seed 7) |\n|---|---|---|")
        for lab, fn in (
            ("best-guess accuracy", lambda R: pct(np.mean([r["correct"] for r in R]))),
            ("auto-accepted", lambda R: pct(np.mean([r["status"] == "AUTO_ACCEPT" for r in R]))),
            ("accuracy when auto-accepted", lambda R: pct(np.mean(
                [r["correct"] for r in R if r["status"] == "AUTO_ACCEPT"]))),
            ("silent failures", lambda R: str(sum(r["status"] == "AUTO_ACCEPT" and not r["correct"]
                                                  for r in R))),
            ("wrong answers caught by review", lambda R: pct(
                sum(r["status"] != "AUTO_ACCEPT" and not r["correct"] for r in R)
                / max(1, sum(not r["correct"] for r in R)))),
        ):
            L.append(f"| {lab} | {fn(rows)} | {fn(hr)} |")
        L.append("")
        hs = [r for r in hr if r["status"] == "AUTO_ACCEPT" and not r["correct"]]
        if hs:
            L.append("Held-out silent failures:\n")
            L.append("| id | truth | predicted | counts |\n|---|---|---|---|")
            for r in hs:
                ph, cts = describe(r)
                L.append(f"| {r['sample']['sample_id']} | {ph} | {' + '.join(r['pred_set'])} | {cts} |")
            L.append("")

    L.append("## By sample type (development set)\n")
    L.append("| type | n | best-guess accuracy | auto-accepted | accuracy when auto | silent failures |")
    L.append("|---|---|---|---|---|---|")
    kinds = list(dict.fromkeys(r["sample"]["kind"] for r in rows))
    for k in kinds:
        g = [r for r in rows if r["sample"]["kind"] == k]
        ga = [r for r in g if r["status"] == "AUTO_ACCEPT"]
        L.append(f"| {k} | {len(g)} | {pct(np.mean([r['correct'] for r in g]))} | "
                 f"{pct(len(ga) / len(g))} | {pct(np.mean([r['correct'] for r in ga])) if ga else '–'} | "
                 f"{sum(not r['correct'] for r in ga)} |")
    L.append("")

    L.append("## Confidence calibration\n")
    L.append("![calibration](figures/calibration.png)\n")
    L.append("| confidence | n | accuracy |\n|---|---|---|")
    conf = np.array([r["confidence"] for r in rows])
    for lo, hi in ((0, .4), (.4, .6), (.6, .8), (.8, .9), (.9, 1.01)):
        m = (conf >= lo) & (conf < hi)
        L.append(f"| {lo:.1f}–{min(hi, 1):.1f} | {m.sum()} | "
                 f"{pct(np.mean([rows[i]['correct'] for i in np.where(m)[0]])) if m.sum() else '–'} |")
    L.append("")

    L.append("## Review flags\n")
    L.append("How often each review flag fired, and how often the best guess was actually wrong "
             "when it did.\n")
    L.append("| flag | fired | best guess wrong |\n|---|---|---|")
    fc = Counter()
    fw = Counter()
    for r in rows:
        for code in {f["code"] for f in r["flags"] if f["severity"] == "review"}:
            fc[code] += 1
            fw[code] += not r["correct"]
    for code, n in fc.most_common():
        L.append(f"| `{code}` | {n} | {fw[code]} ({pct(fw[code] / n)}) |")
    L.append("")

    L.append("## Minor-phase detection\n")
    L.append("![minor](figures/minor_detection.png)\n")
    L.append("| minor fraction | n | detected | missed ones sent to review |\n|---|---|---|---|")
    for d in det:
        L.append(f"| {d[0]} | {d[1]} | {pct(d[2])} | {pct(d[3])} |")
    L.append("")

    L.append("## Lattice parameters (correctly identified cubic phases, ≥3 peaks)\n")
    L.append("![lattice](figures/lattice_error.png)\n")
    L.append("| phase | n | median abs error, mean of peaks (Å) | median abs error, Nelson–Riley (Å) | NR better in |")
    L.append("|---|---|---|---|---|")
    for n in ["FCC", "BCC", "B2"]:
        if errs[n]["nr"]:
            nr, mn = np.abs(errs[n]["nr"]), np.abs(errs[n]["mean"])
            L.append(f"| {n} | {nr.size} | {np.median(mn):.5f} | {np.median(nr):.5f} | "
                     f"{pct(np.mean(nr < mn))} |")
    sig = [(p["a"] - p["a_true"], p["c"] - p["c_true"]) for r in rows for p in r["phases"]
           if p["name"] == "sigma" and "a_true" in p and p["c"] is not None and np.isfinite(p["c"])]
    if sig:
        sig = np.abs(np.array(sig))
        L.append(f"\nSigma (LSQ a, c + displacement), n = {len(sig)}: median |Δa| = "
                 f"{np.median(sig[:, 0]):.4f} Å, median |Δc| = {np.median(sig[:, 1]):.4f} Å.\n")

    L.append("## Crystallite size\n")
    L.append("![size](figures/size_parity.png)\n")
    pa = np.array([(p[1], p[2], p[3], p[4] > 0) for p in pairs], float)
    if pa.size:
        L.append("Median |log₁₀(D_est / D_true)| for correctly identified phases, D reported "
                 "(not lower-bound):\n")
        L.append("| subset | n | naive Scherrer | Lorentzian-part Scherrer |\n|---|---|---|---|")
        for lab, m in (("strain-free", pa[:, 3] == 0), ("with microstrain", pa[:, 3] == 1),
                       ("D_true < 50 nm", pa[:, 0] < 50), ("D_true ≥ 50 nm", pa[:, 0] >= 50)):
            if m.sum():
                e_n = np.median(np.abs(np.log10(pa[m, 2] / pa[m, 0])))
                e_l = np.median(np.abs(np.log10(pa[m, 1] / pa[m, 0])))
                L.append(f"| {lab} | {m.sum()} | {e_n:.3f} (×{10 ** e_n:.2f}) | "
                         f"{e_l:.3f} (×{10 ** e_l:.2f}) |")
        L.append("")

    L.append("## Every wrong answer\n")
    L.append("| id | truth | predicted | status | conf | counts | review flags |")
    L.append("|---|---|---|---|---|---|---|")
    wrong = sorted([r for r in rows if not r["correct"]], key=lambda r: r["status"])
    for r in wrong:
        ph, cts = describe(r)
        fl = ", ".join(sorted({f["code"] for f in r["flags"] if f["severity"] == "review"})) or "—"
        L.append(f"| {r['sample']['sample_id']} | {ph} | {' + '.join(r['pred_set']) or '∅'} | "
                 f"{'**AUTO**' if r['status'] == 'AUTO_ACCEPT' else 'review'} | "
                 f"{r['confidence']:.2f} | {cts} | {fl} |")
    with open(os.path.join(RES, "BENCHMARK.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
