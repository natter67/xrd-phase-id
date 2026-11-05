"""
Benchmark the pipeline on 200 synthetic HEA patterns.

    python scripts/run_benchmark.py            # writes results/benchmark.json
    python scripts/run_benchmark.py --seed 7   # held-out set -> results/benchmark_seed7.json
    python scripts/make_report.py              # figures + results/BENCHMARK.md

Each synthetic sample has a known ground truth (phases, lattice parameters,
crystallite sizes, ...).  We run the full pipeline on the simulated counts
and compare.  The key metrics are:

* how often the phase set is right when the pipeline auto-accepts
  (a wrong auto-accepted answer is a *silent failure* — the worst outcome),
* how many samples it sends to review, and whether those really were hard,
* lattice-parameter and crystallite-size errors on correctly-identified phases.
"""

from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker process: otherwise 4 processes each
# spawn 4 linear-algebra threads and fight over the CPU (10–30× slowdown).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time
from multiprocessing import Pool

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xrd_phase_id.pipeline import analyze  # noqa: E402
from xrd_phase_id.simulate import benchmark_samples, simulate  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def run_one(sample):
    x, y, _ = simulate(sample)
    t0 = time.time()
    try:
        res = analyze(x, y)
    except Exception as exc:  # a crash is a failure mode worth recording too
        return dict(sample=sample.to_dict(), error=repr(exc))
    dt = time.time() - t0
    truth = {p.name: p for p in sample.phases}
    true_set = tuple(sorted(truth))
    phases = []
    for pr in res.phases:
        d = dict(name=pr.name, a=pr.a, a_err=pr.a_err, c=pr.c, a_mean=pr.a_mean_uncorrected,
                 method=pr.lattice_method, n_peaks=pr.n_matched_peaks, confidence=pr.confidence,
                 order=pr.order_parameter,
                 size=pr.size.size_nm if pr.size else None,
                 size_naive=pr.size.size_naive_nm if pr.size else None,
                 size_lb=pr.size.lower_bound_only if pr.size else None,
                 strain=pr.size.strain if pr.size else None)
        if pr.name in truth:
            t = truth[pr.name]
            d.update(a_true=t.a, c_true=t.c, size_true=t.size_nm, strain_true=t.strain,
                     order_true=t.order)
        phases.append(d)
    return dict(
        sample=sample.to_dict(),
        true_set=list(true_set),
        pred_set=list(res.phase_set),
        correct=res.phase_set == true_set,
        status=res.status,
        confidence=res.confidence,
        conf_terms={k: v for k, v in res.confidence_terms.items()},
        flags=[dict(code=f.code, severity=f.severity, message=f.message) for f in res.flags],
        alternatives=res.alternatives[:3],
        phases=phases,
        n_peaks=len(res.peaks),
        explained=res.explained_fraction,
        runtime_s=dt,
    )


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026,
                    help="2026 = development set (thresholds were tuned on it); "
                         "any other seed = held-out set")
    args = ap.parse_args()
    name = "benchmark.json" if args.seed == 2026 else f"benchmark_seed{args.seed}.json"
    OUT = os.path.join(RESULTS, name)
    samples = benchmark_samples(seed=args.seed)
    t0 = time.time()
    with Pool(max(1, os.cpu_count() or 1)) as pool:
        results = pool.map(run_one, samples, chunksize=2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1, default=float)
    ok = [r for r in results if "error" not in r]
    auto = [r for r in ok if r["status"] == "AUTO_ACCEPT"]
    print(f"{len(results)} samples in {time.time() - t0:.0f} s, {len(results) - len(ok)} crashes")
    print(f"overall phase-set accuracy: {np.mean([r['correct'] for r in ok]):.1%}")
    print(f"auto-accepted: {len(auto)} ({len(auto) / len(ok):.0%}), "
          f"accuracy among auto: {np.mean([r['correct'] for r in auto]) if auto else float('nan'):.1%}")


if __name__ == "__main__":
    main()
