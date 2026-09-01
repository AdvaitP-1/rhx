#!/usr/bin/env python3
"""
Gate 0 -- can we measure the theoretical primitive at all?

The theory begins with f(lambda), the page access-rate density. Every downstream
result is a functional of it. Before interpreting Redis or any real application,
the measurement pipeline must be shown to recover a KNOWN density.

This gate runs the synthetic workload, whose per-page rates are drawn from a
distribution chosen by the experimenter and written to disk BEFORE any access
occurs, then estimates the density with each available estimator and reports the
error in the quantities the experiment actually depends on:

  - the frozen fraction below lambda_min = 1/T
  - the survival functional S(t) over the declared window
  - the coefficient of variation, because Theorem 6's negative control is a
    statement about dispersion

Pass criteria are declared in advance, not chosen after seeing the numbers.

If this gate fails, no later gate is interpretable, and the correct response is
to fix the estimator or shorten the window, not to proceed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE.parent / "analysis"))

import numpy as np

import envcapture
from estimator import (compare_densities, estimate_from_counts, estimate_oracle,
                       predict_recovery, victim_weights)
from workload import RateSpec, Workload, parse_counts


# Declared in advance. Changing these after seeing results invalidates the gate.
PASS_CRITERIA = {
    "frozen_abs_err_max": 0.05,      # frozen fraction within 5 percentage points
    "S_max_abs_err_max": 0.05,       # survival functional within 0.05 absolute
    "cv_rel_err_max": 0.15,          # CV within 15 percent
    "q50_rel_err_max": 0.20,         # median rate within 20 percent
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate 0: estimator validation")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--binary", type=Path, default=HERE.parent / "workload" / "rategen")
    ap.add_argument("--pages", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--window-T", type=float, default=60.0,
                    help="declared operating window T; lambda_min = 1/T")
    ap.add_argument("--measure-s", type=float, default=120.0,
                    help="how long to observe before estimating rates")
    ap.add_argument("--dist", default="gamma",
                    choices=["gamma", "uniform", "lognormal"])
    ap.add_argument("--p1", type=float, default=0.8)
    ap.add_argument("--p2", type=float, default=0.5)
    ap.add_argument("--backing", default="FILE", choices=["FILE", "ANON"])
    ap.add_argument("--victim-frac", type=float, default=0.30)
    ap.add_argument("--allow-unpublishable", action="store_true",
                    help="proceed on a VM / non-bare-metal host (development only)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- environment ----------------
    envinfo = envcapture.write_capture(out)
    verdict = envinfo["verdict"]
    print(f"environment captured  sha256={envinfo['sha256'][:16]}...")
    print(f"publishable: {verdict['publishable']}")
    for b in verdict["blockers"]:
        print(f"  BLOCKER: {b}")
    for w in verdict["warnings"]:
        print(f"  warning: {w}")
    if not verdict["publishable"] and not args.allow_unpublishable:
        print("\nRefusing to run: this host cannot produce publishable data.")
        print("Pass --allow-unpublishable to run anyway for development.")
        return 2
    if not verdict["publishable"]:
        print("\n*** DEVELOPMENT RUN -- results are NOT publishable evidence ***\n")

    # ---------------- workload ----------------
    spec = RateSpec(kind=args.dist, p1=args.p1, p2=args.p2)
    wl = Workload(binary=args.binary, workdir=out / "workload",
                  n_pages=args.pages, seed=args.seed, rate_spec=spec,
                  backing=args.backing, prefault=True)

    print(f"\nstarting workload: {args.pages} pages, {args.dist}"
          f"({args.p1},{args.p2}), backing={args.backing}")
    pid = wl.start(duration_s=args.measure_s + 60.0, report_s=30.0)
    print(f"  pid={pid}")

    try:
        # Reset counters so the observation window is exactly known.
        time.sleep(1.0)
        wl.reset_counts()
        t_start = time.monotonic()
        print(f"  observing for {args.measure_s:.0f}s ...")
        while time.monotonic() - t_start < args.measure_s:
            if not wl.alive():
                print("  workload died during observation")
                return 3
            time.sleep(1.0)

        counts_path = wl.dump_counts(tag="gate0")
        truth = wl.read_truth()
        print(f"  counts written: {counts_path}")
    finally:
        wl.stop()

    cd = parse_counts(counts_path)
    window_obs = float(cd["elapsed_s"])
    counts = cd["counts"]
    lam_true = cd["lambda_true"]

    if not np.allclose(lam_true, truth, rtol=1e-6):
        print("FATAL: rates in counts file disagree with the pre-written truth file")
        return 4

    print(f"\nobservation window (workload-reported): {window_obs:.3f}s")
    print(f"total events: {int(counts.sum())}")

    # ---------------- estimators ----------------
    est_oracle = estimate_oracle(lam_true)
    est_counts = estimate_from_counts(counts, window_obs)

    results = {
        "gate": 0,
        "args": vars(args),
        "environment_sha256": envinfo["sha256"],
        "publishable_environment": verdict["publishable"],
        "observation_window_s": window_obs,
        "total_events": int(counts.sum()),
        "truth_summary": est_oracle.summary(),
        "counts_summary": est_counts.summary(),
    }

    cmp = compare_densities(lam_true, est_counts.lam, args.window_T)
    results["comparison_counts_vs_truth"] = cmp

    print("\n--- estimator vs ground truth (window T = "
          f"{args.window_T:.0f}s, lambda_min = {1/args.window_T:.5f}) ---")
    print(f"  frozen fraction   true={cmp['frozen_true']:.4f}  "
          f"est={cmp['frozen_est']:.4f}  abs err={cmp['frozen_abs_err']:.4f}")
    print(f"  CV                true={cmp['cv_true']:.4f}  "
          f"est={cmp['cv_est']:.4f}  rel err={cmp['cv_rel_err']:.4f}")
    for q in (5, 25, 50, 75, 95):
        print(f"  q{q:02d}               true={cmp[f'q{q:02d}_true']:.5f}  "
              f"est={cmp[f'q{q:02d}_est']:.5f}  rel err={cmp[f'q{q:02d}_rel_err']:.4f}")
    print(f"  S(t) max abs err  {cmp['S_max_abs_err']:.5f}")
    print(f"  S(t) rmse         {cmp['S_rmse']:.5f}")
    print(f"  censored (zero-count) fraction: "
          f"{est_counts.censored_mask.mean():.4f}")

    # ---------------- recovery prediction agreement ----------------
    w_true = victim_weights(lam_true, "coldest_first", frac=args.victim_frac)
    w_est = victim_weights(est_counts.lam, "coldest_first", frac=args.victim_frac)
    p_true = predict_recovery(est_oracle, w_true, args.window_T, n_grid=30,
                              n_boot=300, seed=1)
    p_est = predict_recovery(est_counts, w_est, args.window_T, n_grid=30,
                             n_boot=300, seed=1)
    band_overlap = float(np.mean(
        (p_est.S_lo <= p_true.S_hi + 1e-12) & (p_true.S_lo <= p_est.S_hi + 1e-12)
    ))
    results["recovery_prediction"] = {
        "victim_frac": args.victim_frac,
        "frozen_fraction_true": p_true.frozen_fraction,
        "frozen_fraction_est": p_est.frozen_fraction,
        "S_pred_max_abs_diff": float(np.max(np.abs(p_true.S_pred - p_est.S_pred))),
        "bracket_overlap_fraction": band_overlap,
        "t_grid": [float(x) for x in p_true.t_grid],
        "S_true": [float(x) for x in p_true.S_pred],
        "S_est": [float(x) for x in p_est.S_pred],
    }
    print(f"\n--- recovery prediction (victim frac {args.victim_frac}) ---")
    print(f"  frozen fraction  true={p_true.frozen_fraction:.4f}  "
          f"est={p_est.frozen_fraction:.4f}")
    print(f"  max |S_true - S_est| = "
          f"{results['recovery_prediction']['S_pred_max_abs_diff']:.5f}")
    print(f"  bracket overlap = {band_overlap:.3f}")

    # ---------------- verdict ----------------
    checks = {
        "frozen_abs_err": (cmp["frozen_abs_err"], PASS_CRITERIA["frozen_abs_err_max"]),
        "S_max_abs_err": (cmp["S_max_abs_err"], PASS_CRITERIA["S_max_abs_err_max"]),
        "cv_rel_err": (cmp["cv_rel_err"], PASS_CRITERIA["cv_rel_err_max"]),
        "q50_rel_err": (cmp["q50_rel_err"], PASS_CRITERIA["q50_rel_err_max"]),
    }
    passed = {}
    print("\n--- PASS CRITERIA (declared in advance) ---")
    for name, (val, lim) in checks.items():
        ok = bool(np.isfinite(val) and val <= lim)
        passed[name] = ok
        print(f"  {name:20s} {val:.5f}  <=  {lim:.5f}   {'PASS' if ok else 'FAIL'}")
    all_pass = all(passed.values())
    results["pass_criteria"] = PASS_CRITERIA
    results["checks"] = {k: {"value": float(v), "limit": float(l),
                             "pass": passed[k]} for k, (v, l) in checks.items()}
    results["gate0_pass"] = all_pass

    (out / "gate0_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nGATE 0: {'PASS' if all_pass else 'FAIL'}")
    print(f"results written to {out / 'gate0_results.json'}")
    if not all_pass:
        print("\nDo not proceed to Gate 1. Either the estimator or the window "
              "needs work: a shorter T raises lambda_min and shrinks the "
              "unresolved cold tail.")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
