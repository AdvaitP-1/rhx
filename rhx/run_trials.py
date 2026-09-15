#!/usr/bin/env python3
"""
run_trials.py -- repeat a gate across independent trials and aggregate.

B2 in AUDIT.md: a single Gate 1 run performs ONE reclaim event and scores
coverage across ~30 snapshot times. Those times are serially correlated within a
trial, so treating them as independent observations overstates the evidence.
The correct unit of replication is the RECLAIM EVENT, not the snapshot.

This driver runs N independent trials, each with its own workload seed and its
own pre-registration label, and reports the distribution of per-trial coverage
with a bootstrap confidence interval on the mean. Run order is randomized and
recorded so that drift in machine state over the session cannot be mistaken for
an effect.

Usage:
    sudo python3 run_trials.py --gate 1 --trials 20 --out runs/g1 \
        --reclaim-mode proactive --pages 200000 --measure-s 300 --window-T 60
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "harness"))
sys.path.insert(0, str(HERE / "analysis"))

import numpy as np


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x), replace=True).mean()
                      for _ in range(n_boot)])
    return {"mean": float(x.mean()),
            "lo": float(np.quantile(means, alpha / 2)),
            "hi": float(np.quantile(means, 1 - alpha / 2)),
            "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "n": int(len(x))}


def manifest_exit_code(manifest) -> int:
    """Fail if any trial or the gate's existing aggregate criterion failed."""
    gate = manifest.get("gate")
    trials = manifest.get("trials")
    records = manifest.get("results", [])
    if gate not in (0, 1, 2) or not isinstance(trials, int) or trials < 1:
        return 1
    if len(records) != trials:
        return 1
    if any(r.get("returncode") != 0 or
           not isinstance(r.get("results"), dict) or
           r["results"].get(f"gate{gate}_pass") is not True
           for r in records):
        return 1

    agg = manifest.get("aggregate", {})
    if agg.get("gate_pass") is not True:
        return 1
    if gate == 0:
        for key in ("frozen_abs_err", "S_max_abs_err", "cv_rel_err", "q50_rel_err"):
            try:
                limit = records[0]["results"]["checks"][key]["limit"]
                ci = agg[key]
                if ci["n"] != trials or not np.isfinite(ci["hi"]) or ci["hi"] > limit:
                    return 1
            except (KeyError, TypeError, ValueError):
                return 1
        return 0
    if gate == 1:
        try:
            ci = agg["ci"]
            return 0 if (ci["n"] == trials and np.isfinite(ci["lo"])
                         and ci["lo"] >= 0.90) else 1
        except (KeyError, TypeError, ValueError):
            return 1
    try:
        hom = agg["homogeneous"]
        het = agg["heterogeneous"]
        diff = agg["paired_difference"]
        criteria = records[0]["results"]["pass_criteria"]
        return 0 if (hom["n"] == trials and het["n"] == trials and
                     diff["n"] == trials and np.isfinite(hom["hi"]) and
                     np.isfinite(het["lo"]) and np.isfinite(diff["lo"]) and
                     hom["hi"] <= criteria["homogeneous_amplification_max"] and
                     het["lo"] >= criteria["heterogeneous_amplification_min"] and
                     diff["lo"] > 0) else 1
    except (KeyError, TypeError, ValueError):
        return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Repeat a gate across trials")
    ap.add_argument("--gate", type=int, choices=[0, 1, 2])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check-manifest", type=Path,
                    help="check the exit verdict of an existing trials_manifest.json")
    ap.add_argument("--base-seed", type=int, default=20260503)
    ap.add_argument("--stop-on-invalid", action="store_true",
                    help="abort the whole run if any trial is invalidated")
    args, passthrough = ap.parse_known_args()

    if args.check_manifest is not None:
        manifest = json.loads(args.check_manifest.read_text())
        rc = manifest_exit_code(manifest)
        print(f"manifest verdict: {'PASS' if rc == 0 else 'FAIL'}")
        return rc
    if args.gate is None or args.out is None:
        ap.error("--gate and --out are required unless --check-manifest is used")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    script = {0: "gates/gate0_estimator.py",
              1: "gates/gate1_prediction.py",
              2: "gates/gate2_negative_control.py"}[args.gate]

    # Randomize trial order over seeds, and record the permutation.
    rng = np.random.default_rng(args.base_seed)
    seeds = [args.base_seed + i for i in range(args.trials)]
    order = list(rng.permutation(len(seeds)))

    manifest = {"gate": args.gate, "trials": args.trials,
                "base_seed": args.base_seed, "seed_order": [int(i) for i in order],
                "passthrough": passthrough, "started_unix": time.time(),
                "results": []}

    for k, idx in enumerate(order):
        seed = seeds[idx]
        tdir = out / f"trial_{k:03d}_seed{seed}"
        cmd = [sys.executable, str(HERE / script), "--out", str(tdir),
               "--seed", str(seed)] + passthrough
        if args.gate == 1:
            cmd += ["--label", f"trial{k:03d}_seed{seed}"]
        print(f"\n{'='*70}\nTRIAL {k+1}/{args.trials}  seed={seed}\n{'='*70}")
        rc = subprocess.call(cmd)
        rec = {"trial": k, "seed": int(seed), "returncode": rc,
               "dir": str(tdir)}
        rf = tdir / f"gate{args.gate}_results.json"
        if rf.exists():
            try:
                rec["results"] = json.loads(rf.read_text())
            except json.JSONDecodeError:
                rec["results"] = None
        manifest["results"].append(rec)
        (out / "trials_manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str))
        if rc in (6,) and args.stop_on_invalid:
            print("trial invalidated and --stop-on-invalid set; aborting")
            break

    # ---------------- aggregate ----------------
    print(f"\n{'='*70}\nAGGREGATE OVER {len(manifest['results'])} TRIALS\n{'='*70}")
    agg = {}

    if args.gate == 1:
        cov = [r["results"]["score"]["coverage"]
               for r in manifest["results"]
               if r.get("results") and "score" in r["results"]]
        invalid = sum(1 for r in manifest["results"] if r["returncode"] == 6)
        ci = bootstrap_ci(cov)
        agg = {"metric": "coverage", "per_trial": cov, "ci": ci,
               "n_invalid": invalid,
               "n_pass_at_0.90": int(sum(1 for c in cov if c >= 0.90))}
        print(f"  valid trials      {ci['n']}   invalidated {invalid}")
        if ci["n"]:
            print(f"  coverage mean     {ci['mean']:.4f}")
            print(f"  95% CI            [{ci['lo']:.4f}, {ci['hi']:.4f}]")
            print(f"  per-trial sd      {ci['std']:.4f}")
            print(f"  trials >= 0.90    {agg['n_pass_at_0.90']}/{ci['n']}")
            # The gate is decided on the CI, not on any single trial.
            agg["gate_pass"] = bool(ci["lo"] >= 0.90)
            print(f"\n  GATE 1 (aggregate): "
                  f"{'PASS' if agg['gate_pass'] else 'FAIL'}  "
                  f"(criterion: lower CI bound >= 0.90)")
        else:
            agg["gate_pass"] = False

    elif args.gate == 2:
        arm_results = [r["results"]["arms"] for r in manifest["results"]
                       if isinstance(r.get("results"), dict) and
                       "arms" in r["results"] and
                       all(name in r["results"]["arms"] and
                           "amplification_measured" in r["results"]["arms"][name]
                           for name in ("homogeneous", "heterogeneous"))]
        hom = [arms["homogeneous"]["amplification_measured"] for arms in arm_results]
        het = [arms["heterogeneous"]["amplification_measured"] for arms in arm_results]
        ci_h = bootstrap_ci(hom, seed=1)
        ci_e = bootstrap_ci(het, seed=2)
        agg = {"homogeneous": ci_h, "heterogeneous": ci_e,
               "per_trial_hom": hom, "per_trial_het": het}
        print(f"  homogeneous   A = {ci_h['mean']:.4f}  "
              f"95% CI [{ci_h['lo']:.4f}, {ci_h['hi']:.4f}]  n={ci_h['n']}")
        print(f"  heterogeneous A = {ci_e['mean']:.4f}  "
              f"95% CI [{ci_e['lo']:.4f}, {ci_e['hi']:.4f}]  n={ci_e['n']}")
        # Paired comparison: same session, same machine, arms interleaved.
        pairs = [(h, e) for h, e in zip(hom, het)
                 if np.isfinite(h) and np.isfinite(e)]
        if len(pairs) >= 3:
            d = np.array([e - h for h, e in pairs])
            ci_d = bootstrap_ci(d, seed=3)
            agg["paired_difference"] = ci_d
            print(f"  paired het-hom  = {ci_d['mean']:+.4f}  "
                  f"95% CI [{ci_d['lo']:+.4f}, {ci_d['hi']:+.4f}]")
            agg["gate_pass"] = bool(ci_h["hi"] <= 1.10 and ci_d["lo"] > 0)
            print(f"\n  GATE 2 (aggregate): "
                  f"{'PASS' if agg['gate_pass'] else 'FAIL'}")
            print("  criterion: homogeneous upper CI <= 1.10 AND paired "
                  "difference CI excludes 0")
        else:
            agg["gate_pass"] = False
            print("\n  GATE 2 (aggregate): FAIL (fewer than 3 valid pairs)")

    elif args.gate == 0:
        keys = ["frozen_abs_err", "S_max_abs_err", "cv_rel_err", "q50_rel_err"]
        for kk in keys:
            vals = [r["results"]["checks"][kk]["value"]
                    for r in manifest["results"]
                    if isinstance(r.get("results"), dict) and
                    kk in r["results"].get("checks", {})]
            ci = bootstrap_ci(vals, seed=hash(kk) % 1000)
            agg[kk] = ci
            print(f"  {kk:18s} {ci['mean']:.5f}  "
                  f"95% CI [{ci['lo']:.5f}, {ci['hi']:.5f}]  n={ci['n']}")
        first_checks = next((r["results"]["checks"] for r in manifest["results"]
                             if isinstance(r.get("results"), dict) and
                             "checks" in r["results"] and
                             all(k in r["results"]["checks"] for k in keys)), {})
        limits = {k: first_checks[k]["limit"] for k in keys} if first_checks else {}
        agg["gate_pass"] = bool(limits and all(
            agg[k]["n"] == args.trials and np.isfinite(agg[k]["hi"]) and
            agg[k]["hi"] <= limits[k] for k in keys))
        print(f"\n  GATE 0 (aggregate): {'PASS' if agg['gate_pass'] else 'FAIL'}")

    manifest["aggregate"] = agg
    manifest["finished_unix"] = time.time()
    (out / "trials_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    print(f"\nwritten to {out / 'trials_manifest.json'}")
    rc = manifest_exit_code(manifest)
    print(f"TRIAL DRIVER: {'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
