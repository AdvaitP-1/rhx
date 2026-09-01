#!/usr/bin/env python3
"""
Gate 2 -- the homogeneous-access negative control.

Theorem 6 predicts a NULL: a workload whose pages all have the same access rate
has no cold/hot composition to distort, so repeated reclaim should not become
progressively more expensive. Amplification A(Delta) must approach 1.

A correctly predicted null is stronger evidence for the composition-shift
mechanism than any positive result, because a phenomenological debt model makes
no such prediction. It is also the check that distinguishes reclaim hysteresis
from generic "memory pressure hurts".

The gate runs matched arms that differ ONLY in the access-rate distribution:

    homogeneous   all pages at the same rate      -> predict A ~ 1
    heterogeneous Gamma-distributed rates         -> predict A > 1

Everything else is held fixed: page count, backing, reclaim fraction, victim
kernel, snapshot schedule, telemetry rate, and the reclaim mode.

Because a null is only meaningful from a design that COULD have detected an
effect, the gate reports the observed heterogeneous amplification alongside the
homogeneous one, and refuses to certify the null if the positive arm failed to
show an effect.
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
from cgroupv2 import Cgroup, cgroup_v2_available
from kernels import amplification_from_rates
from randomization import (SequentialRandomizer, make_mean_difference_statistic,
                           randomization_test)
from telemetry import Telemetry, to_csv
from workload import RateSpec, Workload, parse_counts, parse_residency


# Declared in advance.
PASS_CRITERIA = {
    "homogeneous_amplification_max": 1.10,   # null arm must stay near 1
    "heterogeneous_amplification_min": 1.30, # positive arm must show an effect
}


def run_arm(name: str, out: Path, args, spec: RateSpec) -> dict:
    """Run one arm: observe, reclaim repeatedly, measure marginal victim rate."""
    arm_dir = out / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    page_size = 4096

    wl = Workload(binary=args.binary, workdir=arm_dir / "workload",
                  n_pages=args.pages, seed=args.seed, rate_spec=spec,
                  backing=args.backing, prefault=True)
    total_s = args.measure_s + args.n_reclaims * (args.interval_s + 5.0) + 60.0
    pid = wl.start(duration_s=total_s, report_s=60.0)
    print(f"[{name}] workload pid={pid}")

    cg = None
    tel = None
    result = {"arm": name, "rate_spec": spec.as_dict()}
    try:
        if cgroup_v2_available():
            cg = Cgroup(f"{args.cgroup_name}_{name}")
            cg.create()
            cg.add_pid(pid)

        tel = Telemetry(cg, arm_dir / "telemetry", interval_s=1.0 / args.telemetry_hz)
        tel.start()

        time.sleep(1.0)
        wl.reset_counts()
        print(f"[{name}] observing {args.measure_s:.0f}s ...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.measure_s:
            if not wl.alive():
                raise RuntimeError("workload died during observation")
            time.sleep(1.0)
        counts_path = wl.dump_counts(tag="pre")
        cd = parse_counts(counts_path)
        lam_true = cd["lambda_true"]
        lam_hat = cd["counts"] / cd["elapsed_s"]
        cv_true = float(lam_true.std(ddof=1) / lam_true.mean())
        print(f"[{name}] CV(true)={cv_true:.4f}")
        result["cv_true"] = cv_true
        result["observation_window_s"] = float(cd["elapsed_s"])

        # Closed-form prediction for amplification (Theorem 5), computed from
        # the TRUE rates so this arm's prediction is independent of estimator
        # error. Reported alongside the measured value.
        amp_pred = amplification_from_rates(lam_true, args.victim_frac,
                                            args.victim_frac)
        result["amplification_predicted"] = amp_pred

        # Repeated reclaim: measure the mean access rate of the victim set each
        # time. Theorem 2 says this is the marginal distortion proxy; if
        # composition shifts, successive victim sets get hotter.
        reclaim_bytes = int(args.pages * args.victim_frac) * page_size
        marginal_rates = []
        victim_counts = []
        for i in range(args.n_reclaims):
            wl.pause()                       # B4
            snap_a = wl.snapshot_residency(f"{name}_pre{i}")
            ra = parse_residency(snap_a)
            tel.event("reclaim_begin", iteration=i, mode=args.reclaim_mode)
            if cg is not None:
                if args.reclaim_mode == "proactive":
                    rec = cg.reclaim_proactive(reclaim_bytes)
                else:
                    cur = cg.memory_current() or 0
                    rec = cg.reclaim_natural(max(cur - reclaim_bytes, page_size),
                                             settle_s=2.0, watch_pid=pid)
                    if getattr(rec, "oom_kill_delta", 0) > 0:
                        raise RuntimeError(
                            f"OOM kill during reclaim {i}; trial invalid")
                tel.event("reclaim_end", iteration=i,
                          actual_delta_bytes=rec.actual_delta_bytes,
                          wrote_ok=rec.wrote_ok, error=rec.error)
            snap_b = wl.snapshot_residency(f"{name}_post{i}")
            rb = parse_residency(snap_b)
            wl.resume()                      # B4
            victims = ra["resident"] & (~rb["resident"])
            nv = int(victims.sum())
            victim_counts.append(nv)
            if nv > 0:
                mr = float(lam_true[victims].mean())
            else:
                mr = float("nan")
            marginal_rates.append(mr)
            print(f"[{name}] reclaim {i}: victims={nv} mean_victim_rate={mr:.6f}")
            # allow partial recovery before the next reclaim
            time.sleep(args.interval_s)

        result["victim_counts"] = victim_counts
        result["marginal_victim_rates"] = marginal_rates

        valid = [r for r in marginal_rates if np.isfinite(r)]
        if len(valid) >= 2 and valid[0] > 0:
            result["amplification_measured"] = float(valid[-1] / valid[0])
            result["amplification_first_to_second"] = float(valid[1] / valid[0])
        else:
            result["amplification_measured"] = float("nan")
            result["amplification_first_to_second"] = float("nan")
        return result

    finally:
        if tel is not None:
            tel.stop()
            to_csv(tel.samples_path, arm_dir / "telemetry" / "samples.csv")
        wl.stop()
        if cg is not None:
            cg.destroy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate 2: homogeneous negative control")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--binary", type=Path, default=HERE.parent / "workload" / "rategen")
    ap.add_argument("--pages", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--measure-s", type=float, default=90.0)
    ap.add_argument("--n-reclaims", type=int, default=5)
    ap.add_argument("--interval-s", type=float, default=10.0)
    ap.add_argument("--victim-frac", type=float, default=0.15)
    ap.add_argument("--reclaim-mode", required=True, choices=["proactive", "natural"])
    ap.add_argument("--backing", default="FILE", choices=["FILE", "ANON"])
    ap.add_argument("--telemetry-hz", type=float, default=2.0)
    ap.add_argument("--hom-rate", type=float, default=0.4,
                    help="the single rate used by the homogeneous arm")
    ap.add_argument("--het-shape", type=float, default=0.8)
    ap.add_argument("--het-scale", type=float, default=0.5)
    ap.add_argument("--cgroup-name", default="rhx_gate2")
    ap.add_argument("--allow-unpublishable", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    envinfo = envcapture.write_capture(out)
    verdict = envinfo["verdict"]
    print(f"environment sha256={envinfo['sha256'][:16]}...  "
          f"publishable={verdict['publishable']}")
    for b in verdict["blockers"]:
        print(f"  BLOCKER: {b}")
    if not verdict["publishable"] and not args.allow_unpublishable:
        print("Refusing to run. Pass --allow-unpublishable for development.")
        return 2

    # Arm order is randomized and recorded, so a systematic drift in machine
    # state over the session cannot be confused with an arm effect.
    rnd = SequentialRandomizer(seed=args.seed, propensity=0.5)
    rec = rnd.draw(0, time.monotonic())
    order = ["homogeneous", "heterogeneous"] if rec.assigned else \
            ["heterogeneous", "homogeneous"]
    print(f"arm order (randomized, recorded): {order}")

    specs = {
        "homogeneous": RateSpec("uniform", p1=args.hom_rate),
        "heterogeneous": RateSpec("gamma", p1=args.het_shape, p2=args.het_scale),
    }

    arms = {}
    for name in order:
        arms[name] = run_arm(name, out, args, specs[name])

    hom = arms["homogeneous"]
    het = arms["heterogeneous"]

    a_hom = hom.get("amplification_measured", float("nan"))
    a_het = het.get("amplification_measured", float("nan"))

    print("\n--- RESULTS ---")
    print(f"  homogeneous   CV={hom.get('cv_true', float('nan')):.5f}  "
          f"amplification={a_hom:.4f}")
    print(f"  heterogeneous CV={het.get('cv_true', float('nan')):.5f}  "
          f"amplification={a_het:.4f}")

    print("\n--- PASS CRITERIA (declared in advance) ---")
    null_ok = bool(np.isfinite(a_hom) and
                   a_hom <= PASS_CRITERIA["homogeneous_amplification_max"])
    pos_ok = bool(np.isfinite(a_het) and
                  a_het >= PASS_CRITERIA["heterogeneous_amplification_min"])
    print(f"  homogeneous   A={a_hom:.4f} <= "
          f"{PASS_CRITERIA['homogeneous_amplification_max']}   "
          f"{'PASS' if null_ok else 'FAIL'}")
    print(f"  heterogeneous A={a_het:.4f} >= "
          f"{PASS_CRITERIA['heterogeneous_amplification_min']}   "
          f"{'PASS' if pos_ok else 'FAIL'}")

    if not pos_ok:
        print("\n  The positive arm did not show an effect. The null from the "
              "homogeneous arm is therefore UNINFORMATIVE: this design has not "
              "demonstrated it could detect hysteresis at all.")

    results = {
        "gate": 2,
        "args": vars(args),
        "environment_sha256": envinfo["sha256"],
        "publishable_environment": verdict["publishable"],
        "arm_order": order,
        "arm_order_assignment": rnd.as_dicts(),
        "reclaim_mode": args.reclaim_mode,
        "arms": arms,
        "pass_criteria": PASS_CRITERIA,
        "null_arm_pass": null_ok,
        "positive_arm_pass": pos_ok,
        "gate2_pass": bool(null_ok and pos_ok),
    }
    (out / "gate2_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nGATE 2: {'PASS' if results['gate2_pass'] else 'FAIL'}")
    print(f"results written to {out / 'gate2_results.json'}")
    return 0 if results["gate2_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
