#!/usr/bin/env python3
"""
Gate 1 -- the pre-registered prediction test.

This is the experiment that can falsify the theory most directly.

Sequence, in this order, enforced by the code:

  1. Observe the workload and estimate f(lambda). No reclaim has occurred.
  2. Compute the predicted recovery bracket S(t) over the declared window,
     conditioned on the victim kernel that the reclaim step will use.
  3. COMMIT the prediction to a hash-chained ledger. Re-registration under the
     same label is refused, so the prediction cannot be revised after seeing the
     outcome.
  4. Reclaim, in one of two modes:
       proactive -- write memory.reclaim. Exercises the same victim-selection
                    machinery but is NOT accounted as pressure, so it is valid
                    for page-dynamics claims only.
       natural   -- lower memory.high and let genuine contention drive reclaim.
                    Required for any claim involving pressure or damage.
     The mode is recorded in every output file.
  5. Measure recovery by taking mincore residency snapshots on a schedule, which
     gives exact per-page residency rather than inferring it from counters.
  6. Score the observation against the committed bracket, and diagnose the
     failure mode when it misses.

Two reclaim modes are never mixed within a run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
sys.path.insert(0, str(HERE.parent / "analysis"))

import numpy as np

import envcapture
import prereg
from cgroupv2 import Cgroup, cgroup_v2_available
from estimator import (estimate_from_counts, predict_recovery, to_registerable,
                       victim_weights)
from kernels import compare_families, score_prediction
from telemetry import Telemetry, load_samples, to_csv
from workload import RateSpec, Workload, parse_counts, parse_residency


def snapshot_schedule(T: float, n: int = 30) -> list:
    """Log-spaced snapshot times on (0, T]. Recovery is fastest early, so linear
    spacing wastes samples late and under-resolves the knee."""
    if n < 2:
        return [T]
    lo = max(T / 200.0, 0.05)
    return list(np.unique(np.geomspace(lo, T, n)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate 1: pre-registered prediction test")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", required=True,
                    help="prediction label; must be unique in the ledger")
    ap.add_argument("--binary", type=Path, default=HERE.parent / "workload" / "rategen")
    ap.add_argument("--pages", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--window-T", type=float, default=60.0)
    ap.add_argument("--measure-s", type=float, default=120.0)
    ap.add_argument("--dist", default="gamma", choices=["gamma", "uniform", "lognormal"])
    ap.add_argument("--p1", type=float, default=0.8)
    ap.add_argument("--p2", type=float, default=0.5)
    ap.add_argument("--backing", default="FILE", choices=["FILE", "ANON"])
    ap.add_argument("--reclaim-mode", required=True, choices=["proactive", "natural"])
    ap.add_argument("--reclaim-frac", type=float, default=0.30,
                    help="fraction of the mapping to reclaim")
    ap.add_argument("--victim-kernel", default="coldest_first",
                    choices=["coldest_first", "softmin", "uniform"])
    ap.add_argument("--telemetry-hz", type=float, default=4.0)
    ap.add_argument("--n-snapshots", type=int, default=30)
    ap.add_argument("--cgroup-name", default="rhx_gate1")
    ap.add_argument("--allow-unpublishable", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- environment ----------------
    envinfo = envcapture.write_capture(out)
    verdict = envinfo["verdict"]
    print(f"environment sha256={envinfo['sha256'][:16]}...  "
          f"publishable={verdict['publishable']}")
    for b in verdict["blockers"]:
        print(f"  BLOCKER: {b}")
    if not verdict["publishable"] and not args.allow_unpublishable:
        print("Refusing to run. Pass --allow-unpublishable for development.")
        return 2
    if not cgroup_v2_available():
        print("cgroup v2 with a memory controller is required for Gate 1.")
        if not args.allow_unpublishable:
            return 2

    page_size = 4096
    n_reclaim_pages = int(args.pages * args.reclaim_frac)
    reclaim_bytes = n_reclaim_pages * page_size

    # ---------------- workload ----------------
    spec = RateSpec(kind=args.dist, p1=args.p1, p2=args.p2)
    wl = Workload(binary=args.binary, workdir=out / "workload",
                  n_pages=args.pages, seed=args.seed, rate_spec=spec,
                  backing=args.backing, prefault=True)
    total_s = args.measure_s + args.window_T + 120.0

    cg = None
    tel = None
    try:
        # Create the cgroup before launch.  Workload.start uses a wrapper that
        # joins this cgroup before exec, so rategen materializes and prefaults
        # the measured mapping with correct charge ownership from page one.
        if cgroup_v2_available():
            cg = Cgroup(args.cgroup_name)
            cg.create()

        pid = wl.start(
            duration_s=total_s, report_s=60.0,
            cgroup_procs=(cg.path / "cgroup.procs") if cg is not None else None,
        )
        print(f"workload pid={pid} pages={args.pages}")
        if cg is not None:
            if pid not in cg.procs():
                raise RuntimeError(
                    f"workload pid {pid} was not launched in cgroup {cg.path}")
            print(f"cgroup {cg.path} created, workload launched inside it")

        tel = Telemetry(cg, out / "telemetry", interval_s=1.0 / args.telemetry_hz)
        tel.start()
        tel.event("workload_started", pid=pid, pages=args.pages)

        # ---------------- STEP 1: observe, estimate f ----------------
        time.sleep(1.0)
        wl.reset_counts()
        tel.event("observation_window_start", planned_s=args.measure_s)
        print(f"observing {args.measure_s:.0f}s to estimate f(lambda) ...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.measure_s:
            if not wl.alive():
                print("workload died during observation")
                return 3
            time.sleep(1.0)
        counts_path = wl.dump_counts(tag="pre")
        tel.event("observation_window_end", counts_path=str(counts_path))

        cd = parse_counts(counts_path)
        est = estimate_from_counts(cd["counts"], cd["elapsed_s"])
        print(f"  estimated rates over {cd['elapsed_s']:.2f}s; "
              f"CV={est.summary()['cv']:.4f}  "
              f"censored={est.censored_mask.mean():.4f}")

        # ---------------- STEP 2: predict ----------------
        w = victim_weights(est.lam, args.victim_kernel, frac=args.reclaim_frac)
        pred = predict_recovery(est, w, args.window_T, n_grid=40,
                                n_boot=500, seed=args.seed)
        reg = to_registerable(pred, estimator_desc={
            "source": est.source,
            "window_s": est.window_s,
            "victim_kernel": args.victim_kernel,
            "reclaim_frac": args.reclaim_frac,
            "frac_censored": float(est.censored_mask.mean()),
            "cv_estimated": est.summary()["cv"],
        })

        # ---------------- STEP 3: COMMIT ----------------
        commit = prereg.register_prediction(
            root=out, label=args.label, prediction=reg,
            env_sha256=envinfo["sha256"],
            notes=(f"Gate 1 mode={args.reclaim_mode} dist={args.dist} "
                   f"({args.p1},{args.p2}) pages={args.pages}"),
        )
        print(f"\nPREDICTION COMMITTED")
        print(f"  sha256      = {commit['digest']}")
        print(f"  ledger head = {commit['ledger_head'][:32]}...")
        print(f"  frozen frac = {pred.frozen_fraction:.4f}  "
              f"lambda_min = {pred.lambda_min:.5f}")
        tel.event("prediction_committed", label=args.label,
                  sha256=commit["digest"], ledger_head=commit["ledger_head"])

        # ---------------- STEP 4: reclaim ----------------
        # B4: quiesce for the whole reclaim window so residency changes are
        # attributable to reclaim rather than to ordinary workload access.
        wl.pause()
        snap_pre = wl.snapshot_residency("pre_reclaim")
        r_pre = parse_residency(snap_pre)
        print(f"\npre-reclaim residency: {r_pre['resident'].mean():.4f}")

        tel.event("reclaim_begin", mode=args.reclaim_mode,
                  requested_bytes=reclaim_bytes)
        rec = None
        if cg is not None:
            if args.reclaim_mode == "proactive":
                rec = cg.reclaim_proactive(reclaim_bytes)
            else:
                cur = cg.memory_current() or 0
                target = max(cur - reclaim_bytes, page_size)
                rec = cg.reclaim_natural(target, settle_s=3.0,
                                         watch_pid=pid)
            print(f"reclaim mode={rec.mode} requested={rec.requested_bytes} "
                  f"actual_delta={rec.actual_delta_bytes} ok={rec.wrote_ok} "
                  f"outcome={rec.outcome} attempts={rec.attempts} "
                  f"err={rec.error}")
            reclaim_info = asdict(rec)
            reclaim_info["duration_s"] = rec.duration_s
            tel.event("reclaim_end", **reclaim_info)
        else:
            print("no cgroup available; skipping reclaim (development run)")
            tel.event("reclaim_skipped", reason="no cgroup")

        t_reclaim = time.monotonic()
        snap_post = wl.snapshot_residency("post_reclaim")
        r_post = parse_residency(snap_post)
        wl.resume()          # recovery must proceed under normal access

        # A partial proactive reclaim does not match the pre-registered victim
        # fraction.  Preserve its measured residency and reclaim accounting,
        # but invalidate the trial instead of scoring it as a smaller success.
        if (rec is not None and args.reclaim_mode == "proactive" and
                not rec.wrote_ok):
            partial_victims = r_pre["resident"] & (~r_post["resident"])
            partial_n = int(partial_victims.sum())
            print(f"\nTRIAL INVALID: proactive reclaim outcome={rec.outcome}; "
                  f"{rec.progress_bytes}/{rec.requested_bytes} bytes completed, "
                  f"measured victims={partial_n}.")
            reclaim_info = asdict(rec)
            reclaim_info["duration_s"] = rec.duration_s
            (out / "gate1_results.json").write_text(json.dumps({
                "gate": 1,
                "invalid": "reclaim_incomplete",
                "reclaim": reclaim_info,
                "pre_resident_frac": float(r_pre["resident"].mean()),
                "post_resident_frac": float(r_post["resident"].mean()),
                "n_victims": partial_n,
            }, indent=2, default=str))
            return 6

        # B3: an OOM kill invalidates the trial. Do not analyze it.
        if rec is not None and getattr(rec, "oom_kill_delta", 0) > 0:
            print(f"\nTRIAL INVALID: {rec.oom_kill_delta} OOM kill(s) during "
                  f"reclaim. Discard this trial and lower the pressure target.")
            (out / "gate1_results.json").write_text(json.dumps(
                {"gate": 1, "invalid": "oom_kill",
                 "oom_kill_delta": rec.oom_kill_delta}, indent=2))
            return 6
        if rec is not None and rec.workload_survived is False:
            print("\nTRIAL INVALID: workload did not survive reclaim.")
            return 6
        print(f"post-reclaim residency: {r_post['resident'].mean():.4f}")

        # The victim set is measured, not assumed: it is exactly the set of
        # pages resident before and absent after.
        victims = r_pre["resident"] & (~r_post["resident"])
        n_victims = int(victims.sum())
        print(f"measured victims: {n_victims} pages "
              f"({n_victims / args.pages:.4f} of mapping)")
        tel.event("victims_identified", n_victims=n_victims)

        if n_victims == 0:
            print("\nNo pages were evicted. Recovery cannot be measured.")
            print("On a development host without cgroup reclaim this is expected.")
            results = {"gate": 1, "aborted": "no_victims",
                       "environment_sha256": envinfo["sha256"]}
            (out / "gate1_results.json").write_text(json.dumps(results, indent=2))
            return 5

        # ---------------- STEP 5: measure recovery ----------------
        sched = snapshot_schedule(args.window_T, args.n_snapshots)
        print(f"\nmeasuring recovery at {len(sched)} log-spaced times "
              f"up to T={args.window_T:.0f}s")
        obs_t, obs_S = [], []
        snaps = []
        for target in sched:
            while time.monotonic() - t_reclaim < target:
                time.sleep(0.005)
            p = wl.snapshot_residency(f"rec_{target:.3f}")
            d = parse_residency(p)
            still_evicted = victims & (~d["resident"])
            S = float(still_evicted.sum() / n_victims)
            elapsed = time.monotonic() - t_reclaim
            obs_t.append(elapsed)
            obs_S.append(S)
            snaps.append({"target_s": float(target), "actual_s": elapsed,
                          "S": S, "path": str(p)})
        tel.event("recovery_measured", n_points=len(obs_t))

        obs_path = out / "gate1_observation.json"
        obs_path.write_text(json.dumps({
            "reclaim_mode": args.reclaim_mode,
            "n_victims": n_victims,
            "t": obs_t, "S": obs_S, "snapshots": snaps,
            "pre_resident_frac": float(r_pre["resident"].mean()),
            "post_resident_frac": float(r_post["resident"].mean()),
        }, indent=2))
        prereg.register_observation(out, args.label, obs_path,
                                    extra={"reclaim_mode": args.reclaim_mode,
                                           "n_victims": n_victims})

        # ---------------- STEP 6: score ----------------
        registered = prereg.load_prediction(out, args.label)   # verifies digest
        score = score_prediction(registered, obs_t, obs_S)
        fams = compare_families(np.array(obs_t), np.array(obs_S))

        print("\n--- PREDICTION SCORE ---")
        print(f"  coverage            {score['coverage']:.4f}")
        print(f"  points below/above  {score['n_below']}/{score['n_above']}")
        print(f"  max excess          {score['max_excess']:.5f}")
        print(f"  mean signed dev     {score['mean_signed_dev_from_mid']:+.5f}")
        print(f"  dev vs t corr       {score['dev_vs_t_correlation']:+.4f}")
        print(f"  bracket mean width  {score['bracket_mean_width']:.5f}")
        print(f"\n  {score['diagnosis']}")
        print(f"\n--- KERNEL FAMILY COMPARISON (observed recovery) ---")
        print(f"  best by AICc: {fams['best_by_aicc']}   "
              f"best out-of-sample: {fams['best_by_oos']}")
        for r in fams["rows"]:
            print(f"    {r['family']:16s} rmse={r['rmse_full']:.5f} "
                  f"dAICc={r['delta_aicc']:8.2f} oos_rmse={r['oos_rmse']:.5f}")

        results = {
            "gate": 1,
            "args": vars(args),
            "environment_sha256": envinfo["sha256"],
            "publishable_environment": verdict["publishable"],
            "reclaim_mode": args.reclaim_mode,
            "reclaim_mode_caveat": (
                "proactive reclaim is not accounted as cgroup pressure; valid "
                "for page-dynamics claims only"
                if args.reclaim_mode == "proactive" else
                "natural reclaim generates genuine pressure and PSI"
            ),
            "prediction_sha256": commit["digest"],
            "ledger_head_at_commit": commit["ledger_head"],
            "n_victims": n_victims,
            "score": score,
            "family_comparison": fams,
            "observation": {"t": obs_t, "S": obs_S},
            "gate1_pass": bool(score["coverage"] >= 0.90),
        }
        (out / "gate1_results.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"\nGATE 1: {'PASS' if results['gate1_pass'] else 'FAIL'} "
              f"(coverage {score['coverage']:.3f}, threshold 0.90)")
        return 0 if results["gate1_pass"] else 1

    finally:
        if tel is not None:
            info = tel.stop()
            print(f"\ntelemetry: {info}")
            csv_info = to_csv(tel.samples_path, out / "telemetry" / "samples.csv")
            print(f"telemetry CSV: {csv_info['n_rows']} rows x "
                  f"{csv_info['n_cols']} columns")
        wl.stop()
        if cg is not None:
            cg.destroy()
        led = prereg.Ledger(out).verify()
        print(f"ledger verification: {led}")


if __name__ == "__main__":
    sys.exit(main())
