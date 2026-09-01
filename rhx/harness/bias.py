"""
bias.py -- explicit controls against the ways this experiment could fool us.

Every mechanism here exists because a specific bias would otherwise be available.
Each is named with the bias it removes.

1. ANALYST DEGREES OF FREEDOM
   Exclusion rules, pass criteria and the trial count are frozen in a protocol
   file and hashed into the pre-registration ledger BEFORE any data is collected.
   `Protocol.freeze()` refuses to overwrite an existing protocol.

2. SELECTION BIAS IN TRIAL EXCLUSION
   A trial may only be discarded for a reason declared in the frozen protocol
   (OOM kill, workload death, scheduler lag over budget, environment change).
   `apply_exclusions` records every decision with its rule, so the exclusion rate
   is auditable and post-hoc exclusion is impossible.

3. ARM ORDER AND MACHINE DRIFT
   Arms are INTERLEAVED trial by trial rather than run in blocks, and the
   within-trial order is randomized from a recorded seed. Blocked designs
   confound arm with session drift (page-cache warmth, thermal state, background
   daemons); interleaving does not.

4. ANALYST BLINDING
   `Blinder` maps arm labels to opaque tokens and writes the key to a separate
   file. Analysis can be run and figures produced against tokens; the key is
   applied only at the end. This removes the temptation to keep tuning until the
   expected arm wins.

5. OPTIONAL STOPPING
   The trial count is frozen in the protocol. `check_stopping` refuses to report
   an aggregate if fewer or more trials were run than declared, unless the
   deviation is itself recorded with a reason.

6. CARRYOVER BETWEEN TRIALS
   Each trial gets a fresh workload process, a fresh backing file and a fresh
   cgroup. `washout_s` enforces an idle gap so debt from trial k does not leak
   into trial k+1, and `check_drift` tests whether trial index predicts the
   outcome, which would indicate carryover survived the washout.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------ #
# 1. Frozen protocol                                                  #
# ------------------------------------------------------------------ #

@dataclass
class Protocol:
    """Everything that must be decided before data collection."""
    name: str
    n_trials: int
    primary_endpoint: str
    pass_criteria: Dict[str, float]
    exclusion_rules: List[str]
    washout_s: float
    max_sched_lag_frac: float      # fraction of the shortest inter-access time
    arm_order: str = "interleaved_randomized"
    secondary_endpoints: List[str] = field(default_factory=list)
    notes: str = ""

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def freeze(self, out_dir: Path) -> Dict[str, Any]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "protocol.json"
        if p.exists():
            existing = json.loads(p.read_text())
            if existing.get("_digest") != self.digest():
                raise FileExistsError(
                    f"a DIFFERENT protocol is already frozen at {p}. "
                    "Changing the protocol after data collection has begun "
                    "invalidates the experiment. Start a new output directory."
                )
            return existing
        payload = asdict(self)
        payload["_digest"] = self.digest()
        payload["_frozen_at_unix"] = time.time()
        payload["_frozen_at_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return payload


DEFAULT_EXCLUSION_RULES = [
    "oom_kill_delta > 0: kernel OOM-killed a process during reclaim",
    "workload_survived is False: workload process died during the trial",
    "sched_lag_exceeded: rategen max_lag exceeded max_sched_lag_frac",
    "environment_sha256 changed mid-run: machine configuration was altered",
    "n_victims == 0: reclaim evicted nothing, so recovery is undefined",
    "returncode not in (0, 1): harness error rather than a scientific result",
]


# ------------------------------------------------------------------ #
# 2. Exclusion with an audit trail                                    #
# ------------------------------------------------------------------ #

def apply_exclusions(trials: List[Dict[str, Any]],
                     protocol: Dict[str, Any],
                     env_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Apply ONLY the frozen exclusion rules. Every decision is recorded."""
    allowed = set(r.split(":")[0].strip() for r in protocol["exclusion_rules"])
    kept, dropped = [], []

    for t in trials:
        res = t.get("results") or {}
        reasons = []

        if "oom_kill_delta > 0" in allowed:
            ok = res.get("oom_kill_delta", 0) or 0
            if ok > 0 or t.get("returncode") == 6:
                reasons.append("oom_kill_delta > 0")
        if "workload_survived is False" in allowed:
            if res.get("workload_survived") is False:
                reasons.append("workload_survived is False")
        if "sched_lag_exceeded" in allowed and res.get("sched_lag_exceeded"):
            reasons.append("sched_lag_exceeded")
        if "environment_sha256 changed mid-run" in allowed and env_sha256:
            e = res.get("environment_sha256")
            if e and e != env_sha256:
                reasons.append("environment_sha256 changed mid-run")
        if "n_victims == 0" in allowed and res.get("n_victims") == 0:
            reasons.append("n_victims == 0")
        if "returncode not in (0, 1)" in allowed:
            if t.get("returncode") not in (0, 1):
                reasons.append("returncode not in (0, 1)")

        (dropped if reasons else kept).append(
            {**t, "_exclusion_reasons": reasons})

    return {
        "n_input": len(trials),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "exclusion_rate": (len(dropped) / len(trials)) if trials else 0.0,
        "kept": kept,
        "dropped": [{"trial": d.get("trial"), "seed": d.get("seed"),
                     "reasons": d["_exclusion_reasons"]} for d in dropped],
        "rules_applied": sorted(allowed),
        "warning": ("exclusion rate above 20% -- report it prominently and "
                    "explain the cause"
                    if trials and len(dropped) / len(trials) > 0.2 else None),
    }


# ------------------------------------------------------------------ #
# 3. Interleaved randomized arm order                                 #
# ------------------------------------------------------------------ #

def interleaved_order(arms: Sequence[str], n_trials: int,
                      seed: int) -> List[Tuple[int, List[str]]]:
    """Return, per trial, a randomized permutation of the arms.

    Arms are run within EVERY trial rather than in blocks, so any drift in
    machine state affects all arms nearly equally and cannot masquerade as an
    arm effect. The permutation is drawn from a recorded seed.
    """
    rng = np.random.default_rng(seed)
    return [(k, [arms[i] for i in rng.permutation(len(arms))])
            for k in range(n_trials)]


# ------------------------------------------------------------------ #
# 4. Analyst blinding                                                 #
# ------------------------------------------------------------------ #

class Blinder:
    """Replace arm labels with opaque tokens so analysis is blind."""

    def __init__(self, arms: Sequence[str], seed: int):
        rng = np.random.default_rng(seed)
        toks = [f"ARM_{c}" for c in "ABCDEFGH"][:len(arms)]
        perm = rng.permutation(len(arms))
        self.to_token = {arms[i]: toks[j] for j, i in enumerate(perm)}
        self.to_arm = {v: k for k, v in self.to_token.items()}

    def blind(self, arm: str) -> str:
        return self.to_token[arm]

    def write_key(self, out_dir: Path) -> Path:
        """Write the unblinding key to a file the analyst should not open until
        the analysis is finalized."""
        p = Path(out_dir) / "UNBLINDING_KEY.json"
        p.write_text(json.dumps({
            "_warning": ("Do not open until analysis is complete and committed. "
                         "Opening early defeats the blinding."),
            "token_to_arm": self.to_arm,
        }, indent=2))
        return p


# ------------------------------------------------------------------ #
# 5. Optional stopping                                                #
# ------------------------------------------------------------------ #

def check_stopping(n_run: int, protocol: Dict[str, Any],
                   reason: Optional[str] = None) -> Dict[str, Any]:
    declared = int(protocol["n_trials"])
    ok = (n_run == declared)
    return {
        "declared_n": declared,
        "actual_n": n_run,
        "matches": ok,
        "deviation_reason": reason,
        "valid": ok or bool(reason),
        "note": (None if ok else
                 "Trial count differs from the frozen protocol. This is only "
                 "acceptable with a recorded reason; stopping early because the "
                 "result looked good is optional stopping and inflates the "
                 "false-positive rate."),
    }


# ------------------------------------------------------------------ #
# 6. Carryover and drift                                              #
# ------------------------------------------------------------------ #

def check_drift(trial_index: Sequence[int],
                outcome: Sequence[float]) -> Dict[str, Any]:
    """Test whether trial ORDER predicts the outcome.

    A significant relationship means carryover survived the washout, or the
    machine drifted over the session. Either way the trials are not exchangeable
    and the aggregate is not trustworthy.
    """
    x = np.asarray(trial_index, float)
    y = np.asarray(outcome, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "r": None,
                "verdict": "too few valid trials to test drift"}
    r = float(np.corrcoef(x, y)[0, 1])
    # permutation test: exact under exchangeability of trial order
    rng = np.random.default_rng(0)
    null = np.array([abs(np.corrcoef(rng.permutation(x), y)[0, 1])
                     for _ in range(5000)])
    p = float((1 + np.sum(null >= abs(r))) / (1 + len(null)))
    return {
        "n": int(len(x)),
        "r": r,
        "p_value": p,
        "verdict": ("no detectable order effect" if p >= 0.05 else
                    "ORDER EFFECT DETECTED: trials are not exchangeable. "
                    "Increase washout, check for thermal or cache carryover, "
                    "and do not pool these trials."),
    }


def balance_report(assignments: Sequence[str],
                   covariates: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    """Check that randomization actually balanced the covariates.

    Reports standardized mean differences. A |SMD| above 0.25 on any covariate
    means the arms differ in something other than the treatment, and the
    comparison is confounded despite randomization.
    """
    a = np.asarray(assignments)
    arms = sorted(set(a.tolist()))
    out: Dict[str, Any] = {"arms": arms, "covariates": {}}
    if len(arms) != 2:
        out["note"] = "SMD is defined here for two arms only"
        return out
    for name, vals in covariates.items():
        v = np.asarray(vals, float)
        g0, g1 = v[a == arms[0]], v[a == arms[1]]
        if len(g0) < 2 or len(g1) < 2:
            out["covariates"][name] = {"smd": None, "note": "too few"}
            continue
        sp = np.sqrt((g0.var(ddof=1) + g1.var(ddof=1)) / 2)
        smd = float((g1.mean() - g0.mean()) / sp) if sp > 0 else 0.0
        out["covariates"][name] = {
            "mean_" + arms[0]: float(g0.mean()),
            "mean_" + arms[1]: float(g1.mean()),
            "smd": smd,
            "balanced": bool(abs(smd) <= 0.25),
        }
    bad = [k for k, v in out["covariates"].items()
           if v.get("balanced") is False]
    out["imbalanced"] = bad
    out["verdict"] = ("balanced" if not bad else
                      f"IMBALANCED on {bad}: arms differ in more than the "
                      "treatment; the comparison is confounded")
    return out


# ------------------------------------------------------------------ #
# Preflight                                                           #
# ------------------------------------------------------------------ #

def preflight(env: Dict[str, Any], protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Everything that must be true before the first trial runs."""
    blockers: List[str] = []
    warnings: List[str] = []

    v = env.get("_verdict", {})
    blockers.extend(v.get("blockers", []))
    warnings.extend(v.get("warnings", []))

    thp = (env.get("thp", {}).get("enabled") or "")
    if "[always]" in thp:
        blockers.append(
            "THP is 'always'. Huge pages change the reclaim unit and make "
            "per-4KiB mincore residency ambiguous. Set to 'never' or 'madvise' "
            "for characterization runs."
        )
    if not env.get("cpu", {}).get("governors_uniform"):
        warnings.append("CPU governors are not uniform; timing varies by core.")
    if env.get("cpu", {}).get("intel_no_turbo") == "0":
        warnings.append("Turbo is enabled; per-trial timing will drift with "
                        "thermal state. Consider disabling for reproducibility.")
    if protocol.get("n_trials", 0) < 10:
        warnings.append(
            f"n_trials={protocol.get('n_trials')} is small. Per-trial coverage "
            "variance is the dominant uncertainty; 20 or more is recommended."
        )
    if protocol.get("washout_s", 0) <= 0:
        blockers.append("washout_s must be positive or debt carries between "
                        "trials and they are not independent.")

    return {"ready": len(blockers) == 0, "blockers": blockers,
            "warnings": warnings}
