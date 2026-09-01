"""
randomization.py -- randomization inference over a RECORDED assignment mechanism.

Why not a conditional-independence test
---------------------------------------
The hypothesis is that reclaim history carries information about future damage
beyond instantaneous telemetry. On observational data from a working controller
this is not identifiable: a policy that reclaims from workloads which look
reclaimable makes history correlate with latent fragility, so a partial
correlation or conditional mutual information test detects an effect that does
not exist. This module includes a demonstration of exactly that failure
(`confounding_demo`) so the point can be shown rather than asserted.

Why not a standard Model-X CRT
------------------------------
Model-X conditional randomization testing requires the conditional law of the
treatment given covariates to be known. Under randomized assignment that law IS
known, which is unusual and favourable. But the treatment here is not a single
scalar: it is a sequential history, and the telemetry X_t is itself downstream of
earlier randomized interventions. Resampling a whole history independently of X
would break the temporal structure and invalidate the test.

What this module does instead
-----------------------------
Randomization inference conditional on the recorded mechanism. The assignment at
each epoch is re-drawn from the SAME per-epoch propensity that was actually used,
holding everything not affected by the assignment fixed, and the test statistic
is recomputed. This yields exact finite-sample Type-I control for the SHARP null
that the assignment had no effect on the outcome at any epoch. The sharp null is
stated explicitly rather than assumed, because a weak null about averages would
not be testable this way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class AssignmentRecord:
    """The recorded randomization mechanism for one epoch.

    Storing the propensity that was actually used (rather than assuming 0.5)
    permits propensity to depend on covariates, which the design allows so long
    as it is bounded away from 0 and 1.
    """
    epoch: int
    t_mono: float
    propensity: float          # P(treat = 1) used at this epoch
    assigned: int              # realized treatment, 0/1
    rng_seed: Optional[int] = None
    covariates: Dict[str, float] = field(default_factory=dict)


class SequentialRandomizer:
    """Draws and records treatment assignments from a known mechanism."""

    def __init__(self, seed: int, propensity: float = 0.5,
                 p_min: float = 0.1):
        if not (p_min <= propensity <= 1 - p_min):
            raise ValueError(
                f"propensity {propensity} must lie in [{p_min}, {1-p_min}] so the "
                "mechanism is bounded away from 0 and 1"
            )
        self.seed = int(seed)
        self.propensity = float(propensity)
        self.p_min = float(p_min)
        self.rng = np.random.default_rng(seed)
        self.records: List[AssignmentRecord] = []

    def draw(self, epoch: int, t_mono: float,
             propensity: Optional[float] = None,
             covariates: Optional[Dict[str, float]] = None) -> AssignmentRecord:
        p = self.propensity if propensity is None else float(propensity)
        if not (self.p_min <= p <= 1 - self.p_min):
            raise ValueError(f"propensity {p} violates the [p_min, 1-p_min] bound")
        a = int(self.rng.random() < p)
        rec = AssignmentRecord(epoch=epoch, t_mono=t_mono, propensity=p,
                               assigned=a, covariates=covariates or {})
        self.records.append(rec)
        return rec

    def as_dicts(self) -> List[Dict]:
        return [{"epoch": r.epoch, "t_mono": r.t_mono,
                 "propensity": r.propensity, "assigned": r.assigned,
                 "covariates": r.covariates} for r in self.records]


def randomization_test(
    assigned: Sequence[int],
    propensities: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    n_resample: int = 10000,
    seed: int = 0,
    alternative: str = "two-sided",
) -> Dict[str, object]:
    """Exact randomization test for the sharp null of no assignment effect.

    `statistic` receives an assignment vector and returns a scalar. Under the
    sharp null the outcomes are fixed, so re-drawing assignments from the true
    mechanism and recomputing the statistic gives its exact null distribution.

    The p-value uses the (1 + #{|T*| >= |T|}) / (1 + B) form, which is valid for
    any B and never returns exactly zero -- a p-value of zero would be an
    impossible claim from a finite resample.
    """
    a = np.asarray(assigned, dtype=int)
    p = np.asarray(propensities, dtype=float)
    if a.shape != p.shape:
        raise ValueError("assigned and propensities must have equal length")
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("propensities must be strictly inside (0,1)")

    T_obs = float(statistic(a))
    rng = np.random.default_rng(seed)
    count = 0
    null = np.empty(n_resample)
    for b in range(n_resample):
        a_star = (rng.random(len(p)) < p).astype(int)
        T_b = float(statistic(a_star))
        null[b] = T_b
        if alternative == "two-sided":
            if abs(T_b) >= abs(T_obs) - 1e-15:
                count += 1
        elif alternative == "greater":
            if T_b >= T_obs - 1e-15:
                count += 1
        elif alternative == "less":
            if T_b <= T_obs + 1e-15:
                count += 1
        else:
            raise ValueError(f"unknown alternative {alternative}")

    return {
        "statistic_observed": T_obs,
        "p_value": (1.0 + count) / (1.0 + n_resample),
        "n_resample": n_resample,
        "alternative": alternative,
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)) if n_resample > 1 else 0.0,
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "sharp_null": ("no effect of assignment on the outcome at any epoch; "
                       "outcomes held fixed under resampling"),
    }


def make_partial_corr_statistic(outcome: np.ndarray,
                                covariates: Optional[np.ndarray]) -> Callable:
    """Statistic: partial correlation of assignment with outcome, adjusting for
    covariates by linear projection. Adjusting is not required for validity
    under randomization, but it increases power when covariates explain outcome
    variance.
    """
    y = np.asarray(outcome, float)
    if covariates is None:
        X = np.ones((len(y), 1))
    else:
        C = np.asarray(covariates, float)
        if C.ndim == 1:
            C = C[:, None]
        X = np.hstack([np.ones((len(y), 1)), C])
    # residualize outcome once; it does not change across resamples
    beta_y, *_ = np.linalg.lstsq(X, y, rcond=None)
    ry = y - X @ beta_y

    def stat(a: np.ndarray) -> float:
        av = np.asarray(a, float)
        beta_a, *_ = np.linalg.lstsq(X, av, rcond=None)
        ra = av - X @ beta_a
        sa, sy = ra.std(), ry.std()
        if sa < 1e-12 or sy < 1e-12:
            return 0.0
        return float(np.mean(ra * ry) / (sa * sy))

    return stat


def make_mean_difference_statistic(outcome: np.ndarray) -> Callable:
    y = np.asarray(outcome, float)

    def stat(a: np.ndarray) -> float:
        av = np.asarray(a, dtype=bool)
        if av.all() or (~av).all():
            return 0.0
        return float(y[av].mean() - y[~av].mean())

    return stat


def confounding_demo(n: int = 4000, seed: int = 0) -> Dict[str, object]:
    """Demonstrate that observational data cannot identify the hypothesis.

    Constructs data with EXACTLY ZERO causal effect of history on damage, but an
    observational policy that depends on latent fragility. Reports the partial
    correlation that a naive test would find, and the randomization-test p-value
    under a proper randomized design on the same data-generating process.
    """
    rng = np.random.default_rng(seed)
    frag = rng.normal(0, 1, n)                       # latent fragility
    X = frag + rng.normal(0, 0.6, n)                 # noisy telemetry proxy
    D = 1.2 * X + rng.normal(0, 1, n)                # damage: NO effect of H

    # observational policy: reclaim from workloads that look robust
    H_obs = (-0.9 * frag + rng.normal(0, 0.5, n) > 0).astype(int)
    stat_obs = make_partial_corr_statistic(D, X)
    pc_obs = stat_obs(H_obs)

    # randomized policy on the same DGP
    p = 0.5
    H_rand = (rng.random(n) < p).astype(int)
    res = randomization_test(H_rand, np.full(n, p),
                             make_partial_corr_statistic(D, X),
                             n_resample=2000, seed=seed + 1)

    return {
        "true_causal_effect": 0.0,
        "observational_partial_corr": float(pc_obs),
        "observational_verdict": ("nonzero: a naive conditional-independence "
                                  "test falsely detects hysteresis"),
        "randomized_p_value": res["p_value"],
        "randomized_verdict": ("calibrated: randomization removes the "
                               "confounding path through latent fragility"),
    }


def power_curve(effect_sizes: Sequence[float], n: int = 2000,
                n_sim: int = 200, n_resample: int = 500,
                seed: int = 0) -> List[Dict[str, float]]:
    """Estimate detection power across effect sizes.

    Needed before the real assay: an underpowered design that reports a null is
    uninformative, and the negative control of Theorem 6 is only meaningful if
    the design COULD have detected an effect of the relevant size.
    """
    rng = np.random.default_rng(seed)
    out = []
    for eff in effect_sizes:
        rejects = 0
        for s in range(n_sim):
            X = rng.normal(0, 1, n)
            H = (rng.random(n) < 0.5).astype(int)
            D = 1.2 * X + eff * H + rng.normal(0, 1, n)
            r = randomization_test(H, np.full(n, 0.5),
                                   make_partial_corr_statistic(D, X),
                                   n_resample=n_resample, seed=int(rng.integers(1 << 30)))
            if r["p_value"] < 0.05:
                rejects += 1
        out.append({"effect_size": float(eff), "power": rejects / n_sim,
                    "n": n, "n_sim": n_sim})
    return out
