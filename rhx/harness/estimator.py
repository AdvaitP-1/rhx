"""
estimator.py -- estimate the access-rate density f(lambda) and derive the
predicted recovery kernel S(t) over a declared operating window.

Why the window matters
----------------------
The theory's recovery kernel is a Laplace transform,

    S(t) = E_q[ exp(-lambda t) ]

where q is the VICTIM-CONDITIONED rate density (not the unconditional f, because
reclaim selects cold pages preferentially).

Estimating f near lambda -> 0 requires observation windows long compared to
1/lambda for the coldest pages, over which the static-rate assumption fails.
These two requirements conflict.

Resolution: over a bounded window [0, T], split at lambda_min = 1/T:

    S(t) = INT_{lambda_min}^{inf} e^{-lt} q dl  +  INT_0^{lambda_min} e^{-lt} q dl

The first term is estimable. The second is NOT equal to the frozen mass
Q(lambda_min) -- that would be an approximation with worst-case error
Q*(1-e^{-1}) over the window. Because e^{-lt} is monotone in l, the second term
is instead BRACKETED with no assumption about the shape of the density below the
cutoff:

    Q(lambda_min)*exp(-t/T)  <=  INT_0^{lambda_min} e^{-lt} q dl  <=  Q(lambda_min)

Both bounds are attainable: the lower if all frozen mass sits exactly at
lambda_min, the upper if it all sits at rate 0. We therefore register a BRACKET
[S_lower, S_upper] that is guaranteed to contain the truth, rather than a point
estimate that is merely close to it. The bracket width, Q*(1 - e^{-t/T}), is the
honest cost of not resolving the cold tail; it shrinks as T is shortened because
fewer pages then fall below the cutoff. That tradeoff is reported rather than
hidden.

Estimator sources
-----------------
  "oracle"  ground-truth rates from the synthetic workload. Gate 0 reference
            only; never available for a real application.
  "counts"  per-page access counts over a known window, from the workload's own
            instrumentation. Poisson rate MLE is count/window.
  "idle"    idle-page tracking: sample the idle bitmap at intervals and count
            the fraction of intervals in which a page was accessed. This is a
            BINARY per-interval observation, not a count, so the estimator is
            different -- see estimate_from_idle_samples.
  "damon"   DAMON region-level access frequencies. Region aggregation means the
            observable is a region rate, not a page rate; the mapping is
            recorded so the aggregation bias is visible in Gate 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------- #
# Rate estimates                                                     #
# ---------------------------------------------------------------- #

@dataclass
class RateEstimate:
    """Per-page rate estimates plus the metadata needed to defend them."""
    lam: np.ndarray                # estimated rate per page (len = n_pages)
    source: str                    # oracle | counts | idle | damon
    window_s: float                # observation window used
    n_pages: int
    params: Dict[str, object] = field(default_factory=dict)
    # observation-limited: pages whose estimate is not resolvable in this window
    censored_mask: Optional[np.ndarray] = None

    def summary(self) -> Dict[str, float]:
        lam = self.lam
        pos = lam[lam > 0]
        return {
            "n_pages": int(self.n_pages),
            "mean": float(lam.mean()),
            "std": float(lam.std(ddof=1)) if len(lam) > 1 else 0.0,
            "cv": float(lam.std(ddof=1) / lam.mean()) if lam.mean() > 0 and len(lam) > 1 else float("nan"),
            "min": float(lam.min()),
            "p05": float(np.quantile(lam, 0.05)),
            "median": float(np.median(lam)),
            "p95": float(np.quantile(lam, 0.95)),
            "max": float(lam.max()),
            "frac_zero": float((lam <= 0).mean()),
            "geomean_pos": float(np.exp(np.log(pos).mean())) if len(pos) else float("nan"),
        }


def estimate_from_counts(counts: np.ndarray, window_s: float) -> RateEstimate:
    """Poisson rate MLE from event counts over a known window.

    lambda_hat = count / window. This is the MLE and is unbiased. Its variance
    is lambda/window, so low-rate pages are noisy; the censored mask marks pages
    with zero observed events, for which the MLE is 0 but the true rate is only
    bounded above by roughly 3/window at 95% confidence.
    """
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    counts = np.asarray(counts, dtype=float)
    lam = counts / window_s
    return RateEstimate(
        lam=lam,
        source="counts",
        window_s=float(window_s),
        n_pages=len(lam),
        params={"mle": "count/window",
                "upper_bound_zero_count_95": 3.0 / window_s},
        censored_mask=(counts == 0),
    )


def estimate_from_idle_samples(hit_counts: np.ndarray,
                               n_samples: int,
                               interval_s: float) -> RateEstimate:
    """Estimate rates from idle-page-tracking style BINARY samples.

    Idle page tracking tells you whether a page was accessed since the bitmap
    was last cleared -- not how many times. Over an interval of length d, the
    probability a Poisson(lambda) page is accessed at least once is

        p = 1 - exp(-lambda d)

    so with k hits out of n intervals, p_hat = k/n and

        lambda_hat = -ln(1 - p_hat) / d.

    This SATURATES: a page hit in every interval gives p_hat = 1 and an infinite
    estimate. Such pages are censored from above, which is recorded. The
    saturation rate is -ln(1/(2n))/d, used as a conservative cap.
    """
    if interval_s <= 0 or n_samples <= 0:
        raise ValueError("interval_s and n_samples must be positive")
    k = np.asarray(hit_counts, dtype=float)
    p = k / float(n_samples)
    # cap p away from 1 so the log is finite; equivalent to a +0.5 continuity
    # correction on the top cell
    p_cap = 1.0 - 1.0 / (2.0 * n_samples)
    saturated = p >= p_cap
    p_adj = np.minimum(p, p_cap)
    lam = -np.log1p(-p_adj) / interval_s
    return RateEstimate(
        lam=lam,
        source="idle",
        window_s=float(n_samples * interval_s),
        n_pages=len(lam),
        params={"interval_s": interval_s, "n_samples": n_samples,
                "saturation_rate": float(-math.log(1.0 / (2 * n_samples)) / interval_s),
                "n_saturated": int(saturated.sum())},
        censored_mask=saturated,
    )


def estimate_oracle(true_lam: np.ndarray) -> RateEstimate:
    lam = np.asarray(true_lam, dtype=float)
    return RateEstimate(lam=lam, source="oracle", window_s=float("inf"),
                        n_pages=len(lam), params={"note": "ground truth"},
                        censored_mask=np.zeros(len(lam), dtype=bool))


# ---------------------------------------------------------------- #
# Victim conditioning                                                #
# ---------------------------------------------------------------- #

def victim_weights(lam: np.ndarray, kind: str = "coldest_first",
                   theta: Optional[float] = None,
                   frac: Optional[float] = None) -> np.ndarray:
    """Victim-selection kernel v(lambda), returned as per-page weights in [0,1].

    The theory conditions the recovery kernel on which pages were ACTUALLY
    evicted. Three kernels are supported:

      "coldest_first"  hard threshold at theta (or the frac-th quantile).
                       Idealized LRU; the analytic case.
      "uniform"        every resident page equally likely. The null victim
                       policy; used to isolate the effect of selection bias.
      "softmin"        smooth cold-preference, exp(-lambda/theta), normalized.
                       A tractable stand-in for MGLRU's generation-based choice,
                       which is not a hard threshold.

    Returning weights rather than a boolean mask lets the same code express both
    the analytic threshold and a smooth approximation of real victim selection.
    """
    lam = np.asarray(lam, dtype=float)
    if kind == "uniform":
        return np.ones_like(lam)
    if theta is None:
        if frac is None:
            raise ValueError("victim_weights needs theta or frac")
        theta = float(np.quantile(lam, frac))
    if kind == "coldest_first":
        return (lam <= theta).astype(float)
    if kind == "softmin":
        w = np.exp(-lam / max(theta, 1e-12))
        return w / w.max() if w.max() > 0 else w
    raise ValueError(f"unknown victim kernel: {kind}")


# ---------------------------------------------------------------- #
# Predicted recovery kernel                                          #
# ---------------------------------------------------------------- #

@dataclass
class KernelPrediction:
    t_grid: np.ndarray
    S_pred: np.ndarray
    S_lo: np.ndarray
    S_hi: np.ndarray
    window_T_s: float
    lambda_min: float
    frozen_fraction: float
    n_victims_effective: float
    details: Dict[str, object] = field(default_factory=dict)


def predict_recovery(
    rate_est: RateEstimate,
    victim_w: np.ndarray,
    window_T_s: float,
    n_grid: int = 40,
    n_boot: int = 400,
    seed: int = 0,
) -> KernelPrediction:
    """Predict S(t) on [0, T] from estimated rates and a victim kernel.

    The returned S_lo / S_hi form a RIGOROUS BRACKET, combining two distinct
    sources of uncertainty:

      (a) Cold-tail indeterminacy. Pages below lambda_min = 1/T are not resolved
          in this window. Their contribution is bracketed by
          Q*exp(-t/T) <= contribution <= Q, with both ends attainable. This is a
          bound, not a confidence interval: the truth is guaranteed inside it.

      (b) Finite-page sampling error on the RESOLVED pages, from a
          nonparametric bootstrap over pages.

    The two are combined conservatively (bootstrap interval on the resolved part,
    widened by the full cold-tail bracket). S_pred is the midpoint of the
    cold-tail bracket and is reported for reference only; the bracket is what
    the prediction test should be judged against.

    Rate-estimation error for an INDIVIDUAL page is not folded in here; it is
    reported separately as frac_censored, because it has different structure and
    merging it would obscure which source dominates.
    """
    if window_T_s <= 0:
        raise ValueError("window_T_s must be positive")
    lam = np.asarray(rate_est.lam, dtype=float)
    w = np.asarray(victim_w, dtype=float)
    if lam.shape != w.shape:
        raise ValueError("rate and victim-weight arrays must have equal length")

    lambda_min = 1.0 / window_T_s
    total_w = w.sum()
    if total_w <= 0:
        raise ValueError("victim weights sum to zero; no pages selected")

    resolved = lam >= lambda_min
    frozen_w = w[~resolved].sum()
    frozen_fraction = float(frozen_w / total_w)

    lam_r = lam[resolved]
    w_r = w[resolved]

    # t=0 is included so the prediction is anchored at S(0)=1 by construction.
    t_grid = np.linspace(0.0, window_T_s, n_grid)

    def resolved_part(lam_v: np.ndarray, w_v: np.ndarray, tot: float) -> np.ndarray:
        """Contribution of pages at or above lambda_min. Exactly computable."""
        if tot <= 0:
            return np.full_like(t_grid, np.nan)
        M = np.exp(-np.outer(t_grid, lam_v))       # (n_grid, n_resolved)
        return (M @ w_v) / tot

    # Cold-tail bracket: monotonicity of exp(-lt) in l gives attainable bounds.
    #   all frozen mass at lambda_min -> Q*exp(-t/T)   (fastest possible decay)
    #   all frozen mass at rate 0     -> Q             (no decay at all)
    cold_lo = (frozen_w / total_w) * np.exp(-t_grid / window_T_s)
    cold_hi = np.full_like(t_grid, frozen_w / total_w)

    res_pred = resolved_part(lam_r, w_r, total_w)
    S_pred = res_pred + 0.5 * (cold_lo + cold_hi)   # midpoint, reference only

    # Bootstrap the RESOLVED part for finite-page sampling error.
    rng = np.random.default_rng(seed)
    n = len(lam)
    boots = np.empty((n_boot, len(t_grid)))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        lb, wb = lam[idx], w[idx]
        tot_b = wb.sum()
        if tot_b <= 0:
            boots[b] = np.nan
            continue
        res_b = lb >= lambda_min
        boots[b] = resolved_part(lb[res_b], wb[res_b], tot_b)
    res_lo = np.nanquantile(boots, 0.025, axis=0)
    res_hi = np.nanquantile(boots, 0.975, axis=0)

    # Combine conservatively: widest resolved interval plus full cold bracket.
    S_lo = res_lo + cold_lo
    S_hi = res_hi + cold_hi
    # S(0) must be exactly 1 by construction; enforce to kill float drift.
    S_pred[0] = 1.0
    S_lo[0] = 1.0
    S_hi[0] = 1.0

    return KernelPrediction(
        t_grid=t_grid,
        S_pred=S_pred,
        S_lo=S_lo,
        S_hi=S_hi,
        window_T_s=float(window_T_s),
        lambda_min=float(lambda_min),
        frozen_fraction=frozen_fraction,
        n_victims_effective=float(total_w),
        details={
            "rate_source": rate_est.source,
            "rate_window_s": rate_est.window_s,
            "n_pages": int(n),
            "n_resolved": int(resolved.sum()),
            "n_frozen": int((~resolved).sum()),
            "frac_censored": float(rate_est.censored_mask.mean())
                             if rate_est.censored_mask is not None else 0.0,
            "n_boot": n_boot,
            "boot_seed": seed,
            "cold_bracket_width_at_T": float(frozen_w / total_w) * (1.0 - math.exp(-1.0)),
            "bracket_note": "S_lo/S_hi bound the truth; cold tail is a bound, resolved part is a 95% bootstrap CI",
        },
    )


def to_registerable(pred: KernelPrediction, estimator_desc: Dict[str, object]) -> Dict[str, object]:
    """Convert a KernelPrediction to the dict accepted by prereg.register_prediction."""
    return {
        "window_T_s": float(pred.window_T_s),
        "lambda_min": float(pred.lambda_min),
        "frozen_fraction": float(pred.frozen_fraction),
        "t_grid": [float(x) for x in pred.t_grid],
        "S_pred": [float(x) for x in pred.S_pred],
        "S_pred_lo": [float(x) for x in pred.S_lo],
        "S_pred_hi": [float(x) for x in pred.S_hi],
        "estimator": estimator_desc,
        "details": {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in pred.details.items()},
    }


# ---------------------------------------------------------------- #
# Gate 0 comparison: estimated vs ground-truth density               #
# ---------------------------------------------------------------- #

def compare_densities(true_lam: np.ndarray, est_lam: np.ndarray,
                      window_T_s: float) -> Dict[str, float]:
    """Quantify how well an estimator recovers a KNOWN rate density.

    Reports quantities the experiment actually depends on, not generic
    goodness-of-fit:
      - error in the quantities that enter S(t): the frozen fraction and the
        resolved-region mean of exp(-lam t)
      - quantile errors across the distribution
      - CV error, because Theorem 6's negative control is a statement about
        dispersion
    """
    t = np.asarray(true_lam, float)
    e = np.asarray(est_lam, float)
    if t.shape != e.shape:
        raise ValueError("true and estimated rate arrays must have equal length")
    lm = 1.0 / window_T_s

    out: Dict[str, float] = {}
    out["lambda_min"] = lm
    out["frozen_true"] = float((t < lm).mean())
    out["frozen_est"] = float((e < lm).mean())
    out["frozen_abs_err"] = abs(out["frozen_est"] - out["frozen_true"])

    for q in (0.05, 0.25, 0.5, 0.75, 0.95):
        qt = float(np.quantile(t, q))
        qe = float(np.quantile(e, q))
        out[f"q{int(q*100):02d}_true"] = qt
        out[f"q{int(q*100):02d}_est"] = qe
        out[f"q{int(q*100):02d}_rel_err"] = abs(qe - qt) / qt if qt > 0 else float("nan")

    cv_t = float(t.std(ddof=1) / t.mean()) if t.mean() > 0 else float("nan")
    cv_e = float(e.std(ddof=1) / e.mean()) if e.mean() > 0 else float("nan")
    out["cv_true"] = cv_t
    out["cv_est"] = cv_e
    out["cv_rel_err"] = abs(cv_e - cv_t) / cv_t if cv_t > 0 else float("nan")

    # The functional that actually matters: S(t) computed from each.
    tg = np.linspace(0, window_T_s, 25)
    S_t = np.array([np.mean(np.exp(-t * x)) for x in tg])
    S_e = np.array([np.mean(np.exp(-e * x)) for x in tg])
    out["S_max_abs_err"] = float(np.max(np.abs(S_t - S_e)))
    out["S_mean_abs_err"] = float(np.mean(np.abs(S_t - S_e)))
    out["S_rmse"] = float(np.sqrt(np.mean((S_t - S_e) ** 2)))
    return out
