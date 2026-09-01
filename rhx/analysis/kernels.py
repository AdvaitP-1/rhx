"""
kernels.py -- fit and compare recovery-kernel families, and score a
pre-registered prediction against an observation.

The paper's Section 8.2 requires out-of-sample comparison against exponential,
multi-exponential, power-law and non-parametric alternatives. Model comparison
uses AICc (small-sample corrected) alongside held-out error, because AIC alone
rewards the flexible non-parametric fit and would mislead.

Scoring a registered prediction
-------------------------------
The prediction is a BRACKET [S_lo, S_hi] guaranteed to contain the truth if the
theory and the estimate are both correct. The primary score is therefore
coverage: the fraction of grid points at which the observed S(t) falls inside
the bracket. A coverage well below 1 falsifies either the theory or the
estimator, and the two are distinguished by whether the miss is a constant
offset (estimator: wrong frozen mass) or a shape difference (theory: wrong
kernel family).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize


# ---------------------------------------------------------------- #
# Kernel families                                                    #
# ---------------------------------------------------------------- #

def k_exponential(t: np.ndarray, tau: float) -> np.ndarray:
    return np.exp(-t / max(tau, 1e-12))


def k_power(t: np.ndarray, tau: float, zeta: float) -> np.ndarray:
    return (1.0 + t / max(tau, 1e-12)) ** (-max(zeta, 1e-9))


def k_biexp(t: np.ndarray, a: float, tau1: float, tau2: float) -> np.ndarray:
    a = min(max(a, 0.0), 1.0)
    return a * np.exp(-t / max(tau1, 1e-12)) + (1 - a) * np.exp(-t / max(tau2, 1e-12))


def k_stretched(t: np.ndarray, tau: float, beta: float) -> np.ndarray:
    """Stretched exponential (KWW). Included because it is the usual empirical
    competitor to a power law and can mimic it over a bounded window."""
    beta = min(max(beta, 1e-3), 2.0)
    return np.exp(-((t / max(tau, 1e-12)) ** beta))


FAMILIES: Dict[str, Dict] = {
    "exponential": {
        "fn": lambda t, p: k_exponential(t, p[0]),
        "p0": [10.0], "bounds": ([1e-3], [1e6]), "names": ["tau"],
    },
    "power_law": {
        "fn": lambda t, p: k_power(t, p[0], p[1]),
        "p0": [10.0, 1.0], "bounds": ([1e-3, 1e-3], [1e6, 50.0]),
        "names": ["tau", "zeta"],
    },
    "biexponential": {
        "fn": lambda t, p: k_biexp(t, p[0], p[1], p[2]),
        "p0": [0.5, 2.0, 60.0], "bounds": ([0.0, 1e-3, 1e-3], [1.0, 1e6, 1e6]),
        "names": ["a", "tau_fast", "tau_slow"],
    },
    "stretched_exp": {
        "fn": lambda t, p: k_stretched(t, p[0], p[1]),
        "p0": [10.0, 0.5], "bounds": ([1e-3, 1e-3], [1e6, 2.0]),
        "names": ["tau", "beta"],
    },
}


@dataclass
class Fit:
    family: str
    params: Dict[str, float]
    sse: float
    rmse: float
    n: int
    k: int
    aic: float
    aicc: float
    converged: bool
    message: str

    def predict(self, t: np.ndarray) -> np.ndarray:
        spec = FAMILIES[self.family]
        p = [self.params[n] for n in spec["names"]]
        return spec["fn"](np.asarray(t, float), p)


def fit_family(t: np.ndarray, S: np.ndarray, family: str,
               weights: Optional[np.ndarray] = None) -> Fit:
    """Least-squares fit of one kernel family.

    S(0) = 1 is a structural constraint of every family here (each returns 1 at
    t=0 by construction), so it is not fitted, which keeps the parameter count
    honest for the information criteria.
    """
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    if t.shape != S.shape:
        raise ValueError("t and S must have equal length")
    spec = FAMILIES[family]
    w = np.ones_like(S) if weights is None else np.asarray(weights, float)

    def resid(p):
        return (spec["fn"](t, p) - S) * np.sqrt(w)

    try:
        r = optimize.least_squares(resid, spec["p0"], bounds=spec["bounds"],
                                   max_nfev=20000)
        p = r.x
        converged = r.success
        msg = r.message
    except Exception as e:                      # pragma: no cover
        p = np.array(spec["p0"], float)
        converged = False
        msg = f"{type(e).__name__}: {e}"

    pred = spec["fn"](t, p)
    sse = float(np.sum(w * (pred - S) ** 2))
    n = len(t)
    k = len(p)
    # Gaussian-likelihood AIC with unknown variance
    sigma2 = max(sse / n, 1e-300)
    aic = n * math.log(sigma2) + 2 * k
    denom = n - k - 1
    aicc = aic + (2 * k * (k + 1) / denom) if denom > 0 else float("inf")
    return Fit(family=family,
               params={nm: float(v) for nm, v in zip(spec["names"], p)},
               sse=sse, rmse=float(math.sqrt(sse / n)), n=n, k=k,
               aic=float(aic), aicc=float(aicc),
               converged=bool(converged), message=str(msg))


def compare_families(t: np.ndarray, S: np.ndarray,
                     holdout_frac: float = 0.3,
                     seed: int = 0) -> Dict[str, object]:
    """Fit every family and rank by AICc and by held-out error.

    The holdout is a contiguous TAIL split, not a random split: the scientific
    question is whether a family fitted on early recovery predicts late
    recovery, and random splits leak neighbouring points and make every family
    look good.
    """
    t = np.asarray(t, float)
    S = np.asarray(S, float)
    n = len(t)
    n_train = max(3, int(round(n * (1.0 - holdout_frac))))
    t_tr, S_tr = t[:n_train], S[:n_train]
    t_te, S_te = t[n_train:], S[n_train:]

    rows = []
    for fam in FAMILIES:
        f_all = fit_family(t, S, fam)
        f_tr = fit_family(t_tr, S_tr, fam)
        if len(t_te) > 0:
            pred_te = f_tr.predict(t_te)
            oos_rmse = float(np.sqrt(np.mean((pred_te - S_te) ** 2)))
            oos_max = float(np.max(np.abs(pred_te - S_te)))
        else:
            oos_rmse = float("nan")
            oos_max = float("nan")
        rows.append({
            "family": fam,
            "params_full": f_all.params,
            "rmse_full": f_all.rmse,
            "aicc_full": f_all.aicc,
            "k": f_all.k,
            "converged": f_all.converged,
            "params_train": f_tr.params,
            "oos_rmse": oos_rmse,
            "oos_max_abs_err": oos_max,
        })

    best_aicc = min(rows, key=lambda r: r["aicc_full"])["aicc_full"]
    for r in rows:
        r["delta_aicc"] = r["aicc_full"] - best_aicc

    return {
        "n_total": n, "n_train": n_train, "n_holdout": len(t_te),
        "holdout_is_tail": True,
        "rows": sorted(rows, key=lambda r: r["aicc_full"]),
        "best_by_aicc": min(rows, key=lambda r: r["aicc_full"])["family"],
        "best_by_oos": min(rows, key=lambda r: (r["oos_rmse"]
                                                if np.isfinite(r["oos_rmse"])
                                                else float("inf")))["family"],
    }


# ---------------------------------------------------------------- #
# Scoring a registered prediction                                    #
# ---------------------------------------------------------------- #

def score_prediction(pred: Dict[str, object],
                     t_obs: Sequence[float],
                     S_obs: Sequence[float]) -> Dict[str, object]:
    """Score an observation against a pre-registered bracket.

    Primary metric is COVERAGE: the fraction of observed points inside
    [S_lo, S_hi]. Secondary metrics diagnose the failure mode when coverage is
    low:

      - signed mean deviation from the bracket midpoint: a roughly constant
        offset implicates the estimator's frozen mass, since that enters S(t)
        additively.
      - correlation between deviation and t: a trend implicates the SHAPE, i.e.
        the kernel family, which is a failure of the theory rather than the
        estimate.
    """
    t_grid = np.asarray(pred["t_grid"], float)
    S_lo = np.asarray(pred["S_pred_lo"], float)
    S_hi = np.asarray(pred["S_pred_hi"], float)
    S_mid = np.asarray(pred["S_pred"], float)

    t_obs = np.asarray(t_obs, float)
    S_obs = np.asarray(S_obs, float)
    if len(t_obs) == 0:
        raise ValueError("no observations to score")

    # Interpolate the registered bracket onto the observed times. The bracket is
    # monotone in t and densely gridded, so linear interpolation is safe; points
    # outside the registered window are excluded and counted.
    in_win = (t_obs >= t_grid[0] - 1e-9) & (t_obs <= t_grid[-1] + 1e-9)
    n_out = int((~in_win).sum())
    to, So = t_obs[in_win], S_obs[in_win]

    lo_i = np.interp(to, t_grid, S_lo)
    hi_i = np.interp(to, t_grid, S_hi)
    mid_i = np.interp(to, t_grid, S_mid)

    inside = (So >= lo_i - 1e-12) & (So <= hi_i + 1e-12)
    dev = So - mid_i
    below = So < lo_i
    above = So > hi_i
    excess = np.where(below, lo_i - So, np.where(above, So - hi_i, 0.0))

    if len(to) > 2 and np.std(to) > 0 and np.std(dev) > 0:
        trend_r = float(np.corrcoef(to, dev)[0, 1])
    else:
        trend_r = float("nan")

    return {
        "n_obs_scored": int(len(to)),
        "n_obs_outside_window": n_out,
        "coverage": float(inside.mean()),
        "n_below": int(below.sum()),
        "n_above": int(above.sum()),
        "max_excess": float(excess.max()) if len(excess) else 0.0,
        "mean_excess": float(excess.mean()) if len(excess) else 0.0,
        "mean_signed_dev_from_mid": float(dev.mean()),
        "std_signed_dev": float(dev.std(ddof=1)) if len(dev) > 1 else 0.0,
        "dev_vs_t_correlation": trend_r,
        "diagnosis": _diagnose(float(inside.mean()), float(dev.mean()), trend_r),
        "bracket_mean_width": float(np.mean(hi_i - lo_i)),
    }


def _diagnose(coverage: float, mean_dev: float, trend_r: float) -> str:
    if coverage >= 0.95:
        return ("PASS: observation lies inside the registered bracket. The "
                "prediction is not falsified.")
    if not math.isfinite(trend_r):
        return "INCONCLUSIVE: too few points to diagnose."
    if abs(trend_r) < 0.3:
        return ("FAIL, offset-like: deviation is roughly constant in t, which "
                "implicates the ESTIMATOR (frozen mass / victim set), not the "
                "kernel family. Re-examine f-hat and the victim conditioning "
                "before rejecting the theory.")
    return ("FAIL, shape-like: deviation trends with t, which implicates the "
            "KERNEL FAMILY and therefore the Poisson access assumption (A1). "
            "This is a falsification of the theory, not of the estimator.")


# ---------------------------------------------------------------- #
# Amplification (Theorem 5) and the homogeneous null (Theorem 6)      #
# ---------------------------------------------------------------- #

def amplification_from_rates(lam: np.ndarray, theta0_quantile: float,
                             delta: float) -> Dict[str, float]:
    """Closed-form amplification A(Delta) = w(F^{-1}(F(th0)+D)) / w(th0) with
    w(lambda) = lambda, evaluated empirically from a rate sample.

    Returns nan when F(th0)+Delta reaches the top of the empirical support,
    which is the regime the feasibility bound excludes and where the empirical
    quantile is bounded by sample size rather than by the distribution.
    """
    lam = np.sort(np.asarray(lam, float))
    n = len(lam)
    q0 = float(theta0_quantile)
    q1 = q0 + float(delta)
    if q1 >= 1.0 - 1.0 / n:
        return {"theta0": float(np.quantile(lam, q0)), "theta1": float("nan"),
                "amplification": float("nan"),
                "note": "F(theta0)+Delta exceeds resolvable empirical support"}
    th0 = float(np.quantile(lam, q0))
    th1 = float(np.quantile(lam, q1))
    return {"theta0": th0, "theta1": th1,
            "amplification": (th1 / th0) if th0 > 0 else float("nan"),
            "note": ""}


def cold_flux(lam: np.ndarray, theta: float) -> float:
    """G(theta) = INT_0^theta lambda f(lambda) dlambda, empirically."""
    lam = np.asarray(lam, float)
    return float(lam[lam <= theta].sum() / len(lam))


def frontier_from_rate(lam: np.ndarray, rho: float) -> float:
    """theta* = G^{-1}(rho), solved on the empirical distribution by bisection.

    Returns nan if rho exceeds the total flux, which is Theorem 9's infeasible
    regime and must not be silently clamped.
    """
    lam = np.asarray(lam, float)
    total = float(lam.sum() / len(lam))
    if rho > total:
        return float("nan")
    lo, hi = 0.0, float(lam.max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cold_flux(lam, mid) < rho:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
