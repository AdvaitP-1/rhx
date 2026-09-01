"""
prereg.py -- cryptographic pre-registration of predictions.

The central experiment (Gate 1) is a prediction test: estimate f(lambda), compute
the predicted recovery curve S(t) over a declared operating window, COMMIT to it,
and only then perform reclaim and measure. Without a tamper-evident commitment,
"we predicted this" is unverifiable and a reviewer is entitled to assume the
prediction was fitted after the fact.

Mechanism
---------
  1. The prediction is serialized to canonical JSON (sorted keys, fixed float
     repr) so the same prediction always produces the same bytes.
  2. SHA-256 of those bytes is appended to an append-only ledger together with a
     monotonic and wall-clock timestamp and the environment digest.
  3. The ledger is hash-chained: each entry includes the digest of the previous
     entry, so entries cannot be reordered, removed or back-dated without
     invalidating every subsequent entry.
  4. Verification recomputes the chain and re-hashes the prediction file.

This does not prevent a determined author from cheating; it makes accidental
post-hoc fitting impossible and deliberate fitting detectable by anyone who has
seen the ledger head at an earlier time. Publishing the ledger head (e.g. in a
git commit or a public post) timestamps it externally.

The observation window
----------------------
Per the experiment design, the registered quantity is S(t) over a DECLARED
window [0, T], with lambda_min = 1/T fixed in advance and everything below
lambda_min entering as a single "frozen fraction" scalar. Registering the
asymptotic exponent instead would require resolving the density near zero, which
demands observation windows over which the static-rate assumption fails.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LEDGER_NAME = "prereg_ledger.jsonl"
GENESIS = "0" * 64


def canonical_json(obj: Any) -> str:
    """Deterministic serialization. repr-stable floats via sort_keys and
    separators; NaN/Infinity are rejected because they are not valid JSON and
    would make the digest ambiguous."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str
    )


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ledger:
    """Append-only hash-chained ledger."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / LEDGER_NAME

    def entries(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def head(self) -> str:
        es = self.entries()
        return es[-1]["entry_digest"] if es else GENESIS

    def append(self, kind: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prev = self.head()
        body = {
            "kind": kind,
            "prev": prev,
            "t_unix": time.time(),
            "t_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "payload": payload,
        }
        body["entry_digest"] = sha256_str(canonical_json(body))
        with open(self.path, "a") as f:
            f.write(json.dumps(body, sort_keys=True) + "\n")
        return body

    def verify(self) -> Dict[str, Any]:
        """Recompute the chain. Returns a verdict with the index of the first
        broken link, if any."""
        prev = GENESIS
        for i, e in enumerate(self.entries()):
            if e.get("prev") != prev:
                return {"ok": False, "broken_at": i, "reason": "prev mismatch"}
            claimed = e.get("entry_digest")
            body = {k: v for k, v in e.items() if k != "entry_digest"}
            if sha256_str(canonical_json(body)) != claimed:
                return {"ok": False, "broken_at": i, "reason": "digest mismatch"}
            prev = claimed
        return {"ok": True, "entries": len(self.entries()), "head": prev}


def register_prediction(
    root: Path,
    label: str,
    prediction: Dict[str, Any],
    env_sha256: Optional[str],
    notes: str = "",
) -> Dict[str, Any]:
    """Commit to a prediction BEFORE the corresponding measurement.

    `prediction` must fully determine the predicted curve, including:
      - window_T_s          declared operating window
      - lambda_min          == 1/window_T_s, fixed in advance
      - frozen_fraction     mass below lambda_min, a single scalar
      - t_grid              times at which S is predicted
      - S_pred              predicted survival values
      - S_pred_lo/S_pred_hi prediction interval
      - estimator           how f was estimated, with its parameters
    """
    required = ["window_T_s", "lambda_min", "frozen_fraction",
                "t_grid", "S_pred", "S_pred_lo", "S_pred_hi", "estimator"]
    missing = [k for k in required if k not in prediction]
    if missing:
        raise ValueError(f"prediction is missing required fields: {missing}")

    lm = prediction["lambda_min"]
    T = prediction["window_T_s"]
    if abs(lm * T - 1.0) > 1e-9:
        raise ValueError(
            f"lambda_min must equal 1/window_T_s by construction "
            f"(got lambda_min={lm}, T={T}, product={lm*T})"
        )
    n = len(prediction["t_grid"])
    for k in ("S_pred", "S_pred_lo", "S_pred_hi"):
        if len(prediction[k]) != n:
            raise ValueError(f"{k} length {len(prediction[k])} != len(t_grid) {n}")
    if any(t > T + 1e-12 for t in prediction["t_grid"]):
        raise ValueError("t_grid extends beyond the declared window T")

    ledger = Ledger(root)
    pred_dir = ledger.root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{label}.json"
    if pred_path.exists():
        raise FileExistsError(
            f"prediction {label} already registered at {pred_path}; "
            "re-registering would defeat the commitment"
        )
    blob = canonical_json(prediction)
    pred_path.write_text(blob)
    digest = sha256_str(blob)

    entry = ledger.append("prediction", {
        "label": label,
        "prediction_sha256": digest,
        "prediction_path": str(pred_path.relative_to(ledger.root)),
        "environment_sha256": env_sha256,
        "window_T_s": T,
        "lambda_min": lm,
        "frozen_fraction": prediction["frozen_fraction"],
        "n_grid": n,
        "notes": notes,
    })
    return {"digest": digest, "path": str(pred_path), "entry": entry,
            "ledger_head": ledger.head()}


def register_observation(
    root: Path,
    label: str,
    observation_path: Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Commit to a measurement AFTER it is written. Records the file digest so
    the observation cannot be edited after the comparison is published."""
    ledger = Ledger(root)
    digest = sha256_file(Path(observation_path))
    entry = ledger.append("observation", {
        "label": label,
        "observation_sha256": digest,
        "observation_path": str(observation_path),
        **(extra or {}),
    })
    return {"digest": digest, "entry": entry, "ledger_head": ledger.head()}


def load_prediction(root: Path, label: str) -> Dict[str, Any]:
    """Load a registered prediction and verify its digest still matches the
    ledger. Raises if the file was modified after registration."""
    ledger = Ledger(root)
    pred_path = ledger.root / "predictions" / f"{label}.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"no registered prediction '{label}'")
    actual = sha256_str(pred_path.read_text())
    recorded = None
    for e in ledger.entries():
        if e["kind"] == "prediction" and e["payload"]["label"] == label:
            recorded = e["payload"]["prediction_sha256"]
    if recorded is None:
        raise ValueError(f"prediction '{label}' has no ledger entry")
    if actual != recorded:
        raise ValueError(
            f"prediction '{label}' was MODIFIED after registration "
            f"(ledger {recorded[:16]}..., file {actual[:16]}...)"
        )
    return json.loads(pred_path.read_text())
