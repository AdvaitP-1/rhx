"""
telemetry.py -- background sampler that records EVERY available metric.

Design constraints
------------------
  - One row per sample, written as newline-delimited JSON so a crashed run still
    yields analyzable data. CSV is generated afterwards for convenience but JSONL
    is the source of truth, because the available field set is kernel-dependent
    and a fixed CSV header would silently drop fields.
  - Sampling runs in a thread with a monotonic schedule. Drift is measured and
    recorded per sample, so a reviewer can see whether the sampler kept up.
  - Cumulative counters (memory.stat pg*, PSI totals, vmstat) are recorded RAW.
    Differencing happens in analysis, never in collection, so that a dropped
    sample cannot corrupt an accumulated total.
  - The sampler records an explicit event log interleaved with samples, so that
    reclaim actions and snapshots are timestamped on the same monotonic clock as
    the metrics.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cgroupv2 import Cgroup, system_sample


class Telemetry:
    def __init__(self,
                 cgroup: Optional[Cgroup],
                 out_dir: Path,
                 interval_s: float = 0.25,
                 extra_sampler: Optional[Callable[[], Dict[str, Any]]] = None):
        self.cgroup = cgroup
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.interval_s = float(interval_s)
        self.extra_sampler = extra_sampler

        self.samples_path = self.out_dir / "samples.jsonl"
        self.events_path = self.out_dir / "events.jsonl"

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._n = 0
        self._t0 = None
        self._max_drift = 0.0

    # ---------------- control ----------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry already running")
        # truncate
        self.samples_path.write_text("")
        if not self.events_path.exists():
            self.events_path.write_text("")
        self._stop.clear()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 10.0) -> Dict[str, Any]:
        if self._thread is None:
            return {"n_samples": 0}
        self._stop.set()
        self._thread.join(timeout=timeout_s)
        self._thread = None
        return {"n_samples": self._n, "max_drift_s": self._max_drift}

    # ---------------- event log ----------------

    def event(self, kind: str, **fields: Any) -> Dict[str, Any]:
        """Record a timestamped event on the SAME monotonic clock as samples."""
        rec = {"t_mono": time.monotonic(), "t_unix": time.time(),
               "kind": kind, **fields}
        with self._lock:
            with open(self.events_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        return rec

    # ---------------- sampling ----------------

    def sample_once(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"seq": self._n}
        if self.cgroup is not None:
            try:
                row.update(self.cgroup.sample())
            except Exception as e:          # never let collection kill the run
                row["cgroup_sample_error"] = f"{type(e).__name__}: {e}"
                row["t_mono"] = time.monotonic()
        else:
            row["t_mono"] = time.monotonic()
        try:
            sysrow = system_sample()
            sysrow.pop("t_mono", None)
            row.update(sysrow)
        except Exception as e:
            row["system_sample_error"] = f"{type(e).__name__}: {e}"
        if self.extra_sampler is not None:
            try:
                extra = self.extra_sampler()
                if extra:
                    row.update(extra)
            except Exception as e:
                row["extra_sample_error"] = f"{type(e).__name__}: {e}"
        return row

    def _loop(self) -> None:
        next_t = time.monotonic()
        with open(self.samples_path, "a") as f:
            while not self._stop.is_set():
                now = time.monotonic()
                drift = now - next_t
                if drift > self._max_drift:
                    self._max_drift = drift
                row = self.sample_once()
                row["sched_drift_s"] = drift
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                self._n += 1
                next_t += self.interval_s
                sleep = next_t - time.monotonic()
                if sleep < 0:
                    # fell behind: resync rather than accumulate debt, and record
                    # that we did so via drift on the next row
                    next_t = time.monotonic()
                    sleep = 0.0
                self._stop.wait(sleep)


# ---------------- post-processing ----------------

def load_samples(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def to_csv(jsonl_path: Path, csv_path: Path) -> Dict[str, Any]:
    """Flatten JSONL to CSV using the UNION of all keys ever seen, so no field
    is silently dropped when a kernel exports a counter only intermittently."""
    import csv as _csv
    rows = load_samples(jsonl_path)
    if not rows:
        Path(csv_path).write_text("")
        return {"n_rows": 0, "n_cols": 0}
    keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    keys.sort(key=lambda k: (k not in ("seq", "t_mono", "t_unix"), k))
    with open(csv_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return {"n_rows": len(rows), "n_cols": len(keys), "columns": keys}


def diff_counters(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
    """Difference cumulative counters between the first and last sample.

    Returns per-key totals and a monotonicity check: a counter that decreases
    indicates either a cgroup being recreated or a counter reset, and must be
    surfaced rather than silently producing a negative delta.
    """
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [(r.get("t_mono"), r.get(k)) for r in rows if r.get(k) is not None]
        if len(vals) < 2:
            out[k] = {"delta": None, "n": len(vals), "monotone": None}
            continue
        series = [v for _, v in vals]
        monotone = all(b >= a for a, b in zip(series, series[1:]))
        out[k] = {
            "delta": series[-1] - series[0],
            "first": series[0],
            "last": series[-1],
            "n": len(series),
            "monotone": monotone,
            "dt_s": vals[-1][0] - vals[0][0] if vals[0][0] and vals[-1][0] else None,
        }
    return out
