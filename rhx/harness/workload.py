"""
workload.py -- launch and control the rategen synthetic workload.

rategen exposes a control FIFO. Residency is read via mincore(2) inside the
workload process, which is the only place it can be read, and is EXACT: it
reports whether each page of the mapping is present in the page cache (for
file-backed MAP_SHARED) or resident (for anonymous). Using mincore rather than
inferring residency from refault counters removes a whole class of ambiguity
from the recovery measurement.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class WorkloadError(RuntimeError):
    pass


@dataclass
class RateSpec:
    """How per-page rates are assigned. Recorded verbatim in the manifest."""
    kind: str                       # gamma | uniform | lognormal | discrete
    p1: float = 0.0
    p2: float = 0.0
    discrete: Optional[List[float]] = None

    def argv(self) -> List[str]:
        if self.kind == "gamma":
            return ["--gamma", str(self.p1), str(self.p2)]
        if self.kind == "uniform":
            return ["--uniform", str(self.p1)]
        if self.kind == "lognormal":
            return ["--lognormal", str(self.p1), str(self.p2)]
        if self.kind == "discrete":
            if not self.discrete:
                raise ValueError("discrete rate spec needs a rate list")
            return ["--discrete", ",".join(str(x) for x in self.discrete)]
        raise ValueError(f"unknown rate spec kind: {self.kind}")

    def as_dict(self) -> Dict[str, object]:
        return {"kind": self.kind, "p1": self.p1, "p2": self.p2,
                "discrete": self.discrete}


class Workload:
    def __init__(self,
                 binary: Path,
                 workdir: Path,
                 n_pages: int,
                 seed: int,
                 rate_spec: RateSpec,
                 backing: str = "FILE",
                 prefault: bool = True):
        self.binary = Path(binary)
        self.workdir = Path(workdir)
        self.n_pages = int(n_pages)
        self.seed = int(seed)
        self.rate_spec = rate_spec
        self.backing = backing
        self.prefault = prefault

        self.workdir.mkdir(parents=True, exist_ok=True)
        self.back_path = self.workdir / "backing.bin"
        self.truth_path = self.workdir / "truth.tsv"
        self.ctl_path = self.workdir / "ctl.fifo"
        self.err_path = self.workdir / "rategen.stderr"
        self.proc: Optional[subprocess.Popen] = None
        self._snap_i = 0

        if not self.binary.exists():
            raise WorkloadError(f"rategen binary not found at {self.binary}; run make")

    # ---------------- lifecycle ----------------

    def start(self, duration_s: Optional[float] = None, report_s: float = 0.0,
              cgroup_procs: Optional[Path] = None) -> int:
        argv = [str(self.binary),
                "--backing", self.backing,
                "--pages", str(self.n_pages),
                "--seed", str(self.seed),
                "--truth", str(self.truth_path),
                "--ctl", str(self.ctl_path)]
        if self.backing == "FILE":
            argv += ["--path", str(self.back_path)]
        argv += self.rate_spec.argv()
        if duration_s is not None:
            argv += ["--duration", str(duration_s)]
        if report_s:
            argv += ["--report", str(report_s)]
        if self.prefault:
            argv += ["--prefault"]

        launch_argv = argv
        if cgroup_procs is not None:
            cgroup_procs = Path(cgroup_procs)
            if not cgroup_procs.exists():
                raise WorkloadError(f"cgroup.procs does not exist: {cgroup_procs}")
            # Join the target cgroup in the wrapper process, then exec rategen
            # with the same PID.  No rategen initialization can occur before
            # the join, so FILE materialization and both FILE/ANON prefaults are
            # charged to the experimental cgroup from their first page.
            launch_argv = [
                "/bin/sh", "-c",
                'printf "0\\n" > "$1" || exit 126; shift; exec "$@"',
                "rhx-cgroup-launch", str(cgroup_procs), *argv,
            ]

        errf = open(self.err_path, "w")
        self.proc = subprocess.Popen(
            launch_argv, stdout=subprocess.DEVNULL, stderr=errf)

        # Wait for the FIFO to appear and the process to announce readiness.
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise WorkloadError(
                    f"rategen exited early (rc={self.proc.returncode}); "
                    f"see {self.err_path}"
                )
            if self.ctl_path.exists() and self.truth_path.exists():
                txt = self.err_path.read_text()
                if "ready" in txt:
                    return self.proc.pid
            time.sleep(0.05)
        raise WorkloadError(f"rategen did not become ready; see {self.err_path}")

    def stop(self, timeout_s: float = 10.0) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.send("QUIT")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5.0)
        self.proc = None

    @property
    def pid(self) -> int:
        if self.proc is None:
            raise WorkloadError("workload is not running")
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---------------- control ----------------

    def send(self, cmd: str) -> None:
        if not self.ctl_path.exists():
            raise WorkloadError("control FIFO missing")
        # Opening a FIFO for write blocks until a reader exists; rategen holds it
        # open O_RDWR so this returns immediately.
        fd = os.open(str(self.ctl_path), os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, (cmd + "\n").encode())
        finally:
            os.close(fd)

    def snapshot_residency(self, tag: str = "", wait_s: float = 5.0) -> Path:
        """Request a mincore residency bitmap. Returns the path once written."""
        self._snap_i += 1
        name = f"resid_{self._snap_i:05d}{('_' + tag) if tag else ''}.txt"
        out = self.workdir / name
        if out.exists():
            out.unlink()
        self.send(f"SNAPSHOT {out}")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if out.exists():
                # ensure the file is complete: header + bitmap line
                try:
                    txt = out.read_text()
                    if txt.count("\n") >= 2:
                        return out
                except OSError:
                    pass
            time.sleep(0.01)
        raise WorkloadError(f"residency snapshot {out} not produced within {wait_s}s")

    def dump_counts(self, tag: str = "", wait_s: float = 15.0) -> Path:
        name = f"counts{('_' + tag) if tag else ''}.tsv"
        out = self.workdir / name
        if out.exists():
            out.unlink()
        self.send(f"COUNTS {out}")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if out.exists() and out.stat().st_size > 0:
                txt = out.read_text()
                if txt.rstrip().endswith(tuple("0123456789")):
                    return out
            time.sleep(0.02)
        raise WorkloadError(f"counts dump {out} not produced within {wait_s}s")

    def reset_counts(self) -> None:
        self.send("RESETCOUNTS")

    def pause(self) -> None:
        """B4: quiesce page access so the victim set measured across a reclaim
        is attributable to reclaim alone and not to ordinary workload access."""
        self.send("PAUSE")

    def resume(self) -> None:
        """Resume access. Pending events are shifted by the pause duration
        inside rategen, so resuming does not produce a compensating burst."""
        self.send("RESUME")

    def snapshot_quiesced(self, tag: str = "") -> Path:
        """Snapshot with access paused for the duration of the mincore call."""
        self.pause()
        try:
            return self.snapshot_residency(tag)
        finally:
            self.resume()

    # ---------------- parsing ----------------

    def read_truth(self) -> np.ndarray:
        return parse_truth(self.truth_path)


def parse_truth(path: Path) -> np.ndarray:
    lam: List[float] = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("page"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                lam.append(float(parts[1]))
    return np.asarray(lam, dtype=float)


def parse_counts(path: Path) -> Dict[str, object]:
    elapsed = None
    lam_true: List[float] = []
    counts: List[int] = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                for tok in line.split():
                    if tok.startswith("elapsed="):
                        elapsed = float(tok.split("=", 1)[1])
                continue
            if line.startswith("page"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                lam_true.append(float(parts[1]))
                counts.append(int(parts[2]))
    if elapsed is None:
        raise WorkloadError(f"counts file {path} has no elapsed= header")
    return {"elapsed_s": elapsed,
            "lambda_true": np.asarray(lam_true, dtype=float),
            "counts": np.asarray(counts, dtype=np.int64)}


def parse_residency(path: Path) -> Dict[str, object]:
    """Returns t (seconds since workload start) and a boolean resident mask."""
    with open(path) as f:
        header = f.readline()
        bits = f.readline().strip()
    t = None
    npages = None
    for tok in header.split():
        if tok.startswith("t="):
            t = float(tok.split("=", 1)[1])
        elif tok.startswith("npages="):
            npages = int(tok.split("=", 1)[1])
    if t is None or npages is None:
        raise WorkloadError(f"malformed residency header in {path}: {header!r}")
    if len(bits) != npages:
        raise WorkloadError(
            f"residency bitmap length {len(bits)} != npages {npages} in {path}"
        )
    resident = np.frombuffer(bits.encode(), dtype=np.uint8) == ord("1")
    return {"t": t, "resident": resident, "n_pages": npages}
