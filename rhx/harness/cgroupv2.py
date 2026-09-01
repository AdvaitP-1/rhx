"""
cgroupv2.py -- cgroup v2 interface for the reclaim-hysteresis experiment.

Every read is defensive: a missing file yields None rather than an exception,
because kernel builds differ in which fields they expose and the experiment must
record what is actually available rather than assume a field set.

Two reclaim modes are supported and are NOT interchangeable:

  Mode A (proactive):  write to memory.reclaim.
      The kernel documentation states that proactive reclaim invoked this way is
      not treated as memory pressure on the cgroup. It therefore exercises the
      same victim-selection machinery (LRU/MGLRU ordering) WITHOUT generating
      the pressure signal. It is valid for testing page-dynamics claims
      (recovery kernel, composition shift, amplification) and INVALID for
      testing damage claims that depend on pressure accounting.

  Mode B (natural):    lower memory.high and let the workload's own
      allocation drive reclaim under genuine contention. This produces real
      pressure and real PSI, and is the mode required for any SLO or distortion
      claim.

The distinction is recorded in every result file so that no analysis can silently
mix them.
"""

from __future__ import annotations

import errno
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

CGROUP_ROOT = Path("/sys/fs/cgroup")

# memory.stat keys we attempt to read. Missing keys are recorded as None so that
# the analysis can distinguish "absent on this kernel" from "zero".
MEMORY_STAT_KEYS = [
    "anon", "file", "kernel", "kernel_stack", "pagetables", "sec_pagetables",
    "percpu", "sock", "vmalloc", "shmem", "zswap", "zswapped",
    "file_mapped", "file_dirty", "file_writeback", "swapcached",
    "anon_thp", "file_thp", "shmem_thp",
    "inactive_anon", "active_anon", "inactive_file", "active_file",
    "unevictable", "slab_reclaimable", "slab_unreclaimable", "slab",
    "workingset_refault_anon", "workingset_refault_file",
    "workingset_activate_anon", "workingset_activate_file",
    "workingset_restore_anon", "workingset_restore_file",
    "workingset_nodereclaim",
    "pgscan", "pgsteal",
    "pgscan_kswapd", "pgscan_direct", "pgscan_khugepaged",
    "pgsteal_kswapd", "pgsteal_direct", "pgsteal_khugepaged",
    "pgfault", "pgmajfault", "pgrefill",
    "pgactivate", "pgdeactivate", "pglazyfree", "pglazyfreed",
    "zswpin", "zswpout", "zswpwb",
    "thp_fault_alloc", "thp_collapse_alloc", "thp_swpout", "thp_swpout_fallback",
]

MEMORY_EVENT_KEYS = ["low", "high", "max", "oom", "oom_kill", "oom_group_kill"]


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _read_int(path: Path) -> Optional[int]:
    txt = _read_text(path)
    if txt is None:
        return None
    txt = txt.strip()
    if txt == "max":
        return -1  # sentinel: unlimited
    try:
        return int(txt)
    except ValueError:
        return None


def cgroup_v2_available() -> bool:
    """True only if cgroup v2 unified hierarchy is mounted with a memory
    controller. Checked explicitly rather than assumed."""
    if not CGROUP_ROOT.exists():
        return False
    if not (CGROUP_ROOT / "cgroup.controllers").exists():
        return False
    controllers = _read_text(CGROUP_ROOT / "cgroup.controllers") or ""
    return "memory" in controllers.split()


def parse_psi(text: Optional[str]) -> Dict[str, Optional[float]]:
    """Parse a PSI file.

    Format (per kernel Documentation/accounting/psi.rst):
        some avg10=0.00 avg60=0.00 avg300=0.00 total=0
        full avg10=0.00 avg60=0.00 avg300=0.00 total=0

    'total' is in MICROSECONDS of stall time and is the field to use for
    differencing; the avg fields are decayed percentages and are not additive.
    """
    out: Dict[str, Optional[float]] = {
        f"{kind}_{f}": None
        for kind in ("some", "full")
        for f in ("avg10", "avg60", "avg300", "total")
    }
    if not text:
        return out
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        kind = parts[0]
        if kind not in ("some", "full"):
            continue
        for kv in parts[1:]:
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            key = f"{kind}_{k}"
            if key in out:
                try:
                    out[key] = float(v)
                except ValueError:
                    out[key] = None
    return out


def parse_keyed(text: Optional[str], keys: List[str], prefix: str) -> Dict[str, Optional[int]]:
    """Parse a flat 'key value' file, returning only the requested keys."""
    out: Dict[str, Optional[int]] = {f"{prefix}{k}": None for k in keys}
    if not text:
        return out
    seen = {}
    for line in text.strip().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                seen[parts[0]] = int(parts[1])
            except ValueError:
                pass
    for k in keys:
        if k in seen:
            out[f"{prefix}{k}"] = seen[k]
    return out


@dataclass
class ReclaimResult:
    """Outcome of a reclaim attempt. Records what was REQUESTED and what the
    kernel actually accounted for, because memory.reclaim can partially fail."""
    mode: str                      # "proactive" | "natural"
    requested_bytes: int
    memory_current_before: Optional[int]
    memory_current_after: Optional[int]
    actual_delta_bytes: Optional[int]
    wrote_ok: bool
    error: Optional[str]
    t_start_mono: float
    t_end_mono: float
    # B3: OOM safety. A nonzero delta here means the trial is INVALID.
    oom_kill_delta: int = 0
    oom_delta: int = 0
    workload_survived: Optional[bool] = None

    @property
    def duration_s(self) -> float:
        return self.t_end_mono - self.t_start_mono


class Cgroup:
    """A cgroup v2 memory cgroup under CGROUP_ROOT."""

    def __init__(self, name: str, parent: Path = CGROUP_ROOT):
        self.name = name
        self.path = parent / name
        self._created = False

    # ---------- lifecycle ----------

    def create(self) -> None:
        if not cgroup_v2_available():
            raise RuntimeError(
                "cgroup v2 with memory controller is not available. "
                "This experiment requires a Linux host with the unified hierarchy."
            )
        # Ensure the memory controller is delegated to children of the parent.
        subtree = self.path.parent / "cgroup.subtree_control"
        ctrls = _read_text(self.path.parent / "cgroup.controllers") or ""
        if "memory" in ctrls.split():
            try:
                subtree.write_text("+memory\n")
            except OSError as e:
                # EBUSY/EINVAL here usually means it is already enabled.
                if e.errno not in (errno.EBUSY, errno.EINVAL):
                    raise
        self.path.mkdir(parents=True, exist_ok=True)
        self._created = True

    def destroy(self) -> None:
        if self.path.exists():
            # Move any surviving processes to the root before rmdir, else EBUSY.
            procs = self.procs()
            for pid in procs:
                try:
                    (CGROUP_ROOT / "cgroup.procs").write_text(f"{pid}\n")
                except OSError:
                    pass
            try:
                self.path.rmdir()
            except OSError:
                pass

    def add_pid(self, pid: int) -> None:
        (self.path / "cgroup.procs").write_text(f"{pid}\n")

    def procs(self) -> List[int]:
        txt = _read_text(self.path / "cgroup.procs") or ""
        out = []
        for line in txt.split():
            try:
                out.append(int(line))
            except ValueError:
                pass
        return out

    # ---------- limits ----------

    def set_limit(self, knob: str, value) -> None:
        """knob in {memory.max, memory.high, memory.low, memory.min}.
        value: int bytes, or the string 'max'."""
        assert knob in ("memory.max", "memory.high", "memory.low", "memory.min")
        s = "max" if value in ("max", None, -1) else str(int(value))
        (self.path / knob).write_text(s + "\n")

    def get_limit(self, knob: str) -> Optional[int]:
        return _read_int(self.path / knob)

    # ---------- reclaim ----------

    def reclaim_proactive(self, nbytes: int, swappiness: Optional[int] = None) -> ReclaimResult:
        """Mode A. Write to memory.reclaim.

        NOTE: per kernel docs this is NOT accounted as pressure on the cgroup.
        Use only for page-dynamics measurements.

        The kernel returns EAGAIN if it could not reclaim the full amount; this
        is captured rather than raised, because partial reclaim is informative.
        """
        before = self.memory_current()
        t0 = time.monotonic()
        arg = str(int(nbytes))
        if swappiness is not None:
            # Supported on newer kernels only; failure is recorded, not fatal.
            arg += f" swappiness={int(swappiness)}"
        ok, err = True, None
        try:
            (self.path / "memory.reclaim").write_text(arg + "\n")
        except OSError as e:
            ok = False
            err = f"{type(e).__name__}: errno={e.errno} {e.strerror}"
            if swappiness is not None:
                # retry without the extra token, which older kernels reject
                try:
                    (self.path / "memory.reclaim").write_text(str(int(nbytes)) + "\n")
                    ok, err = True, "retried_without_swappiness"
                except OSError as e2:
                    err = f"{err}; retry: errno={e2.errno} {e2.strerror}"
        t1 = time.monotonic()
        after = self.memory_current()
        delta = (before - after) if (before is not None and after is not None) else None
        return ReclaimResult("proactive", int(nbytes), before, after, delta, ok, err, t0, t1)

    def _oom_counters(self) -> Dict[str, int]:
        ev = parse_keyed(_read_text(self.path / "memory.events"),
                         MEMORY_EVENT_KEYS, "")
        return {"oom": ev.get("oom") or 0, "oom_kill": ev.get("oom_kill") or 0}

    def reclaim_natural(self, target_bytes: int, settle_s: float = 2.0,
                        watch_pid: Optional[int] = None) -> ReclaimResult:
        """Mode B. Apply genuine memory pressure so reclaim is accounted as
        pressure and drives PSI. Required for any distortion or SLO claim.

        B3 SAFETY: memory.max is a HARD limit and must not be used here. This
        method uses memory.high, which THROTTLES the allocator and drives
        reclaim without imposing an OOM-enforced hard limit.

        OOM counters are read before and after. A nonzero oom_kill_delta marks
        the trial INVALID; the caller must discard it rather than analyze it.
        """
        knob = "memory.high"
        before = self.memory_current()
        prev = self.get_limit(knob)
        oom_before = self._oom_counters()
        t0 = time.monotonic()
        ok, err = True, None
        try:
            self.set_limit(knob, target_bytes)
            time.sleep(settle_s)
        except OSError as e:
            ok = False
            err = f"{type(e).__name__}: errno={e.errno} {e.strerror}"
        t1 = time.monotonic()
        after = self.memory_current()
        oom_after = self._oom_counters()
        delta = (before - after) if (before is not None and after is not None) else None

        survived = None
        if watch_pid is not None:
            try:
                os.kill(watch_pid, 0)
                survived = True
            except ProcessLookupError:
                survived = False
            except PermissionError:
                survived = True

        res = ReclaimResult(
            "natural", int(target_bytes), before, after, delta, ok, err, t0, t1,
            oom_kill_delta=oom_after["oom_kill"] - oom_before["oom_kill"],
            oom_delta=oom_after["oom"] - oom_before["oom"],
            workload_survived=survived,
        )
        try:
            self.set_limit(knob, "max" if prev in (None, -1) else prev)
        except OSError:
            pass
        return res

    # ---------- observation ----------

    def memory_current(self) -> Optional[int]:
        return _read_int(self.path / "memory.current")

    def sample(self) -> Dict[str, Optional[float]]:
        """One complete metric sample for this cgroup.

        Returns a flat dict. Every documented memory.stat field we know about is
        attempted; absent fields are None so that analysis can distinguish
        'kernel does not export this' from 'value is zero'.
        """
        s: Dict[str, Optional[float]] = {}
        s["t_mono"] = time.monotonic()
        s["t_unix"] = time.time()

        s["memory_current"] = _read_int(self.path / "memory.current")
        s["memory_peak"] = _read_int(self.path / "memory.peak")
        s["memory_max"] = _read_int(self.path / "memory.max")
        s["memory_high"] = _read_int(self.path / "memory.high")
        s["memory_low"] = _read_int(self.path / "memory.low")
        s["memory_min"] = _read_int(self.path / "memory.min")
        s["memory_swap_current"] = _read_int(self.path / "memory.swap.current")
        s["memory_swap_max"] = _read_int(self.path / "memory.swap.max")
        s["memory_zswap_current"] = _read_int(self.path / "memory.zswap.current")

        s.update(parse_keyed(_read_text(self.path / "memory.stat"),
                             MEMORY_STAT_KEYS, "stat_"))
        s.update(parse_keyed(_read_text(self.path / "memory.events"),
                             MEMORY_EVENT_KEYS, "ev_"))
        s.update(parse_keyed(_read_text(self.path / "memory.events.local"),
                             MEMORY_EVENT_KEYS, "evlocal_"))
        s.update(parse_keyed(_read_text(self.path / "memory.swap.events"),
                             ["high", "max", "fail"], "swapev_"))

        psi = parse_psi(_read_text(self.path / "memory.pressure"))
        s.update({f"psi_mem_{k}": v for k, v in psi.items()})
        psi_io = parse_psi(_read_text(self.path / "io.pressure"))
        s.update({f"psi_io_{k}": v for k, v in psi_io.items()})
        psi_cpu = parse_psi(_read_text(self.path / "cpu.pressure"))
        s.update({f"psi_cpu_{k}": v for k, v in psi_cpu.items()})

        return s


def system_sample() -> Dict[str, Optional[float]]:
    """System-wide counters that contextualize per-cgroup numbers.

    /proc/vmstat is global and is the ground truth for whether reclaim happened
    anywhere on the machine, which matters when checking that an experiment's
    reclaim was actually confined to the target cgroup.
    """
    s: Dict[str, Optional[float]] = {"t_mono": time.monotonic()}

    vm = _read_text(Path("/proc/vmstat"))
    wanted = [
        "nr_free_pages", "nr_inactive_anon", "nr_active_anon",
        "nr_inactive_file", "nr_active_file", "nr_unevictable",
        "nr_file_pages", "nr_dirty", "nr_writeback", "nr_anon_pages",
        "nr_mapped", "nr_shmem", "nr_slab_reclaimable", "nr_slab_unreclaimable",
        "workingset_refault_anon", "workingset_refault_file",
        "workingset_activate_anon", "workingset_activate_file",
        "workingset_restore_anon", "workingset_restore_file",
        "workingset_nodereclaim",
        "pgpgin", "pgpgout", "pswpin", "pswpout",
        "pgfault", "pgmajfault",
        "pgscan_kswapd", "pgscan_direct", "pgsteal_kswapd", "pgsteal_direct",
        "pgrefill", "pgactivate", "pgdeactivate",
        "allocstall_dma", "allocstall_dma32", "allocstall_normal",
        "allocstall_movable",
        "compact_stall", "thp_fault_alloc", "thp_collapse_alloc",
        "zswpin", "zswpout",
    ]
    s.update(parse_keyed(vm, wanted, "vm_"))

    mi = _read_text(Path("/proc/meminfo"))
    if mi:
        for line in mi.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip().split()
            if not v:
                continue
            try:
                val = int(v[0])
            except ValueError:
                continue
            if v[-1].lower() == "kb":
                val *= 1024
            key = f"mi_{k.strip().replace('(', '_').replace(')', '')}"
            s[key] = val

    for name, path in (("sys_psi_mem", "/proc/pressure/memory"),
                       ("sys_psi_io", "/proc/pressure/io"),
                       ("sys_psi_cpu", "/proc/pressure/cpu")):
        psi = parse_psi(_read_text(Path(path)))
        s.update({f"{name}_{k}": v for k, v in psi.items()})

    la = _read_text(Path("/proc/loadavg"))
    if la:
        p = la.split()
        if len(p) >= 3:
            try:
                s["loadavg_1"] = float(p[0])
                s["loadavg_5"] = float(p[1])
                s["loadavg_15"] = float(p[2])
            except ValueError:
                pass
    return s
