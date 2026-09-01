"""
envcapture.py -- record every environment variable that can change the outcome.

A reclaim experiment is only reproducible if the memory-management configuration
is recorded. Reviewers will ask for kernel version, MGLRU state, swap topology,
THP policy, overcommit mode, NUMA layout, and CPU governor. Anything not
captured here cannot be defended later.

The capture is written once per run and its SHA-256 is included in the run
manifest, so an environment change between pre-registration and execution is
detectable rather than silent.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _run(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    if not shutil.which(cmd[0]):
        return None
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def capture() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "captured_at_unix": time.time(),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
    }

    # ---- platform / kernel ----
    env["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }
    env["kernel_version_proc"] = _read("/proc/version")
    env["kernel_cmdline"] = _read("/proc/cmdline")
    env["uname_a"] = _run(["uname", "-a"])

    # A macOS or non-Linux host cannot produce publishable reclaim data. This is
    # recorded explicitly so no result file can be mistaken for a valid run.
    env["is_linux"] = platform.system() == "Linux"

    # ---- virtualization: VM is valid for development, not for claims ----
    virt = _run(["systemd-detect-virt"])
    env["virtualization"] = {
        "systemd_detect_virt": virt,
        "is_virtualized": (virt is not None and virt not in ("none",)),
        "dmi_product": _read("/sys/class/dmi/id/product_name"),
        "dmi_vendor": _read("/sys/class/dmi/id/sys_vendor"),
        "hypervisor_flag": ("hypervisor" in (_read("/proc/cpuinfo") or "")),
    }

    # ---- cgroup v2 ----
    env["cgroup"] = {
        "unified_mounted": Path("/sys/fs/cgroup/cgroup.controllers").exists(),
        "root_controllers": _read("/sys/fs/cgroup/cgroup.controllers"),
        "root_subtree_control": _read("/sys/fs/cgroup/cgroup.subtree_control"),
        "self_cgroup": _read("/proc/self/cgroup"),
    }

    # ---- MGLRU: directly changes victim selection, so it is critical ----
    env["mglru"] = {
        "enabled_raw": _read("/sys/kernel/mm/lru_gen/enabled"),
        "min_ttl_ms": _read("/sys/kernel/mm/lru_gen/min_ttl_ms"),
        "debugfs_present": Path("/sys/kernel/debug/lru_gen").exists(),
    }

    # ---- THP: changes page granularity and therefore reclaim units ----
    env["thp"] = {
        "enabled": _read("/sys/kernel/mm/transparent_hugepage/enabled"),
        "defrag": _read("/sys/kernel/mm/transparent_hugepage/defrag"),
        "shmem_enabled": _read("/sys/kernel/mm/transparent_hugepage/shmem_enabled"),
        "khugepaged_defrag": _read("/sys/kernel/mm/transparent_hugepage/khugepaged/defrag"),
    }

    # ---- swap / zswap / zram: determines whether anon pages are reclaimable ----
    env["swap"] = {
        "proc_swaps": _read("/proc/swaps"),
        "swappiness": _read("/proc/sys/vm/swappiness"),
        "zswap_enabled": _read("/sys/module/zswap/parameters/enabled"),
        "zswap_compressor": _read("/sys/module/zswap/parameters/compressor"),
        "zswap_max_pool_percent": _read("/sys/module/zswap/parameters/max_pool_percent"),
        "zram_devices": sorted(p.name for p in Path("/sys/block").glob("zram*"))
                        if Path("/sys/block").exists() else [],
    }

    # ---- VM tunables that alter reclaim behaviour ----
    vm_keys = [
        "overcommit_memory", "overcommit_ratio", "overcommit_kbytes",
        "min_free_kbytes", "watermark_scale_factor", "watermark_boost_factor",
        "vfs_cache_pressure", "dirty_ratio", "dirty_background_ratio",
        "dirty_expire_centisecs", "dirty_writeback_centisecs",
        "page-cluster", "compaction_proactiveness", "extfrag_threshold",
        "zone_reclaim_mode", "panic_on_oom", "oom_kill_allocating_task",
        "max_map_count", "laptop_mode", "stat_interval",
    ]
    env["vm_sysctl"] = {k: _read(f"/proc/sys/vm/{k}") for k in vm_keys}

    # ---- NUMA ----
    node_dir = Path("/sys/devices/system/node")
    nodes = sorted(p.name for p in node_dir.glob("node*")) if node_dir.exists() else []
    env["numa"] = {
        "nodes": nodes,
        "num_nodes": len(nodes),
        "numactl_hardware": _run(["numactl", "--hardware"]),
        "demotion_enabled": _read("/sys/kernel/mm/numa/demotion_enabled"),
    }

    # ---- CPU: governor and turbo affect timing-sensitive measurements ----
    govs = []
    cpud = Path("/sys/devices/system/cpu")
    if cpud.exists():
        for p in sorted(cpud.glob("cpu[0-9]*/cpufreq/scaling_governor")):
            g = _read(str(p))
            if g:
                govs.append(g)
    env["cpu"] = {
        "count": os.cpu_count(),
        "governors": sorted(set(govs)),
        "governors_uniform": (len(set(govs)) <= 1),
        "intel_no_turbo": _read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
        "smt_active": _read("/sys/devices/system/cpu/smt/active"),
    }

    # ---- DAMON availability: the access-rate estimator depends on it ----
    env["damon"] = {
        "sysfs_present": Path("/sys/kernel/mm/damon/admin").exists(),
        "debugfs_present": Path("/sys/kernel/debug/damon").exists(),
        "damo_binary": shutil.which("damo"),
    }

    # ---- idle page tracking: the fallback estimator ----
    env["idle_page_tracking"] = {
        "bitmap_present": Path("/sys/kernel/mm/page_idle/bitmap").exists(),
    }

    # ---- filesystem holding the backing file ----
    env["mounts"] = _read("/proc/mounts")

    # ---- privileges ----
    env["privileges"] = {
        "euid": os.geteuid() if hasattr(os, "geteuid") else None,
        "is_root": (os.geteuid() == 0) if hasattr(os, "geteuid") else False,
    }

    return env


def validate_for_claims(env: Dict[str, Any]) -> Dict[str, Any]:
    """Decide whether this environment can support PUBLISHABLE results.

    Returns a verdict dict. The rule adopted for this program is:
        VM for development, bare metal for claims.
    A run on a virtualized or non-Linux host is permitted but is marked
    development-only in the manifest, so it cannot later be mistaken for
    evidence.
    """
    blockers: List[str] = []
    warnings: List[str] = []

    if not env.get("is_linux"):
        blockers.append(
            "Host is not Linux. cgroup v2, PSI, MGLRU and DAMON are unavailable; "
            "no reclaim measurement is possible."
        )
    if env.get("virtualization", {}).get("is_virtualized"):
        blockers.append(
            "Host is virtualized. The guest does not own host memory management, "
            "so page-cache and reclaim behaviour reflect a guest view. Valid for "
            "development, not for claims."
        )
    if not env.get("cgroup", {}).get("unified_mounted"):
        blockers.append("cgroup v2 unified hierarchy is not mounted.")
    if not env.get("privileges", {}).get("is_root"):
        blockers.append(
            "Not running as root. Creating cgroups, writing memory.reclaim and "
            "reading idle-page bitmaps all require privilege."
        )

    mg = env.get("mglru", {}).get("enabled_raw")
    if mg is None:
        warnings.append("MGLRU state unreadable; victim-selection policy is unrecorded.")
    thp = (env.get("thp", {}).get("enabled") or "")
    if "[always]" in thp:
        warnings.append(
            "THP is 'always'. Huge pages change the reclaim unit and blur "
            "per-4KiB-page residency; consider 'madvise' or 'never' for "
            "characterization runs."
        )
    if not env.get("cpu", {}).get("governors_uniform"):
        warnings.append("CPU governors are not uniform across cores.")
    # A stored None means the field was unavailable; dict.get's default only
    # applies when the key is absent.  Preserve that distinction in the capture,
    # but normalize locally before using string operations in the validator.
    proc_swaps = env.get("swap", {}).get("proc_swaps") or ""
    if proc_swaps.count("\n") == 0:
        warnings.append(
            "No swap configured. Anonymous pages cannot be reclaimed; use "
            "--backing FILE, or configure zram/swap before anon experiments."
        )
    if not env.get("damon", {}).get("sysfs_present") and \
       not env.get("idle_page_tracking", {}).get("bitmap_present"):
        warnings.append(
            "Neither DAMON sysfs nor idle-page tracking is present. The "
            "access-rate estimator will be limited to workload self-report."
        )

    return {
        "publishable": len(blockers) == 0,
        "blockers": blockers,
        "warnings": warnings,
    }


def validate_for_claims_safe(env: Dict[str, Any]) -> Dict[str, Any]:
    """Validate fail-closed so an internal error can never certify a host.

    The captured environment remains intact (including unavailable fields as
    null), while the validation failure is recorded as an explicit blocker.
    Catch Exception rather than BaseException so interrupts still propagate.
    """
    try:
        return validate_for_claims(env)
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        return {
            "publishable": False,
            "blockers": [
                "Environment validation failed internally "
                f"({error_type}: {error_message}). This run cannot be certified."
            ],
            "warnings": [],
            "validation_error": {
                "type": error_type,
                "message": error_message,
            },
        }


def write_capture(out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = capture()
    verdict = validate_for_claims_safe(env)
    env["_verdict"] = verdict
    payload = json.dumps(env, indent=2, sort_keys=True, default=str)
    path = out_dir / "environment.json"
    path.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    (out_dir / "environment.sha256").write_text(digest + "\n")
    return {"env": env, "verdict": verdict, "sha256": digest, "path": str(path)}


if __name__ == "__main__":
    import sys
    r = write_capture(Path(sys.argv[1] if len(sys.argv) > 1 else "./run"))
    v = r["verdict"]
    print(f"environment.json written  sha256={r['sha256'][:16]}...")
    print(f"publishable: {v['publishable']}")
    for b in v["blockers"]:
        print(f"  BLOCKER: {b}")
    for w in v["warnings"]:
        print(f"  warning: {w}")
