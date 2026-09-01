# Quick Start

Run `bash INSTALL.sh` once after extracting, then `./run.sh check`.

## 1. Install

```bash
tar xzf rhx.tar.gz && cd rhx
pip install numpy scipy
./run.sh build
```

## 2. Check the machine

```bash
sudo ./run.sh check
```

Prints a pass/fail line per requirement with the fix for each failure. It refuses
to certify a virtualized or non-Linux host: **VM for development, bare metal for
claims.**

Common fixes:

```bash
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled   # THP must not be 'always'
# PSI absent: rebuild kernel with CONFIG_PSI=y
```

## 3. Verify the code

```bash
./run.sh selftest        # 36 checks, no root needed
```

Must be 36 pass / 0 fail before any experiment.

## 4. Run the gates

```bash
sudo ./run.sh all runs/$(date +%F)
```

Or one at a time:

```bash
sudo ./run.sh gate0 runs/g0     # can we measure f(lambda)?
sudo ./run.sh gate1 runs/g1     # pre-registered prediction test
sudo ./run.sh gate2 runs/g2     # homogeneous negative control
```

`all` stops at the first failing gate. That is intentional: a Gate 1 result is
uninterpretable if Gate 0 failed, because a prediction miss could not be
attributed to the theory rather than to the estimator.

## Tuning

Environment variables, with defaults:

```bash
PAGES=200000      # 200k pages = 800 MiB working set
MEASURE_S=300     # observation window before reclaim
WINDOW_T=60       # declared operating window; lambda_min = 1/T
TRIALS=20         # replication unit is the reclaim EVENT
MODE=proactive    # proactive | natural
```

Example:

```bash
sudo PAGES=500000 TRIALS=30 MODE=natural ./run.sh gate1 runs/big
```

## The two reclaim modes are not interchangeable

**proactive** writes `memory.reclaim`. Documented as *not* accounted as cgroup
pressure. Same victim machinery, so valid for page-dynamics claims: recovery
kernel, composition shift, amplification. Invalid for anything involving PSI,
damage or SLOs.

**natural** applies pressure via `memory.high`, which throttles rather than
kills. Produces real pressure and real PSI. Required for distortion and SLO
claims. Run both; every output file records which mode produced it.

## Development on a VM

```bash
sudo EXTRA=--allow-unpublishable ./run.sh gate0 runs/dev
```

Every output is stamped development-only and cannot be mistaken for evidence.

## What you get per run

```
runs/g1/
  protocol.json            frozen before trial 1; changing it is refused
  prereg_ledger.jsonl      hash-chained; tampering is detectable
  predictions/*.json       committed before reclaim
  UNBLINDING_KEY.json      do not open until analysis is finalized
  trials_manifest.json     per-trial results + bootstrap CI on the mean
  trial_000_seed*/
    environment.json       kernel, MGLRU, THP, swap, NUMA, governors
    gate1_results.json     coverage, diagnosis, family comparison
    telemetry/samples.jsonl  ~150 metrics per sample
    telemetry/samples.csv
    workload/truth.tsv     ground-truth rates, written before any access
    workload/resid_*.txt   mincore residency bitmaps
```

## Reading a Gate 1 failure

The score diagnoses the failure mode rather than only reporting a miss:

- **offset-like** — deviation roughly constant in `t`. Implicates the estimator
  (frozen mass, victim set). Fix the estimate, not the theory.
- **shape-like** — deviation trends with `t`. Implicates the kernel family and
  therefore the Poisson access assumption. This is a falsification.

## Bias controls

Frozen protocol, cryptographic pre-registration, declared exclusion rules,
interleaved randomized arm order, analyst blinding, order-effect permutation
test, covariate balance check, bootstrap CI over trials rather than snapshots,
optional-stopping guard, OOM safety, quiesced measurement. See `AUDIT.md`.

## Not implemented

Gate 3 (real workloads: Redis, RocksDB, PostgreSQL, JVM). Do not attempt until
Gates 0 through 2 pass on bare metal.
