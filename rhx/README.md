# RHX — Reclaim Hysteresis Experiment Harness

Measurement program for the paper *Reclaim Hysteresis: History-Dependent Memory
Reclamation from Page-Access Dynamics*.

The gates are ordered so each can falsify the theory before the next is
attempted. Do not skip ahead: a Gate 1 result is uninterpretable if Gate 0 failed,
because you would not know whether a prediction miss came from the theory or from
the estimator.

## Quick start

```bash
cd workload && make && cd ..
python3 selftest.py                    # 36 checks, must pass before anything else
```

## Gate 0 — can the theoretical primitive be measured?

Runs the synthetic workload, whose per-page rates are drawn from a chosen
distribution and written to disk *before any access occurs*, then estimates the
density and reports error in the quantities the experiment depends on.

```bash
sudo python3 gates/gate0_estimator.py --out runs/gate0 \
    --pages 200000 --measure-s 300 --window-T 60 --dist gamma --p1 0.8 --p2 0.5
```

Pass criteria are declared in `PASS_CRITERIA` at the top of the file. Changing
them after seeing results invalidates the gate.

## Gate 1 — the pre-registered prediction test

Estimates `f(lambda)`, computes the predicted recovery bracket, **commits it to a
hash-chained ledger**, then reclaims and measures. Re-registering the same label
is refused, so the prediction cannot be revised after seeing the outcome.

```bash
sudo python3 gates/gate1_prediction.py --out runs/gate1a \
    --label gamma08_proactive --reclaim-mode proactive \
    --pages 200000 --measure-s 300 --window-T 60

sudo python3 gates/gate1_prediction.py --out runs/gate1b \
    --label gamma08_natural --reclaim-mode natural \
    --pages 200000 --measure-s 300 --window-T 60
```

Run **both modes**. See "Reclaim modes" below — they are not interchangeable.

## Gate 2 — the homogeneous negative control

Theorem 6 predicts a null. Two arms differing only in the rate distribution, run
in randomized order:

```bash
sudo python3 gates/gate2_negative_control.py --out runs/gate2 \
    --reclaim-mode proactive --pages 200000 --n-reclaims 6
```

The gate refuses to certify the null if the positive arm showed no effect, since
a null from an underpowered design is uninformative.

## Reclaim modes — these are not interchangeable

**Mode A, proactive** (`memory.reclaim`). Kernel documentation states this is not
accounted as memory pressure on the cgroup. It drives the same victim-selection
machinery, so it is valid for page-dynamics claims — recovery kernel, composition
shift, amplification. It is **invalid** for any claim involving damage, PSI or
SLOs, because the pressure signal it would need is absent by construction.

**Mode B, natural** (lower `memory.high`). Produces genuine contention and real
PSI without imposing an OOM-enforced hard limit. Required for distortion and
SLO claims. OOM counters and workload liveness are still checked; any kill
invalidates the trial.

Every output file records which mode produced it. A reviewer who sees only Mode A
results supporting a distortion claim is entitled to reject it, so run both.

## Environment gate

`envcapture.py` refuses to certify a run as publishable on a non-Linux host, a
virtualized host, without cgroup v2, or without root. `--allow-unpublishable`
overrides for development and stamps every output as development-only.

The rule: **VM for development, bare metal for claims.**

Warnings that do not block but must be addressed before publication: MGLRU state
unreadable, THP set to `always` (huge pages change the reclaim unit and blur
per-page residency), non-uniform CPU governors, no swap configured with
`--backing ANON`.

## What is measured

Residency comes from `mincore(2)` inside the workload, giving exact per-page
presence rather than inference from refault counters.

Telemetry samples, per interval: `memory.current`, `memory.peak`, all four limit
knobs, swap and zswap current, ~55 `memory.stat` fields, `memory.events` and
`memory.events.local`, memory/io/cpu PSI (both `some` and `full`, all four
fields), ~40 `/proc/vmstat` counters, all of `/proc/meminfo`, system-wide PSI, and
load average. Absent fields record as `null`, distinguishing "this kernel does not
export it" from "the value is zero".

Cumulative counters are stored raw; differencing happens in analysis, so a dropped
sample cannot corrupt a total. `samples.jsonl` is the source of truth; the CSV is
generated from the union of all keys ever seen so no field is silently dropped.

## The observation window

Registering the asymptotic recovery exponent would require resolving the density
near `lambda = 0`, which needs observation windows over which the static-rate
assumption fails. Instead a window `T` is declared in advance, `lambda_min = 1/T`
is fixed, and everything colder enters as a single frozen-mass scalar.

That substitution is **not** an identity — an earlier draft of this work claimed
it was, incorrectly. Replacing the cold contribution by the frozen mass has
worst-case error `Q*(1 - 1/e)`. The harness instead reports a rigorous bracket,
using monotonicity of `exp(-lambda t)`:

```
Q*exp(-t/T)  <=  cold-tail contribution  <=  Q
```

Both ends are attainable, so the truth is guaranteed inside. The bracket width is
the honest cost of not resolving the cold tail, and it shrinks as `T` shortens.

Choose `T` from the control timescale, not from theory.

## Reading a Gate 1 failure

`score_prediction` diagnoses the failure mode rather than only reporting a miss:

- **offset-like** (deviation roughly constant in `t`) implicates the estimator —
  wrong frozen mass or wrong victim set. Fix the estimate, not the theory.
- **shape-like** (deviation trends with `t`) implicates the kernel family, and
  therefore assumption A1, Poisson access. That is a falsification of the theory.

## Layout

```
workload/rategen.c          synthetic workload, known ground-truth rates
harness/cgroupv2.py         cgroup v2 + full metric collection, both reclaim modes
harness/envcapture.py       environment capture and the publishability gate
harness/prereg.py           hash-chained pre-registration ledger
harness/estimator.py        f(lambda) estimation, rigorous recovery bracket
harness/workload.py         workload process control, mincore snapshots
harness/telemetry.py        background sampler, event log on the same clock
analysis/kernels.py         family fitting, prediction scoring, frontier, amplification
analysis/randomization.py   randomization inference, confounding demo, power curves
gates/gate0_estimator.py    Gate 0
gates/gate1_prediction.py   Gate 1
gates/gate2_negative_control.py  Gate 2
selftest.py                 36 checks
```

## Known limitations

The Poisson access model (A1) is a first-order approximation; real workloads have
phases and correlated access. Gate 1 is designed so a failure localizes here.

`victim_weights` offers `coldest_first` (the analytic idealization) and `softmin`
(a smooth stand-in for MGLRU's generation-based selection). Neither is MGLRU. The
gap between predicted and observed bracket width measures victim-selection
fidelity and should be reported, not hidden.

Gate 3, real workloads, is not implemented. It should not be attempted until
Gates 0 through 2 pass.
