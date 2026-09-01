# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

RHX is the measurement harness for a systems-research hypothesis about Linux
memory reclaim. It is **not** a product. Its purpose is to produce evidence that
could falsify a theory, so correctness and honesty matter more than features.

The hypothesis: kernel reclaim selects cold pages, which biases the access-rate
composition of the surviving resident set, which raises the cost of the next
reclaim, and that bias decays as evicted pages fault back in.

## Hard constraints — do not violate these

**Never fabricate, simulate, or default data.** If a measurement is unavailable,
record `null`. There is a deliberate distinction throughout between "this kernel
does not export this field" and "the value is zero". Never collapse them.

**Never widen a pass criterion to make a gate pass.** `PASS_CRITERIA` at the top
of each gate file and the frozen `protocol.json` are the decision rule. Changing
them after data exists invalidates the experiment. If a gate fails, the correct
action is to report the failure or fix a genuine bug, never to move the line.

**Never re-register a pre-registered prediction.** `prereg.py` refuses this by
design. If you find yourself wanting to, the answer is a new label in a new run
directory.

**Never mix reclaim modes.** `proactive` (memory.reclaim) is documented as not
accounted as cgroup pressure. It is valid for page-dynamics claims only. Any
claim touching PSI, damage, or SLOs requires `natural` mode. Every output file
records which mode produced it; do not pool them.

**Never skip a gate.** Gate 1 is uninterpretable if Gate 0 failed, because a
prediction miss could not be attributed to the theory rather than the estimator.

**Do not add features that do not answer the research question.** The project
has enough surface area. It needs evidence.

## Before you change anything

```bash
bash INSTALL.sh
./run.sh selftest      # must be 36 pass / 0 fail
```

If the self-test fails after your change, your change is wrong until proven
otherwise. Treat the self-test as the spec.

## Environment

`./run.sh check` reports whether the machine can produce publishable data. It
refuses virtualized hosts, non-Linux hosts, missing cgroup v2, missing PSI, and
non-root. **This refusal is correct behaviour.** Do not remove or weaken it.

In a sandboxed agent environment it will fail. That is expected. Use
`EXTRA=--allow-unpublishable` for development runs; every output is then stamped
development-only.

## Layout

```
workload/rategen.c        C workload, known per-page Poisson rates (ground truth)
harness/cgroupv2.py       cgroup v2 + ~150 metrics per sample, both reclaim modes
harness/envcapture.py     environment capture + publishability gate
harness/prereg.py         hash-chained pre-registration ledger
harness/estimator.py      f(lambda) estimation, rigorous recovery bracket
harness/workload.py       process control, mincore snapshots, PAUSE/RESUME
harness/telemetry.py      background sampler, event log on one monotonic clock
harness/bias.py           frozen protocol, blinding, exclusions, drift tests
analysis/kernels.py       kernel family fitting, prediction scoring, frontier
analysis/randomization.py randomization inference, confounding demo, power
gates/gate0_estimator.py  can we measure f(lambda)?
gates/gate1_prediction.py pre-registered prediction test
gates/gate2_negative_control.py  homogeneous null
run_trials.py             N independent trials + bootstrap CI
selftest.py               36 checks
```

## Subtleties that have already caused bugs

**The backing file must not be sparse.** `ftruncate` alone creates holes;
read-only faults on a hole may be served by the shared zero page, leaving nothing
for reclaim to evict and making `mincore` residency meaningless. `rategen`
materializes every block and `fsync`s. Verified by checking `st_blocks*512 ==
st_size`. Do not "optimize" this away.

**The frozen-fraction split is a bound, not an identity.** Pages colder than
`lambda_min = 1/T` are not resolvable in the window. Their contribution to `S(t)`
is bracketed by `Q*exp(-t/T) <= contribution <= Q`, both ends attainable. An
earlier version treated it as an identity, which was wrong. `predict_recovery`
returns a rigorous bracket; do not replace it with a point estimate.

**The replication unit is the reclaim event, not the snapshot.** Snapshot times
within one trial are serially correlated. Aggregate over trials via
`run_trials.py`, never over snapshots within a trial.

**A wide rate band is a mixture of Poissons and is overdispersed by
construction.** `var/mean != 1` there is correct, not a bug. The invariant that
holds for any band is the law of total variance:
`Var(count) = E[lam]*W + W^2*Var(lam)`.

**Measurement must be quiesced.** `wl.pause()` before the pre-reclaim snapshot
and `wl.resume()` after the post-reclaim snapshot, so the measured victim set is
attributable to reclaim rather than to ordinary access. `rategen` shifts pending
events by the pause duration so resuming causes no burst.

**Mode B uses `memory.high`, not `memory.max`.** `memory.max` is a hard limit and
OOM-kills the workload. OOM counters are checked before and after; a nonzero kill
delta marks the trial invalid rather than analyzable.

## Good tasks for an agent

- Implement Gate 3 (real workloads: Redis, RocksDB, PostgreSQL, JVM). Reuse the
  Gate 1 structure; the workload is external so `f(lambda)` must come from DAMON
  or idle-page tracking rather than workload self-report.
- Implement the DAMON estimator in `estimator.py`. DAMON reports region-level
  access frequencies, not per-page rates; the region-to-page mapping and its
  aggregation bias must be recorded, not hidden.
- Add figure generation: the six figures listed in the paper's Section 8.8.
- Extend `analysis/` with the family-comparison and drift plots.
- Add a `--dry-run` mode to the gates that exercises every code path without
  requiring root.

## Bad tasks

- Adding CLI polish, packaging, or orchestration features.
- Relaxing the environment gate so runs "work" in a sandbox.
- Replacing `mincore` residency with inference from refault counters.
- Any change whose effect is to make a gate pass.
