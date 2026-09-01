# Independent Audit of the RHX Harness

Findings from an adversarial review of the harness, ordered by severity.
"Blocking" means a reviewer could reject results produced with the defect present.

## BLOCKING

### B1. Sparse backing file (FIXED)
`ftruncate` created a sparse file; read-only faults on a hole may be served by
the shared zero page rather than allocating a page-cache page. Reclaim would
then have nothing to evict and `mincore` residency would be meaningless.
Verified: after `ftruncate` and read-only touching every page, `st_blocks` was 0.
**Fix applied**: every block is written with real data and `fsync`ed before
mapping. Verified `st_blocks*512 == st_size` after the fix.

### B2. Unit of replication was wrong — n = 1 (FIXED)
Gate 1 previously performed ONE reclaim event and scored coverage across ~30
snapshot times. Those times are serially correlated within a single trial, so
treating them as 30 independent observations overstates evidence by roughly the
autocorrelation factor. The correct unit of replication is the **reclaim
event**, not the snapshot.
**Fix applied**: `run_trials.py` repeats each configuration with independent
seeds (20 trials by default), treats each reclaim event as one replication unit,
and reports bootstrap confidence intervals over per-trial outcomes. Gate 1's
aggregate decision uses the confidence interval rather than treating snapshots
as independent observations.

### B3. `reclaim_natural` could trigger the OOM killer (FIXED)
Mode B previously lowered `memory.max`, which is a hard limit. If the cgroup
cannot be reclaimed to the target, the kernel OOM-kills the workload, destroying
the trial and possibly the machine's stability under repetition.
**Fix applied**: natural reclaim uses `memory.high`, which throttles rather than
imposing an OOM-enforced hard limit. The harness reads OOM counters before and
after reclaim and checks workload liveness; a kill delta invalidates the trial.

### B4. No workload quiesce around the victim-set measurement (FIXED)
The victim set is `resident_before AND NOT resident_after`. Previously, between
the two snapshots the workload continued touching pages, so some residency
changes were caused by ordinary access rather than by reclaim. This biases the
measured victim set and therefore `S(t)`.
**Fix applied**: the control protocol now has `PAUSE`/`RESUME`. The workload is
paused before the pre-reclaim snapshot and resumed after the post-reclaim
snapshot. Pending events shift by the pause duration, preventing a burst on
resume.

### B5. Environment validation crashed on unavailable fields (FIXED)
`dict.get(key, default)` does not apply its default when a key exists with value
`None`. On hosts without `/proc/swaps`, validation called `.count()` on `None`
and crashed. The validator now normalizes the unavailable value only for the
string operation, while retaining `null` in the capture. `write_capture` invokes
a fail-closed wrapper: any unexpected validation exception becomes an explicit
publishability blocker and is recorded in the verdict.

## SIGNIFICANT

### S1. `mincore` on MAP_SHARED reports global page-cache residency
Page cache is shared machine-wide. If any other process reads the backing file,
pages return to residency without the workload touching them, contaminating
`S(t)`. In a controlled single-tenant experiment nothing else opens the file, but
this must be asserted and checked, not assumed.

### S2. Victim kernels are not MGLRU
`coldest_first` is the analytic idealization; `softmin` is a smooth stand-in.
Neither is what MGLRU actually does. The gap between predicted and observed
bracket width measures victim-selection fidelity, which is informative, but no
claim about MGLRU's policy can be made from these.

### S3. THP is warned about but not enforced
Huge pages change the reclaim unit and blur per-4KiB residency. The environment
capture warns when THP is `always` but nothing prevents a run. Characterization
runs should set THP to `never` or `madvise` and record it.

### S4. No multiple-comparison control
Gate 0 evaluates four criteria, Gate 2 two, and family comparison ranks four
kernels. Across a full experimental program this inflates false-positive risk.
Pre-register the primary endpoint per gate and label everything else secondary.

### S5. Scheduling lag is measured but not gated
`rategen` reports `max_lag` but no gate fails on it. If aggregate rate exceeds
what the machine can schedule, realized rates fall below assigned rates and the
ground truth silently becomes wrong. Add a hard check that `max_lag` stays below
a declared fraction of the shortest inter-access interval.

## MINOR

### M1. `Cgroup.destroy` assumes it may move processes to the root cgroup, which
fails under delegation.
### M2. `extra_sampler` exists but is never wired to workload-side state, so
telemetry rows contain no workload counters.
### M3. `snapshot_schedule` starts at `max(T/200, 0.05)`; for large page counts
a `mincore` snapshot may itself take longer than the first interval.
### M4. Gate 3 (real workloads) is intentionally absent.

## What the harness does correctly

Ground truth is written before any access. The Poisson generator is validated
statistically rather than assumed (chi-square GOF p = 0.87, aggregate z = -0.45,
law of total variance within 25%). Residency is measured with `mincore`, not
inferred. The victim set is measured, not assumed. Pre-registration is
hash-chained with tamper detection verified in the self-test. Both reclaim modes
are implemented and labeled in every output. Absent kernel fields record as null,
distinguishing "not exported" from "zero". Cumulative counters are stored raw.
The environment gate refuses to certify virtualized or non-Linux hosts. The
prediction bracket is a rigorous bound, not a point estimate, and is verified to
contain the truth. The randomization test is calibrated at nominal size
(observed 0.033 at alpha 0.05 over 120 trials).

## Verdict

The harness does not fabricate, assume, or approximate data. Every number it
produces traces to a measurement or to a declared ground truth. B1 through B5
are fixed. The significant and minor limitations above still apply and must be
reported honestly.
