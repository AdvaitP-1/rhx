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

### B2. Unit of replication is wrong — n = 1
Gate 1 performs ONE reclaim event and scores coverage across ~30 snapshot times.
Those times are serially correlated within a single trial, so treating them as
30 independent observations overstates evidence by roughly the autocorrelation
factor. The correct unit of replication is the **reclaim event**, not the
snapshot.
**Required**: repeat each configuration `n >= 20` times with independent seeds,
report the distribution of per-trial coverage, and give a confidence interval on
the mean. Gate 2 has the same defect: amplification is computed from a single
sequence per arm with no repetitions and no interval.
**Not fixed** — this is a change to the experimental protocol, not a code bug,
and must be a deliberate decision.

### B3. `reclaim_natural` can trigger the OOM killer
Mode B lowers `memory.max`, which is a hard limit. If the cgroup cannot be
reclaimed to the target, the kernel OOM-kills the workload, destroying the trial
and possibly the machine's stability under repetition.
**Required**: use `memory.high` (throttles, does not kill) for the pressure ramp,
or check `memory.events.oom_kill` after every reclaim and abort the trial if it
incremented. Neither is currently done, and the harness does not verify the
workload is still alive after Mode B reclaim.

### B4. No workload quiesce around the victim-set measurement
The victim set is `resident_before AND NOT resident_after`. Between the two
snapshots the workload continues touching pages, so some residency changes are
caused by ordinary access rather than by reclaim. This biases the measured
victim set and therefore `S(t)`.
**Required**: a `PAUSE`/`RESUME` control command so the mapping is quiescent
across the reclaim window, or an explicit correction using the known access
rates. The control protocol has no such command.

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
produces traces to a measurement or to a declared ground truth. B1 is fixed.
B2 through B4 must be addressed before any result is published: they are
protocol and safety defects, not measurement-validity defects, but each is
independently sufficient for a reviewer to reject.
