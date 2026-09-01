# Running RHX with Codex

## What Codex can and cannot do here

| Task | Codex cloud | Codex CLI on your Linux box |
|---|---|---|
| Read code, review, refactor | yes | yes |
| `./run.sh selftest` (36 checks) | yes | yes |
| Development gate runs (`--allow-unpublishable`) | yes | yes |
| Implement Gate 3, DAMON estimator, figures | yes | yes |
| **Publishable experiments** | **no** | **yes, on bare metal** |

Codex cloud runs in a container: virtualized, no root over the cgroup hierarchy,
`memory.reclaim` not writable. `./run.sh check` will refuse it. That refusal is
the harness working correctly, not something to bypass.

---

## Path A — Codex cloud, for code work

Push the repo to GitHub, then in the Codex web interface point a task at it.
Codex reads `AGENTS.md` automatically, which tells it the constraints.

Setup script for the environment:

```bash
pip install numpy scipy
cd workload && make && cd ..
python3 selftest.py
```

Prompts that work well:

> Read AGENTS.md. Implement Gate 3 in gates/gate3_real_workloads.py, following
> the structure of gate1_prediction.py. The workload is an external process
> (Redis) rather than rategen, so f(lambda) must come from DAMON or idle-page
> tracking instead of workload self-report. Record the region-to-page
> aggregation and its bias explicitly. Do not weaken any pass criterion.

> Read AGENTS.md. Implement the DAMON estimator in harness/estimator.py as
> estimate_from_damon(). DAMON reports region-level access frequency over
> sampling intervals, not per-page rates. The estimator must record the region
> boundaries and the aggregation bias so Gate 0 can quantify it. Add
> corresponding checks to selftest.py.

> Read AGENTS.md. Add figure generation in analysis/figures.py for the six
> figures in the paper: resident density before/after reclaim and after partial
> recovery; f(lambda) with the advancing frontier; predicted vs observed
> recovery kernels with the pre-registered bracket; matched-state different-history
> comparison; marginal distortion curves with the common mu*; oversubscription
> vs SLO violation. Use matplotlib, no seaborn, one figure per function.

> Read AGENTS.md, then review harness/estimator.py adversarially. I want you to
> find cases where the rigorous bracket could fail to contain the truth. Write
> failing tests first if you find any.

Always tell it to read `AGENTS.md` first. It contains the constraints that keep
it from "helpfully" widening a pass criterion.

---

## Path B — Codex CLI on bare metal, for real experiments

This is the one that produces evidence.

### 1. Get a bare-metal Linux box

Not a VM. Options: a spare desktop, a Hetzner or OVH dedicated server (roughly
$40–70/month), or a bare-metal instance from AWS (`m5.metal`), Equinix, or
Vultr. You need root and a kernel with `CONFIG_PSI=y`. Ubuntu 22.04+ or Fedora
38+ ships with everything.

### 2. Install Codex CLI on that box

```bash
npm install -g @openai/codex
codex login
```

### 3. Set up RHX

```bash
scp rhx.tar.gz user@yourbox:~
ssh user@yourbox
tar xzf rhx.tar.gz && cd rhx
bash INSTALL.sh
./run.sh selftest        # must be 36 pass / 0 fail
```

### 4. Fix whatever `check` complains about

```bash
sudo ./run.sh check
```

Typical fixes:

```bash
# THP must not be 'always'
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled

# cgroup v2 memory controller not delegated
sudo mkdir -p /sys/fs/cgroup/rhx
echo "+memory" | sudo tee /sys/fs/cgroup/cgroup.subtree_control

# PSI absent: check first, it is usually on
grep CONFIG_PSI /boot/config-$(uname -r)
# if not set you need a kernel rebuild or a different distro image

# pin CPU frequency for reproducible timing
sudo cpupower frequency-set -g performance
```

Do not proceed until `check` prints READY.

### 5. Run the gates

```bash
sudo ./run.sh gate0 runs/$(date +%F)-g0
```

Gate 0 asks whether `f(lambda)` is measurable at all. If it fails, nothing
downstream is interpretable. Default is 20 trials at 200k pages with a 300s
observation window — budget roughly two hours.

```bash
sudo ./run.sh gate1 runs/$(date +%F)-g1a                 # proactive
sudo MODE=natural ./run.sh gate1 runs/$(date +%F)-g1b    # natural
```

Run both modes. They test different things and are not interchangeable.

```bash
sudo ./run.sh gate2 runs/$(date +%F)-g2
```

Or all three with automatic stop-on-failure:

```bash
sudo ./run.sh all runs/$(date +%F)
```

### 6. Use Codex to interpret, not to decide

```bash
codex "Read AGENTS.md. Read runs/2026-09-01-g1a/trials_manifest.json and
summarize: the bootstrap CI on coverage, how many trials were excluded and
under which rules, the order-effect test result, and which kernel family won
out-of-sample. Do not interpret whether the gate passed; just report what the
numbers are."
```

Keep the decision in the frozen protocol, not in the model's judgment. Ask Codex
to report, and read the verdict yourself.

---

## The mistake to avoid

Do not ask Codex to make a gate pass. If Gate 1 comes back with coverage below
threshold, the harness already diagnoses why: an offset-like deviation implicates
the estimator, a shape-like deviation implicates the theory. A shape-like failure
means the composition-shift mechanism is wrong, and the right response is to
report that and fall back to the history-free results, which stand on their own.

A research program that cannot fail is not measuring anything.
