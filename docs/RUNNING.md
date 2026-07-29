# Running the Search

This guide describes the current executable behavior. Mathematical conclusions
must be interpreted under the finite-box contract in
[Mathematical Correctness](../MATHEMATICAL_CORRECTNESS.md).

## Installation

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install "pytest>=7"
```

The search is started from the repository root:

```bash
python opn_main.py
```

There is no command-line configuration layer. Edit the configuration block at
the top of `opn_core.py` before starting a new search.

## Search-box parameters

These values define the finite domain and therefore change the conclusion of a
completed run:

| Parameter | Meaning |
|---|---|
| `MAX_PRIME` | Upper bound for odd primes in the complete prime pool |
| `MAX_FACTORS` | Maximum number of explicitly selected distinct prime bases |
| `MAX_EXP` | Maximum exponent permitted for a prime |
| `PROPAGATE` | `True` for Euler-form factor chains; `False` for spoof DFS |
| `SEARCH_MODE` | Target abundancy index, Euler rule, and forced/excluded primes |

The checked-in box is `P=20,000,000,000`, `F=61`, `E=35`. For a short
functional run, use `P=100,000`, `F=12`, `E=6`.

A checkpoint stores its own prime limit, factor limit, exponent limit, search
mode, and frontier. If `checkpoint_merged.pkl` exists, the program attempts to
resume that saved box. Move the checkpoint out of the repository or remove it
before intentionally starting a different search.

In Euler-form OPN mode, the pool contains every permitted prime divisor of
$N$, and `MAX_FACTORS` bounds $\omega(N)$. In spoof DFS mode, these limits apply
only to explicitly selected prime bases. The derived pseudo-prime factor `r`
is outside the pool contract and is not counted by `MAX_FACTORS`.

## Mathematical-policy controls

The current OPN configuration uses:

```python
ENABLE_FERMAT_DEBT = False
Q3_PREPOOL_MODE = "enforce"
DOMAIN_RATIO_MODE = "off"
PENDING_SELECTION = "fifo"
```

`off`, `shadow`, and `enforce` have distinct meanings:

- `off`: do not evaluate the optional bound;
- `shadow`: evaluate and report what the bound would do without pruning;
- `enforce`: apply the proven bound to the search.

For `Q3_PREPOOL_MODE`, the evaluated quantity is the exact
$v_3\!\left(\sigma(p^{\mathrm{exp}})\right)$ obtained from an LTE closed form.
In `enforce` mode, an
excluded, overrun, or exponent-budget contradiction for prime 3 is rejected
before sigma-database lookup, plan construction, or GCD scanning. In `shadow`
mode, the ordinary sigma path still runs and records whether its valuation
agreed with the precheck. The checked-in policy uses `enforce`.

Touchard congruence is an independent always-active OPN condition rather than a
mode in this configuration block. It may force prime 3 into the pending chain
when the selected Euler prime is 2 modulo 3. The q=3 prepool check then accounts
for how processed sigma factors contribute to the required exponent of 3.

The active policy is written to `manifest.json`. Runs should not be compared as
the same mathematical experiment unless their manifests agree on the search
box, target, propagation mode, and pruning policy.

### Friend-of-10 verification mode

`opn_core.py` also defines the inactive `FRIEND_10_MODE` preset:

```python
SEARCH_MODE = FRIEND_10_MODE
PROPAGATE = True
```

Two distinct positive integers are friends when they have the same abundancy
index. Since $I(10)=\sigma(10)/10=9/5$, this mode searches the corresponding
odd candidate domain described by
[Thackeray (2024)](https://doi.org/10.1016/j.indag.2024.04.011). It sets the
target to $9/5$, forces prime 5 with minimum exponent 2, excludes prime 3, and
uses all-even exponents, so every represented candidate is an odd square.

The commented preset beside the mode definition also supplies a small
verification box. Treat it as a separate experiment: move any OPN checkpoint
aside before switching, and confirm the resulting manifest before interpreting
the run.

The mode exercises the arithmetic conditions used in friend-of-10 research.
A finite run proves only the corresponding configured box is empty; selecting
the preset alone is not a reproduction of an external theorem or published
large-scale computation.

## Engineering controls

The principal engineering settings are:

```python
CHECKPOINT_INTERVAL_SECONDS = 300.0
POOL_GCD_MODE = "hierarchical"
POOL_SUPERBLOCK_FANOUT = 16
SIGMA_DATABASE_ENABLED = True
SIGMA_DATABASE_FILE = "sigma_pool.sqlite3"
POOL_PLAN_BUILD_POLICY = "adaptive"
POOL_PLAN_DISK_CACHE_ENABLED = True
POOL_PLAN_DISK_CACHE_DIR = "plan_cache"
POOL_PLAN_DISK_MIN_FREE_BYTES = 2 * 1024**3
```

These controls change resource use and execution paths, not the intended
mathematical search set. `hierarchical` is the production GCD mode. `flat`
retains leaf products and is primarily a correctness oracle.

`adaptive` plan construction starts on demand and begins batched construction
after repeated distinct plan misses. This avoids unnecessary plans on a warm
database while preventing excessive build fragmentation on a cold search.

## Cold, warm, and expansion runs

A **cold run** uses neither compatible sigma rows nor compatible plan entries.
It includes prime generation, plan construction, and all required pool scans.

A **warm run** repeats a box after its sigma analyses have been persisted.
Memory-cache hits are process-local, but validated SQLite rows survive process
exit. If the database covers all requested `(p, exponent)` pairs, plans may
never be needed.

An **expansion run** increases the prime window:

- exact sigma rows remain reusable because their factorization is complete;
- compatible partial rows retain the complete outside-window residual and can
  continue into the new interval after validation;
- plan entries are keyed by their source interval and pool identity, so a
  different window generally creates additional entries rather than mutating
  an old plan.

Increasing `MAX_EXP` preserves rows for exponent values already analyzed. New
exponents require new sigma rows and, when necessary, new component plans.

## Persistent derived data

### Sigma database

`sigma_pool.sqlite3` stores validated exact or window-partial sigma analyses.
Writes are batched and flushed at stable search boundaries. The database is
not required for correctness and is not a checkpoint.

To discard it, stop the process and remove all three possible SQLite files
together:

```text
sigma_pool.sqlite3
sigma_pool.sqlite3-wal
sigma_pool.sqlite3-shm
```

Deleting them during a run can lose pending writes or damage the cache.

### Plan cache

`plan_cache/` stores filtered prime arrays and serialized superblock products.
Compatible entries can be reused across restarts. Different windows,
component orders, integer widths, or plan geometry use distinct keys, so the
directory can grow over a sequence of experiments.

Entries are derived data. Stop the process before deleting the directory.
Missing, corrupt, incompatible, busy, or space-constrained entries fall back
to rebuild or in-memory operation.

## Progress and stopping

The console shows completed states, elapsed time, average rate, assigned-factor
count, ratio, and resonance telemetry.

`reson` is a project-defined structural diagnostic based on reuse of odd sigma
factors already present in the state versus newly introduced factors. It is
not a probability, candidate score, or proof-strength measure. Its priority
weight is zero in the checked-in configuration, so it does not affect search
order, pruning, or completeness.

- One `Ctrl+C` requests a stop at a stable state boundary, writes a complete
  checkpoint, and reports `STOPPED`.
- A second `Ctrl+C` interrupts immediately. The most recent complete atomic
  checkpoint is retained, but a small amount of work may be repeated.
- A normally completed search removes `checkpoint_merged.pkl`.

The runtime sampler writes process measurements to the run directory while the
search is active. See [Outputs and Recovery](OUTPUTS_AND_RECOVERY.md) for file
semantics and comparison rules.

## Preflight for a large run

Before starting a large window:

1. confirm that `opn_core.py` contains the intended box and policy;
2. move any checkpoint belonging to a different experiment;
3. record free RAM and disk space;
4. decide whether existing sigma and plan caches are part of the experiment;
5. preserve the starting Git commit and dirty-worktree state;
6. run the test suite;
7. perform a smaller-box run with the same policy.

Use a fresh database and plan directory when measuring cold performance. Use
the normal persistent locations when the objective is efficient continuation.
