# Outputs and Recovery

Every invocation receives a run identifier of the form

```text
YYYYMMDD-HHMMSS_P<prime>_F<factors>_E<exponent>
```

and writes reports under `runs/<run_id>/`. Resuming a checkpoint continues to
use the saved run identifier and appends resource samples to the same CSV.

## Run reports

| File | Purpose |
|---|---|
| `summary.txt` | Status, source identity, search box, elapsed time, and report index |
| `manifest.json` | Machine-readable configuration, pruning policy, Git commit, and dirty-worktree flag |
| `structure.txt` | Human-readable mathematical search structure |
| `structure.json` | Machine-readable prune counts, depths, propagation edges, sigma classifications, and valuation contradictions |
| `performance.txt` | Human-readable phase, cache, GCD, plan, and memory measurements |
| `performance.json` | Machine-readable engineering metrics |
| `performance_samples.csv` | Time series of phase, progress, frontier, RSS, CPU, threads, and system memory |
| `abundancy_gap_states.jsonl` | Complete streamed records for qualifying productive partial states |
| `abundancy_gap_index.csv` | Compact index of all saved abundancy-gap records |
| `abundancy_gap_top.txt` | Detailed report for the 100 smallest positive gaps |
| `abundancy_sigma_maps.json` | Shared validated sigma factorizations referenced by those states |
| `abundancy_gap_summary.json` | Capture configuration, counts, limits, errors, and output status |

`structure.*` and `performance.*` deliberately serve different questions.
Structural data describes the explored mathematical tree. Performance data
describes how the current machine executed it. Cache hits, plan builds, RSS,
and timing may differ while the mathematical structure remains identical.

The q=3 fields follow the same separation:

- `performance.json:q3_prepool_prunes` and
  `q3_prepool_prunes_by_exp` count contradictions detected specifically by the
  LTE prepool execution path;
- the `q3_prepool_shadow_*` fields report shadow verification only;
- `structure.json:valuation_contradictions_by_exponent[].q3_total` counts all
  recorded q=3 valuation contradictions, regardless of the execution path
  that found them.

Possible terminal statuses are:

- `COMPLETE`: the configured finite frontier was exhausted;
- `STOPPED`: a requested stable stop was checkpointed;
- `INTERRUPTED`: execution stopped before a new stable save completed;
- `FAILED`: report finalization occurred during an unhandled failure.

Only `COMPLETE` supports a finite-box exhaustion conclusion.

## Files outside the run directory

The repository root may contain:

- `checkpoint_merged.pkl`: authoritative resumable search frontier;
- `solutions_merged.txt`: accumulated human-readable candidates;
- `factor_graph_<n>.dot` and `.json`: dependency graph for a reported candidate;
- `sigma_pool.sqlite3` and SQLite companions: reusable sigma analyses;
- `plan_cache/`: reusable component plans.

The database and plan directory are not part of the exhaustive-search
conclusion. They may be rebuilt without changing the intended search.

## Console and candidate fields

Progress output includes:

- `States`: completed search states;
- `Rate`: average completed states per second;
- `|f|`: distinct primes assigned in the displayed state;
- `ratio`: a decimal display of the abundancy index stored internally as an
  exact rational number;
- `reson`: structural sigma-factor reuse telemetry.

Resonance has zero priority weight in the checked-in configuration. It does
not influence reachability, pruning, or the mathematical status of a
candidate.

For a reported OPN candidate, `Euler` names the selected Euler prime and
`verified` is the result of recomputing the complete target identity directly
from all assigned prime powers. `req` and `cur` beside a factor display the
incoming sigma-side valuation and assigned exponent recorded by the state. For
a Descartes-type candidate, `r (derived)` is the pseudo-prime factor treated
as if its divisor sum were $r+1$. The equation alone does not certify that
`r` is composite.

Each factor-graph edge

```text
p -> q
```

means that the odd prime $q$ divides $\sigma(p^a)$ for the assigned exponent
$a$. The DOT file is intended for graph visualization. The JSON file contains
`edges`, detected `cycles`, the assigned exponent map, and the Euler prime.
Cycles are reported as structural observations; their presence alone is not a
proof that a candidate is a spoof or an odd perfect number.

## Productive partial states with small positive abundancy gap

For an assigned partial factorization

$$
S=\prod_{p^a\parallel S}p^a,
$$

the abundancy index is

$$
I(S)=\frac{\sigma(S)}{S}
    =\prod_{p^a\parallel S}\frac{\sigma(p^a)}{p^a}.
$$

In a true-OPN factor-chain run, the default capture interval is

$$
0<2-I(S)\le 10^{-2}.
$$

All comparisons use the exact numerator and denominator. The decimal values
in the CSV and text report are displays only. A record is written after the
new prime power has passed the configured checks, its complete in-window
sigma factorization has been propagated, and the resulting state has been
counted as productive. Before writing, the recorder applies the same
mandatory-pending lower bound used by the search:

$$
L(S)=I(S)\prod_{q\in U(S)}\frac{q+1}{q},
$$

where $U(S)$ is the set of unassigned odd primes already required by the
factor chain. If $L(S)>2$, the state is not written: every completion must
contain every $q\in U(S)$ to a positive exponent, and therefore has
abundancy index greater than 2. This reporting filter does not prune or
otherwise modify the search. The summary separately reports the number of
small-gap states rejected by this lower bound.

These records are productive **partial states**. They are not solutions,
complete odd-perfect-number candidates, or evidence that a family of states
must converge to abundancy index 2. In particular, a small positive value of
$2-I(S)$ does not show that the remaining factor-chain and valuation
conditions can be satisfied.

For every odd prime $q$ occurring in the assigned factorization or in the
propagated sigma factorizations, the detailed report displays

$$
\Delta_q(S)=
v_q\!\left(\prod_{p^a\parallel S}\sigma(p^a)\right)-v_q(S).
$$

The sign has the following limited, current-state interpretation:

- $\Delta_q(S)=0$: the two valuations are equal in the current partial state;
- $\Delta_q(S)>0$: the $S$ side must acquire additional $q$-adic valuation;
- $\Delta_q(S)<0$: sigma factors of future assigned prime powers must
  contribute at least $-\Delta_q(S)$.

The comparison is not a completion test because adding another prime power
changes both the factorization and its sigma product.

`abundancy_gap_states.jsonl` is the authoritative capture output. It retains
assignment order, the Euler component, unassigned required prime factors,
the propagated and assigned valuations, exact ratio and gap, search depth,
prime-pool cursor, and the number of explicitly excluded primes. It does not
store a complete excluded-prime set, search priority, parent-child graph, or
a claimed later failure reason.

The shared entries in `abundancy_sigma_maps.json` are read from exact records
in the persistent sigma database. Every row is checked using the database
checksum, prime validation, and

$$
\operatorname{oddpart}(\sigma(p^a))
  =\prod_{\substack{q\ \mathrm{odd}\\q\ \mathrm{prime}}}
     q^{v_q(\sigma(p^a))}.
$$

If the database is unavailable, the state records remain valid and the
factorization report marks the corresponding sigma maps as unavailable. The
reporting path never starts a new factorization.

Capture is capped at 50,000 saved records by default. If that limit is
reached, the summary sets `truncated` and reports how many qualifying states
were not written. This limit affects only observability output.

## Checkpoint semantics

The current executable accepts checkpoint format 4. A checkpoint records:

- run identity and elapsed offset;
- search target and Euler rules;
- prime-pool limit, length, integer width, and boundary values;
- factor and exponent limits;
- frontier entries and heap counter;
- completed-state counters;
- solutions and accumulated metrics.

An initial stable checkpoint is published when the search frontier is created,
before ordinary state processing begins. Periodic, stop-requested, and
solution-boundary saves then replace it atomically.

Loading validates the format, search mode, propagation strategy, prime
metadata, heap structure, counters, and state consistency. The prime pool is
regenerated and checked against the saved length, width, first value, and last
value before search resumes. A failed validation leaves the file unchanged and
refuses the resume.

The checkpoint also retains the set of `(p, exponent)` sigma analyses already
counted in structural telemetry. This set prevents a validated database result
from being counted twice after a restart. It is not mathematical state, is not
emitted in reports, and never controls a prune or propagation decision.

Checkpoint files use Python `pickle`. Load only checkpoints created locally
from a trusted repository state. See [Security Policy](../SECURITY.md).

## Stopping and recovery

For a planned pause:

1. press `Ctrl+C` once;
2. wait for the current state to reach a stable boundary;
3. confirm the `STOPPED` message and saved frontier count;
4. retain `checkpoint_merged.pkl`;
5. restart with `python opn_main.py`.

If a second interrupt is necessary, the most recent fully written checkpoint
remains recoverable because saves use write, flush, `fsync`, and atomic replace.
Work performed after that checkpoint may be repeated.

The abundancy JSONL stream is flushed at every stable checkpoint. Each record
contains the cumulative productive-state ordinal. On recovery, records whose
ordinal is later than the authoritative checkpoint are removed before the
repeated work starts, preventing duplicate output. Only an incomplete final
JSONL line is repaired automatically. Damage to a complete interior line is
reported and the file is preserved unchanged.

When a run reaches `COMPLETE`, the executable removes the checkpoint. Reports
and solutions remain.

## Starting a different experiment

A saved checkpoint takes precedence over the checked-in box. Before starting a
new experiment:

1. stop the current process;
2. move the checkpoint to a labeled archival location or delete it;
3. edit `opn_core.py`;
4. decide whether the new run should reuse or isolate derived caches.

Do not modify a checkpoint by hand.

## Cache recovery

An invalid sigma row is treated as a database miss. An invalid plan entry is
ignored and rebuilt or replaced by an in-memory plan. These fallbacks can
increase runtime but must not change the mathematical result.

If manual cache cleanup is needed, stop the process first. Remove the SQLite
main file, `-wal`, and `-shm` companions as a unit. A plan cache directory may
be removed as a unit. Neither operation restores a search frontier.

## Comparing runs

Before comparing structure or performance, compare `manifest.json`:

- `max_prime`, `max_factors`, and `max_exp`;
- target abundancy index and Euler requirement;
- propagation mode;
- pruning policy;
- pending selection;
- Git commit and dirty-worktree flag.

For mathematical regression, compare `structure.json` fields and solution
sets. Do not use wall time, cache hit counts, or GCD workload as a substitute
for structural equality.

For performance regression:

- use the same hardware and power policy;
- isolate each process;
- state whether caches are cold, warm, or expansion-compatible;
- use multiple alternating runs and report medians and dispersion;
- compare sampled peak RSS and plan-disk use as well as elapsed time.
