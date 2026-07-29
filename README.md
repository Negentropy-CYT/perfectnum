# perfectnum — Odd Perfect Number Search Engine

High-performance constraint-propagation search engine for odd perfect
numbers and Descartes-type spoof candidates.  Factor-chain propagation,
finite-window proof pruning, Nielsen-interval bounds, and bounded structural
telemetry.

$$N = q^{4k+1} \prod p_i^{2a_i} \qquad\text{(Euler form)}$$
$$\sigma(N) = 2N \qquad\text{(perfect number condition)}$$

---

## Quick Start

> **Note:** The checked-in configuration is **experimental**
> (`MAX_PRIME=9e9`, `MAX_FACTORS=60`, `MAX_EXP=25`).  For a first run,
> set `MAX_PRIME = 100_000`, `MAX_FACTORS = 12`, and `MAX_EXP = 6`
> at the top of `opn_core.py`.  See
> [Configuration](#configuration) for safe defaults.

```bash
python -m pip install -r requirements.txt
python opn_main.py
```

**Requirements:** Python 3.10+, gmpy2, numpy, psutil.

The finite search box is configured at the top of `opn_core.py`.
See [MATHEMATICAL_CORRECTNESS.md](MATHEMATICAL_CORRECTNESS.md) before
interpreting an exhausted search.  Exhaustion proves only that no candidate
exists inside the configured finite box; it is not a proof that odd perfect
numbers do not exist.

---

## Background

### Odd Perfect Numbers

A perfect number *N* satisfies σ(N) = 2N, where σ is the sum-of-divisors
function.  All known perfect numbers are even (Euclid-Euler form).  Whether
an **odd** perfect number exists is a millennial open problem.

Euler proved that any odd perfect number must have the form

$$N = q^{4k+1} \prod_{i} p_i^{2a_i}$$

where $q \equiv 1 \pmod{4}$ is the *special* (Euler) prime — the only
prime factor with odd exponent.  All other exponents are even.

### Descartes Spoofs

If we relax the requirement that the "Euler factor" be a single prime
power and instead allow a composite *r* satisfying

$$(r+1) \prod \sigma\!\left(p_i^{a_i}\right) = 2r \prod p_i^{a_i}$$

we obtain *spoofs* or *Descartes spoofs*.  The smallest known example,
due to Descartes, has

$$N = 3^{2} \cdot 7^{2} \cdot 11^{2} \cdot 13^{2} \cdot 22021 \qquad (r = 22021 = 19^{2} \cdot 61)$$

where 22021 is treated *as if* it were prime — σ(22021) is replaced
by 22021 + 1 in the perfect-number equation.

This program searches for both true OPN (Euler-prime) candidates **and**
Descartes-type Descartes spoofs.

### Factor Chains

A key structural constraint exploited by modern OPN research:

$$p^{a} \mid N \;\Longrightarrow\; \sigma(p^{a}) \mid \sigma(N) = 2N$$
$$\Longrightarrow\; \text{every prime factor of }\sigma(p^{a})\text{ must also divide }2N$$

Since N is odd, prime factors of σ(p^a) (excluding 2) must themselves
appear in N.  This creates **forced chains** — including one prime
forces others.  The search engine propagates these constraints
additively, tracking q-adic valuations.

The engine incorporates **Touchard's theorem** ($N \equiv 1 \pmod{12}$ or
$N \equiv 9 \pmod{36}$) as an O(1) congruence check, pre-clone valuation
contradiction detection using lazily cached $\sigma(p^a)$ factor maps, and
finite-window logical pruning whenever the σ-pool analyser certifies a
mandatory odd factor beyond the configured prime window.  A
**maximum-prime capacity bound**, with a Lean formalisation in the author's
local development tree, constrains the exponent of the largest prime factor via
$B(\mathrm{oddpart}(R-1)) = \frac12\sum_{d\mid u,\,d>1}\varphi(d)^2$.
A typed observability system records prune reasons and execution mechanisms
as orthogonal dimensions, with per-run output written to a timestamped
`runs/` directory (structure, performance, CSV samples, and JSON reports).

---

## Project Structure

```
opn_main.py        Entry point (main loop, SIGINT, checkpoint/resume)
opn_core.py        Arithmetic engine
                     · segmented odd-prime sieve → compact uint32/64 array
                     · chunked NumPy validation for compact prime pools
                     · exponent-specific necessary-order filtering
                     · compact persistent superblocks + on-demand leaf GCD
                     · exact / outside-window σ-pool analysis
                     · general-purpose Brent Pollard-Rho factorisation
                     · factor-slot-aware ratio bounds
                     · max-prime capacity bound + LTE valuation helpers
                     · all user-configurable constants
opn_sigma_db.py    Validated persistent σ-analysis cache
                     · exact and window-partial SQLite records
                     · payload checksums + prime-prefix certificates
opn_plan_cache.py  Validated persistent hierarchical-plan storage
                     - mmap filtered-prime arrays + resident mpz products
                     - atomic transactions, checksums, compatibility keys
opn_metrics.py     Typed observability data models
                     · PruneReason / PruneMechanism / CloneEffect enums
                     · StructureMetrics, PoolPerformance, PerformanceMetrics
                     · RunMetrics — single stats sink, checkpoint serialization
opn_state.py       Search state & constraint propagation
                     · DFSState (8 fields, 2 collections cloned)
                     · ChainState (14 fields, 6 collections cloned)
                     · assign_prime_dfs / assign_prime_chain
                     · pending-queue dedup helpers
opn_search.py      Search engine
                     · search_opn() — generator, polymorphic dispatch
                     · DFS (stack) for spoof mode; best-first (heap) for OPN
                     · Touchard congruence pruning + exact-state deduplication
                     · capacity-bound prune + pending drain
opn_io.py          Display, checkpoint, file I/O
                     · display_solution() + factor-chain trace
                     · atomic pickle checkpoint save/load + validation (v4)
                     · save_solutions_txt, export_factor_graph (DOT + JSON)
opn_reports.py     Report writers
                     · structure.txt / structure.json (mathematical results)
                     · performance.txt / performance.json (engineering metrics)
                     · summary.txt, manifest.json
opn_runtime.py     Background performance sampler
                     · RuntimeSampler — daemon thread, psutil RSS/CPU/rate CSV
```

**Dependency graph** (no cycles):

```
opn_metrics  (standalone)
opn_sigma_db ← sqlite3, gmpy2
opn_core     ← gmpy2, numpy, opn_metrics, opn_sigma_db, opn_plan_cache
opn_state    ← opn_core, opn_metrics
opn_search   ← opn_core, opn_state, opn_metrics
opn_io       ← opn_core, opn_state
opn_reports  ← opn_metrics
opn_runtime  ← psutil
opn_main     ← opn_core, opn_search, opn_io, opn_metrics, opn_reports, opn_runtime
```

### Legacy Reference Implementation

The original single-process DFS searcher (a_i = 1, all exponents fixed to 2)
is preserved under `legacy/` for reproducibility and comparison:

```
legacy/
  main.py        entry point (configuration + main loop)
  core.py        prime generation + trial-division factorisation
  search.py      search_v4_safe() + verify_solution()
  io.py          checkpoint save / load
```

```bash
python legacy/main.py          # runs the original searcher
```

An early factor-chain prototype is also preserved under `legacy/`: `opn_factor_chain.py`.

---

## Comparison: Legacy vs Current Engine

### Search Space

| | Legacy (`legacy/`) | Current (`opn_*.py`) |
|---|---|---|
| **Candidate form** | $N = r \prod p_i^{2}$ | $N = q^{4k+1} \prod p_i^{2a_i}$ |
| **Exponents** | fixed: all $a_i = 1$ | variable, bounded by `MAX_EXP` |
| **Euler prime** | folded into composite $r$ | explicitly tracked ($q \equiv 1 \pmod{4}$, exponent $\equiv 1 \pmod{4}$) |
| **Factor coupling** | none — primes are independent | factor chains propagate via $\sigma(p^{a})$ factorisation |
| **Search strategy** | DFS (stack, fixed order) | DFS for Descartes-spoof; best-first heap for true OPN |
| **Spoofs** | primary output (composite $r$) | found in `propagate=False` mode |

### Performance

| | Legacy | Current (DFS mode) | Current (factor chain) |
|---|---|---|---|
| **Per-state cost** | $O(1)$ — stack push + mpz multiply | $O(1)$ — same core operations | $\sigma$-pool analysis + tiered GCD |
| **Memory** | small stack footprint | stack + shared arithmetic caches | scales with prime pool, live heap, σ cache, and built plans |
| **Time to first Descartes-spoof** (PRIME=397) | ~7 min | ~7 min | N/A (not applicable) |

### Why the Current Engine Is Slower Per State

1. **σ-pool analysis** — each reached $\sigma(p^{a})$ must have all mandatory
   odd factors resolved before factor-chain propagation.  The primary path uses
   memory and persistent caches followed by exponent-filtered hierarchical GCD
   scans; general-purpose Pollard-Rho remains available outside this cold-scan
   path.  The legacy engine does not resolve σ factor chains.
2. **State cloning** — `ChainState` carries 14 fields (6 collections); `clone()`
   deep-copies all of them.  In DFS mode the lightweight `DFSState` (8 fields,
   2 collections) avoids this overhead.  The legacy engine reuses 5-element tuples.
3. **Resonance computation** — computing σ-factor overlap via set intersections on
   every `assign_prime_chain` call adds measurable overhead.  In DFS mode this
   is skipped entirely.
4. **Best-first heap** — `heapq.heappush`/`heappop` are $O(\log h)$ vs
   $O(1)$ stack operations.  DFS mode uses a plain stack.  The proof search
   does not trim live heap states; a future memory budget must stop explicitly
   as unresolved rather than silently changing the search space.

### Why the Current Engine Matters Despite the Slowdown

- **The legacy engine cannot search beyond $a_i = 1$.**  Adding variable exponents
  requires the factor-chain framework — resolving the odd prime factors and
  valuations of $\sigma(p^{a})$ is mandatory to determine which new primes are
  forced into $N$.
- **Resonance telemetry** is computed but its priority weight is currently
  zero — it does not affect search order.
- **The additive q-adic valuation** provides a correctness guarantee that the
  legacy `max()` heuristic lacks: tracking $\sum v_q(\sigma(\cdot))$ against
  $v_q(N)$ enables precise contradiction detection.
- **Touchard congruence pruning** catches impossible branches in O(1) without
  any modulo arithmetic, by tracking prime 3's assigned/excluded status.
- **Exact-state deduplication** (optional) removes only states with identical
  assignments, exclusions, valuations, pending order, ratio, and search index.
  It does not generalise one contradiction to a broader family of states.

### When to Use Which

| Goal | Recommended |
|------|-------------|
| Find known Descartes-type spoofs quickly | Legacy (`legacy/main.py`) |
| Explore a configured finite Euler-form box | Current, `PROPAGATE=True` |
| Verify results against prior work | Legacy (reference implementation) |
| Extend to new exponent ranges or factor-chain depth | Current |

---

## Installation

**Requirements:** Python 3.10+, gmpy2, numpy, psutil

```bash
python -m pip install -r requirements.txt
```

Or:

```bash
python -m pip install "gmpy2>=2.0.0" numpy psutil
```

---

## Usage

### Basic

```bash
python opn_main.py
```

### Configuration

Edit the constants at the top of `opn_core.py`.

**Safe demonstration** (completes in seconds):
```python
MAX_PRIME  = 100_000; MAX_FACTORS = 12; MAX_EXP = 6
```

**Regression-validated** (observed identical search-tree and classification
counters across implementation changes):
```python
MAX_PRIME  = 1_000_000_000; MAX_FACTORS = 60; MAX_EXP = 18
```

**Experimental** (large-pool feasibility only):
```python
MAX_PRIME  = 9_000_000_000  # requires ~4 GiB prime storage
```

The checked-in local configuration is currently:

```python
MAX_PRIME = 9_000_000_000
MAX_FACTORS = 60
MAX_EXP = 25
```

`PROPAGATE=True` enables factor-chain (true-OPN) search;
`PROPAGATE=False` runs Descartes-spoof DFS.

Additional controls:
```python
POOL_GCD_MODE = "hierarchical"  # "flat" or "hierarchical"
POOL_SUPERBLOCK_FANOUT = 16
SIGMA_DATABASE_ENABLED = True
SIGMA_DATABASE_FILE = "sigma_pool.sqlite3"
POOL_PLAN_BUILD_POLICY = "adaptive"  # eager / after_db_miss / adaptive
POOL_PLAN_DISK_CACHE_ENABLED = True
POOL_PLAN_DISK_CACHE_DIR = "plan_cache"
POOL_PLAN_DISK_MIN_FREE_BYTES = 2 * 1024**3
ENABLE_FERMAT_DEBT = False      # conservative debt-capacity bound (off)
CHECKPOINT_INTERVAL_SECONDS = 300.0
PRUNING_POLICY = "baseline-v0"  # identity tag in manifest.json
TELEMETRY_SCHEMA_VERSION = 3    # per-exponent breakdowns active
```

In `hierarchical` mode, plans retain only exact superblock products.  Leaf
products are rebuilt from the immutable eligible-prime array after a positive
superblock GCD and released immediately after the leaf check.  The `flat` mode
continues to retain leaf products as a correctness oracle.  Performance schema
5 reports logical/resident leaf counts, dynamic rebuild work, and persistent
cache health; schema-1/2/3/4 checkpoints remain readable.

With the disk-plan cache enabled, exponent-filtered prime arrays are created
directly as read-only NumPy memory maps under `plan_cache/`. Exact superblock
products are also persisted, but are checksum-verified and fully deserialised
to resident `mpz` values before any GCD scan; the hot mathematical path never
decodes products from disk. A cache key includes the complete prime-pool
digest, limit, source interval, radical filter, dtype, leaf size, fanout,
schema, and semantics version. Consequently a 32-bit 4B plan cannot be
mistaken for a 64-bit 5B plan, and a partial expansion interval cannot collide
with a full-window plan.

Cold creation uses a locked staging directory, `fsync`, checksums, and atomic
rename. An interrupted or corrupt entry is invisible or treated as a miss;
insufficient disk space falls back to an ordinary in-memory plan without
changing the search. Files remain useful across restarts with exactly the same
compatible plan key. Different windows or plan geometry create separate
entries, so `plan_cache/` can grow over a series of experiments and may be
deleted while the program is stopped. It is derived data, not a checkpoint.

Local acceptance measurements produced identical plan digests at 100M, 500M,
and 1B. At 1B, pure-memory/cold-disk/warm-disk plan construction was
39.89/52.06/1.74 seconds, while the sampled construction peak was
1.300/0.833/0.818 GiB. A completed 100M search had identical 48,980-state
trees and prune counters in all modes; three warm runs took
13.53/13.41/13.38 seconds and peaked near 0.132 GiB, versus 16.98 seconds and
0.178 GiB for pure memory. These figures are engineering observations, not
mathematical assumptions or universal hardware guarantees.

The sigma database stores validated exact and window-partial analyses.  Lookup
occurs before any pool plan is built.  Exact records are window-independent;
when the prime window grows, a compatible partial record continues from its
complete residual.  Before a shared full plan exists, a substantial certified
prefix uses a new-interval plan.  Once a shared plan exists, its scan starts at
the leaf containing the first prime above the certified limit, so the old
prefix is not traversed again (at most that one boundary leaf is conservatively
rechecked).  Prefix positions are cached and searched with a scalar of the
array's exact NumPy dtype, preserving logarithmic lookup for both 32-bit and
64-bit prime arrays.  Every row carries a checksum and arithmetic identity
check, while partial rows also require a SHA-256 certificate for the
corresponding prime-pool prefix.  Invalid or incompatible rows are treated as
ordinary cache misses.

`adaptive` is the recommended plan policy.  A warm repeated search can avoid
all plans, while a cold search initially builds plans on demand and switches to
batched prebuilding after several distinct plan misses.  Database commits are
batched and flushed at stable checkpoint boundaries.  Deleting
`sigma_pool.sqlite3` and its `-wal`/`-shm` companions discards only reusable
derived data; it does not affect checkpoints or mathematical results.  Stop
the program before deleting any of these three files.

### Search Modes

| `PROPAGATE` | Strategy | Description |
|:-----------:|----------|-------------|
| `False` | DFS (stack) | Independent primes. Finds Descartes spoofs via composite‑r formula. Fast per-state. |
| `True` | best-first (heap) | Factor-chain propagation with pool-analyser outside-window pruning. Searches for genuine Euler-prime OPN. |

### Interpreting Output

**Progress line** (time-based, approximately once per second):
```
[Progress] States:  884,300,000 | Time:  582.1s | Rate: 1519000/s | |f|=8 ratio=1.8486 reson=-3.42
```
- `|f|` — distinct primes currently assigned
- `ratio` — current $\sigma(N)/N$ (target: $2.0$)
- `reson` — resonance heuristic score (higher = more $\sigma$-factor reuse)

**Descartes spoof:**
```
*** Descartes Spoof  #1 ***
  N              = 198585576189
  r (composite)  = 22021  =  19^2 × 61^1
  r ≡ 1 mod 4    = True
  resonance      = +0.87
  Factors:
    3^2
    7^2
    11^2
    13^2
```

**True OPN candidate** (if found — none known to exist):
```
*** OPN Candidate  #1 ***
  N          = ...
  Euler      = 5
  σ(N)/N     = 2.000000000000
  verified   = True
  Factor chain (from Euler prime 5):
    σ(5^1) = 6 = 2 × 3
    σ(3^2) = 13 = 13
    ...
```

### Factor Graph

When a solution is found, the σ-factor dependency graph is automatically
exported as two files (numbered by solution index):

- `factor_graph_N.dot` — Graphviz DOT format.
- `factor_graph_N.json` — machine-readable edge list with cycle detection.

Each edge `p → q` means σ(p^a) contains prime factor q, i.e. assigning
p forces q into N.  Cycles in this graph (e.g. 3 → 13 → 3) are the
structural signature of Descartes-type Descartes-spoofs and explain why
the resonance heuristic works.

### Search Telemetry

Output is written to a timestamped `runs/<run_id>/` directory:

| File | Content |
|---|---|
| `summary.txt` | Run identity, configuration, result, report index |
| `manifest.json` | Machine-readable config + `telemetry_schema_version` + `pruning_policy` |
| `structure.txt` / `.json` | Mathematical results: prune reasons, depth histogram, ratio headroom, propagation edges, sigma classifications, contradiction attribution, pending-prime frequency, **sigma classifications by exponent**, **valuation contradictions by source exponent** |
| `performance.txt` / `.json` | Engineering metrics: phase timings, clone payload, pool cache, GCD workload, plan build timing, prune mechanisms, core timings, slowest analyses, memory snapshots, **sigma-pool workload by exponent** |
| `performance_samples.csv` | Time-series: RSS, CPU, rate, frontier size (2 s interval) |

Prune counters are recorded in two orthogonal dimensions:

- **PruneReason** (structure) — mathematical cause: `outside_window`, `valuation_contradiction`, `ratio_overshoot`, `ratio_unreachable`, `factor_slots`, `touchard`, etc.
- **PruneMechanism** (performance) — execution path: `known_outside_cache`, `cold_pool_certificate`, `preclone_valuation`, `tail_ratio_bound`, `prospective_ratio`, etc.

This separation keeps the mathematical report deterministic across
hardware while surfacing cache and implementation bottlenecks in the
performance report.  Per-exponent breakdowns are pre-allocated via
`metrics.configure_exponent_telemetry(max_exp)` at search start and
persist through checkpoint cycles; missing fields in old checkpoints
are filled with zero defaults.

### Checkpoint / Resume

The engine writes `checkpoint_merged.pkl` at startup and then every
`CHECKPOINT_INTERVAL_SECONDS` at a stable boundary between states.
The active heap/stack is serialised synchronously at that boundary, so
the hot search loop no longer copies and heapifies the full frontier
after every processed state.

Press `Ctrl+C` once to request a cooperative stop.  The current state is
finished, the exact remaining frontier and telemetry counters are
atomically saved, and the program exits.  Press `Ctrl+C` a second time
to stop immediately; in that case the last completed periodic
checkpoint is retained and at most one checkpoint interval of work is
replayed on resume.  A forced interruption never overwrites a valid
checkpoint with a partially processed frontier.

Running `python opn_main.py` again resumes from the saved frontier with
cumulative statistics.  Checkpoint writes use a temporary file,
`flush`/`fsync`, and atomic replacement, so an interrupted write leaves
the previous checkpoint intact.

On resume the checkpoint is validated for internal consistency
(pending/pending_set agreement, non-negative valuations, heap counter
coherence, mode fingerprint, format version).  Issues cause the
checkpoint to be rejected (not silently continued).

Selected search-state and telemetry counters are persisted via
`RunMetrics.checkpoint_payload()`.  The checkpoint stores a prime
*fingerprint* (limit, count, typecode, first, last) rather than the
full array; on resume the prime pool is regenerated from the sieve
and validated against all five fingerprint fields.  A mismatch raises
`RuntimeError` — the search will not silently continue with an
incompatible prime pool.

The checkpoint also retains the set of `(p, exponent)` sigma analyses already
counted in structural telemetry.  This set is not part of the mathematical
state and is not emitted in reports; it only prevents a database-loaded result
from being counted twice after a restart.  Search decisions depend on the
validated sigma result itself, never on this telemetry set.

Delete `checkpoint_merged.pkl` to force a fresh start.

### Comparing Runs

The `structure.json` and `performance.json` files are machine-readable and can
be compared with ordinary JSON tooling.  The repository does not currently
ship a dedicated run-comparison command.

---

## Key Algorithms

### Additive q-adic Valuation

When $p^{a}$ is assigned and $\sigma(p^{a})$ contains factor $q^{e}$, we track

$$\begin{aligned}
\mathrm{req}_v[q] &\gets \mathrm{req}_v[q] + e \qquad &\text{(total $q$-demand from the $\sigma$ side)} \\
\mathrm{cur}_v[q]  &\gets \mathrm{cur}_v[q] + a_q \qquad &\text{($q$'s exponent in $N$)}
\end{aligned}$$

If `required_v[q] > current_v[q]`, the prime q is **forced** into the
pending queue — it must appear in N with sufficient exponent to
satisfy the valuation balance from σ(N) = 2N.

This additive formulation (summing contributions across all assigned
primes) replaces the weaker `max()` heuristic used in earlier versions.

### Resonance Telemetry

The engine computes a resonance diagnostic for structural analysis.
Its priority weight is currently **zero** (`PRIORITY_RESONANCE_W = 0.0`
in `opn_core.py`), so resonance does not affect search order or
completeness in the checked-in configuration.

For each candidate prime *p* with exponent *a*, the σ(p^a) factor set is
compared against the primes already in N:

$$\begin{aligned}
\mathrm{reuse} &= \bigl|\Sigma_{\sigma} \cap N\bigr| \\[2pt]
\mathrm{newf}  &= \bigl|\Sigma_{\sigma} \setminus N\bigr| \\[4pt]
\Delta\mathrm{res} &= 1.5 \cdot \mathrm{reuse} - 0.7 \cdot \mathrm{newf} \\
                   &- 0.15 \cdot \log_{10}(\mathrm{largest} + 1)
\end{aligned}$$

where $\Sigma_{\sigma}$ is the set of odd prime factors of $\sigma(p^{a})$.
This score is recorded for telemetry but does not influence search order.

### Pre-Clone Valuation Check

Before cloning a state in chain mode, an exact cached
$\{q: v_q(\sigma(p^{a}))\}$ map enables valuation contradiction detection
*without* paying the clone cost.  The map is populated only when that
assignment is actually reached.
This eliminates wasted clones — the dominant structural overhead in
factor-chain search where 49% of clones were previously discarded after
post-clone factorisation.

### σ-Pool Outside-Window Pruning

Before a chain assignment is cloned, the σ-pool analyser strips every eligible
prime in the configured complete odd-prime pool.  If the complete residual is
greater than one, at least one mandatory odd factor lies beyond `MAX_PRIME`,
and the branch is rejected.  This applies to every reached exponent rather
than relying on a separately precomputed exponent-4 table.

`EXCLUDE_EXP_4` and its helper functions remain in `opn_core.py` for
compatibility and focused tests, but the current production search does not
populate or consult that specialised table.

### Precise Next-Prime Interval Bounds (Nielsen Prop. 3)

In chain mode, each expansion step computes lower and upper bounds on the
next unknown prime. The lower bound comes from
$R \times (p+1)/p \leq T$. For the upper bound, if only $r$ distinct-prime
slots remain, the best relaxed tail is formed by the $r$ smallest available
primes because $p/(p-1)$ decreases with $p$.

The implementation multiplies at most `MAX_FACTORS` exact rational factors.
Mandatory pending primes are included even when they precede `next_idx`, and
they consume slots before optional primes are selected. The former full-suffix
big-integer arrays are no longer allocated.

### Exact Reverse Valuations and Fermat-Prime Debt

For an outstanding Fermat-prime valuation debt, the engine uses the exact
order/LTE identity for $v_q(\sigma(p^a))$.  It computes the maximum amount
each still-available component can contribute and sums the largest capacities
over the remaining factor slots.  A branch is pruned only if even this relaxed
upper bound cannot pay the debt.  The former high-exponent congruence-count
heuristic was removed because it was not a valid consequence of the cited
Nielsen lemmas.

### Large-Power Thresholds

`is_prime_infinite()` retains the four piecewise thresholds used by the
Mathematica reference as classification metadata.  Factor-chain propagation
is never skipped merely because $p^a$ crosses one of these thresholds:
omitting the odd factors of $\sigma(p^a)$ would lose mandatory obligations.

### Friend-of-10 Verification Mode

The `SearchMode` descriptor (presets `OPN_MODE` / `FRIEND_10_MODE`)
bundles target abundance, Euler requirements, and forced/excluded primes
into a single immutable configuration object.
`SEARCH_MODE = FRIEND_10_MODE` switches to $\sigma(N)/N = 9/5$, forces
$5 \mid N$, excludes $3$, and disables Euler checks.  Reproduces the
Thackeray (2024) result: $\omega \ge 10$ for friends of 10.

---

## Author

**Chengyuan Tang**  ·  chengyuantang37@gmail.com

> This project was developed with AI-assisted tooling (Codex, Claude Code).
> All algorithms, mathematical derivations, and code architecture were
> reviewed and validated by the author.

---

## License

MIT License — see [LICENSE](LICENSE) for full text.

Copyright (c) 2025 Chengyuan Tang

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## References

- Descartes, R. (1638).  Letter to Mersenne (manuscript).  The number
  $198585576189 = 3^{2} \cdot 7^{2} \cdot 11^{2} \cdot 13^{2} \cdot 22021$
  would be perfect if $22021$ were prime.  Reproduced in *Oeuvres de
  Descartes*, Vol. II.
- Nielsen, P. P. (2007).  *Odd perfect numbers have at least nine
  distinct prime factors*.  Math. Comp. 76, 2109–2126.
  doi:[10.1090/S0025-5718-07-01990-4](https://doi.org/10.1090/S0025-5718-07-01990-4)
- Nielsen, P. P. (2015).  *Odd perfect numbers, Diophantine equations,
  and upper bounds*.  Math. Comp. 84, 2549–2567.
  doi:[10.1090/S0025-5718-2015-02941-X](https://doi.org/10.1090/S0025-5718-2015-02941-X)
- Ochem, P. & Rao, M. (2012).  *Odd perfect numbers are greater than
  10^1500*.  Math. Comp. 81, 1869–1877.
  doi:[10.1090/S0025-5718-2012-02563-4](https://doi.org/10.1090/S0025-5718-2012-02563-4)
- Thackeray, H. R. (2024).  *Each friend of 10 has at least 10
  nonidentical prime factors*.  Indag. Math. (N.S.) 35, 595–607.
  arXiv:[2310.15900](https://arxiv.org/abs/2310.15900)
- Touchard, J. (1953).  *On prime numbers and perfect numbers*.
  Scripta Math. 19, 35–39.
