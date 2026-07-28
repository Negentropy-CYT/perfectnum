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
> (`MAX_PRIME=5e9`).  For a first run, set `MAX_PRIME = 100_000`
> and `MAX_FACTORS = 12` at the top of `opn_core.py`.  See
> [Configuration](#configuration-1) for safe defaults.

```bash
python -m pip install -r requirements.txt
python opn_main.py
```

**Requirements:** Python 3.10+, gmpy2, numpy.

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
finite-window logical pruning of exponent-4 branches where $\sigma(p^4)$
has at least one mandatory odd prime factor beyond the window.  A **maximum-prime capacity bound**
(proved in Lean) constrains the exponent of the largest prime factor via
$B(\mathrm{oddpart}(R-1)) = \frac12\sum_{d\mid u,\,d>1}\varphi(d)^2$.
A comprehensive telemetry system
(`telemetry.txt`) records prune reasons, clone economics, depth histograms,
and high-frequency pending-prime patterns across parameter configurations.

---

## Project Structure

```
opn_main.py        Entry point (configuration + main loop)
opn_core.py        Arithmetic engine
                     · segmented odd-prime sieve → compact uint32/64 array
                     · exponent-specific necessary-order filtering
                     · indexed prime blocks + hierarchical GCD screening
                     · exact / outside-window σ-pool analysis
                     · general-purpose Brent Pollard-Rho factorisation
                     · factor-slot-aware ratio bounds
                     · max-prime capacity bound + LTE valuation helpers
                     · telemetry counters + all user-configurable constants
opn_state.py       Search state & constraint propagation
                     · DFSState (8 fields, 2 collections cloned) — lightweight
                       Descartes-spoof DFS
                     · ChainState (14 fields, 6 collections cloned) — full
                       factor-chain search
                     · assign_prime_dfs / assign_prime_chain — separate
                       constraint propagation per mode
                     · pending-queue dedup helpers
opn_search.py      Search engine
                     · search_opn() — generator with polymorphic dispatch
                     · DFS (stack) for Descartes-spoof mode (DFSState)
                     · best-first (heap) for factor-chain mode (ChainState)
                     · Touchard congruence pruning + exact-state deduplication
                     · capacity-bound prune on last-slot expansions + pending drain
                     · true-OPN & Descartes-spoof checks
                     · Descartes-spoofs continue expanding in chain mode
opn_io.py          Display, checkpoint, file I/O
                     · display_solution()
                     · factor-chain trace
                     · atomic pickle checkpoint save/load + validation
                     · human-readable solutions file
                     · write_telemetry_report() — structured Markdown report
                     · display_telemetry_brief() — console summary
                     · clone economics — pre-clone avoidance rate
                     · export_factor_graph() — DOT + JSON σ-dependency graph
```

**Dependency graph** (no cycles):

```
opn_core   ← gmpy2, numpy, math, random
opn_state  ← opn_core
opn_search ← opn_core + opn_state
opn_io     ← opn_core + opn_state
opn_main   ← opn_core + opn_search + opn_io
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
| **Memory** | ~1 MB (stack only) | ~10 MB (stack + caches) | ~50 MB (heap + factor/sigma/power caches) |
| **Time to first Descartes-spoof** (PRIME=397) | ~7 min | ~7 min | N/A (not applicable) |

### Why the Current Engine Is Slower Per State

1. **Brent Pollard-Rho factorisation** — each reached $\sigma(p^{a})$ must be
   fully factorised to propagate factor chains.  Maps are cached lazily after
   cheaper bounds pass; the legacy engine never factorises $\sigma$ values.
2. **State cloning** — `ChainState` carries 14 fields (7 collections); `clone()`
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
  requires the factor-chain framework — factorising $\sigma(p^{a})$ is mandatory
  to determine which new primes are forced into $N$.
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

**Requirements:** Python 3.10+, gmpy2, numpy

```bash
python -m pip install -r requirements.txt
```

Or:

```bash
python -m pip install "gmpy2>=2.0.0" numpy
```

Via conda:

```bash
conda install -c conda-forge gmpy2 numpy
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
MAX_PRIME  = 5_000_000_000  # requires ~2 GiB prime storage
```

`PROPAGATE=True` enables factor-chain (true-OPN) search;
`PROPAGATE=False` runs Descartes-spoof DFS.

Additional controls:
```python
POOL_GCD_MODE = "hierarchical"  # "flat" or "hierarchical"
POOL_SUPERBLOCK_FANOUT = 16
ENABLE_FERMAT_DEBT = False      # conservative debt-capacity bound (off)
CHECKPOINT_INTERVAL_SECONDS = 300.0
```

### Search Modes

| `PROPAGATE` | Strategy | Description |
|:-----------:|----------|-------------|
| `False` | DFS (stack) | Independent primes. Finds Descartes spoofs via composite‑r formula. Fast per-state. |
| `True` | best-first (heap) | Factor-chain propagation with pool-analyser outside-window pruning. Searches for genuine Euler-prime OPN. |

### Interpreting Output

**Progress line** (update interval set by `PROGRESS_INTERVAL` in `opn_core.py`):
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

On completion or `Ctrl+C`, the engine prints three telemetry reports.
Selected telemetry counters are serialised into the checkpoint and
accumulate across interrupt/resume cycles.  Per-run pool timing,
slowest-analysis records, and some diagnostic counters are not
persisted.

**Prune Statistics** — per-reason rejection rates, normalised against
`attempted = actual_clones + avoided_pre_clone` (‰ = per 1000 clones):

```
Prune statistics:
  ratio             916,000+  (98.0% of prunes,  51.5‰ of clones)
  touchard            3,688  ( 2.0% of prunes,   1.0‰ of clones)
```

Each counter is incremented at the exact `return None` site inside
`assign_prime_dfs` / `assign_prime_chain`.  The ‰ rate isolates which
heuristic dominates *per clone attempt*, enabling cross-configuration
comparison.

**Depth Histogram** — productive clone depth (expansion steps). , *not*
number of assigned prime factors.  The bell shape reveals where the
search tree expands and where ratio pruning begins to dominate.

**Clone Effectiveness** — classifies every `clone()` call by outcome:

```
Clone effectiveness:
  attempted branches    1,109,587
  actual clones           144,575
  avoided (pre-clone)     965,012
  avoidance rate            87.0%
    productive             144,570  (100.0% of actual)
```

- **avoided** = prunes that executed *before* `clone()` — these are the
  highest-value heuristics.
- **wasted** = prunes that executed *after* `clone()` — mathematically
  correct but computationally expensive.
- **overhead** = skip branches and initial states — inherent DFS
  enumeration cost.

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

Selected search-state and telemetry counters are persisted.  Per-run
pool timing, slowest-analysis records, and some diagnostic counters
are not guaranteed to survive resume.  The checkpoint also stores the
generated prime pool — for very large `MAX_PRIME` this can produce
large checkpoint files.

Delete `checkpoint_merged.pkl` to force a fresh start.

### Telemetry Report

On completion (or Ctrl+C), a structured Markdown report is written to
`telemetry.txt` covering prune statistics, clone economics, pool
analysis, core timings, depth histograms, and propagation edges.
Counters marked as checkpointed persist across resume cycles.

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

### EXCLUDE_EXP_4 Pruning

If $\sigma(p^{4})$ has any odd prime factor exceeding `MAX_PRIME`, the
$a=4$ include branch is skipped.  This is **window-complete** logical
pruning: every such factor is mandatory and would become an unresolvable
pending obligation.  $a=2$ is never filtered by this specialised check.

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
