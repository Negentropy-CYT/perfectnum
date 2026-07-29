# Architecture

The engine is a single-process search with a compact prime pool, persistent
derived caches, exact state propagation, atomic checkpoints, and separate
structural and performance reporting.

## Data flow

```text
configuration
    |
complete odd-prime pool
    |
search frontier ---- checkpoint
    |
candidate p^a
    |
exact q=3 LTE prepool check
    |
memory sigma cache
    |
validated SQLite lookup
    |
exact cyclotomic components Phi_d(p)
    |
component-filtered hierarchical GCD plans
    |
valuations + complete residual
    |
factor-chain propagation and sound pruning
    |
solutions / structure reports / performance reports
```

The search state never treats a database row or plan file as an axiom. Both
persistent stores are validated derived representations; a failed validation
returns the execution to a mathematically equivalent cold path.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `opn_main.py` | Process lifecycle, prime generation, signal handling, sampling, report orchestration |
| `opn_core.py` | Configuration, arithmetic helpers, sigma analysis, GCD plans, bounds |
| `opn_search.py` | Frontier management, state expansion, propagation, pruning |
| `opn_state.py` | Search-state representations and assignment operations |
| `opn_metrics.py` | Typed mathematical and engineering counters |
| `opn_io.py` | Atomic checkpoints, solutions, factor graphs |
| `opn_reports.py` | Human- and machine-readable run reports |
| `opn_runtime.py` | Background process and progress sampler |
| `opn_sigma_db.py` | SQLite persistence and sigma-record validation |
| `opn_plan_cache.py` | Atomic persistent storage for filtered plans |

Dependencies flow from process orchestration toward arithmetic and state
models. The cache modules do not control reachability or pruning decisions.

## Prime pool

`generate_odd_primes()` produces the complete ordered odd-prime sequence from
3 through the configured limit. Values use `array('I')` while the limit fits
32 bits and `array('Q')` for larger windows. Internally generated pools are
trusted sieve output. External test inputs receive order, range, and parity
validation.

The pool is immutable during a search and is shared by all plans. NumPy views
provide chunked filtering and logarithmic boundary lookup without copying the
entire pool.

## Sigma analysis

For a requested `(p, exponent)`, the analyzer consults:

1. its process-local result cache;
2. the validated SQLite database;
3. a cold component scan on a miss.

Let $n=\mathrm{exponent}+1$. The exact identity

```math
\sigma\!\left(p^{\mathrm{exponent}}\right)
=\prod_{\substack{d\mid n\\d>1}}\Phi_d(p)
```

splits the work into cyclotomic components. Each component of order `d` scans
the necessary prime set

```math
q\mid d
\qquad\text{or}\qquad
q\equiv1\pmod d.
```

The filter is allowed to contain false positives but cannot omit an actual
odd prime divisor. Each component is scanned completely, valuations are added,
and component residuals are multiplied. The final result is either:

- exact: a complete odd-prime valuation map and residual 1; or
- outside-window: an in-window partial map and the complete remaining cofactor.

Only exact maps may drive factor-chain propagation.

Before this analyzer is called, the Euler-form OPN assignment path computes the
exact $v_3\!\left(\sigma(p^{\mathrm{exponent}})\right)$ by LTE. A resulting
valuation contradiction can
be rejected without touching the sigma database or any plan. This fast path
uses the same valuation ledger and contradiction categories as the complete
analysis; it does not replace analysis for branches that remain possible.

## Hierarchical GCD plans

A component plan groups eligible primes into leaf ranges and superblocks.
Production plans keep:

- a filtered read-only prime array;
- exact resident `mpz` products for superblocks;
- leaf boundaries and geometry.

They do not permanently retain every leaf product. After a superblock GCD is
positive, the required leaf products are rebuilt from the immutable prime
array and released after use. The superblock GCD result is reused as the
smaller left operand for leaf localization.

This representation lowers resident memory without changing the set of primes
tested or the final valuation map.

## Plan lifecycle

Plan construction can be eager, database-miss-driven, or adaptive. The
production `adaptive` policy balances two cases:

- a warm database may answer every request without constructing a plan;
- a cold search eventually benefits from building the remaining reusable
  plans in a batch.

The in-process plan cache owns resident products. Persistent plan files contain
filtered arrays and framed products with compatibility metadata and checksums.
Publication uses a staging directory, file synchronization, and atomic rename.

## Search state and propagation

`PROPAGATE=True` uses a best-first heap of factor-chain states. Mandatory odd
factors from an exact sigma map are inserted into a FIFO pending queue.
Incoming q-adic valuations are additive across every processed prime power and
are checked against selected exponents and the target abundancy index.

After cheaper state bounds pass, an exact sigma valuation map is populated
lazily and cached. The same map is used first for a pre-clone valuation
contradiction check and then, if the assignment survives, for mandatory
factor-chain propagation. Detecting the contradiction before cloning changes
allocation cost only; it applies the same valuation rule as the post-clone
path.

`PROPAGATE=False` uses the independent-prime DFS state for restricted
Descartes-type candidates. It does not perform Euler-form OPN factor-chain
propagation.

Priorities affect exploration order only. The engine does not discard live
states to enforce a heap-memory budget.

## Pruning and observability

Mathematical prune reasons and engineering execution mechanisms are recorded
separately:

- a prune reason describes why a state is impossible;
- a mechanism identifies where the contradiction was detected.

Performance telemetry does not influence reachability. Optional `shadow`
controls measure a proven check without enforcing it. The mathematical
conditions for active prunes are specified in
[Mathematical Correctness](../MATHEMATICAL_CORRECTNESS.md).

For prime 3, `ORDER_LTE_PRECHECK` identifies the early execution mechanism.
Touchard residue checks and the q=3 valuation precheck are recorded separately:
the former constrains `N`, while the latter constrains sigma-side valuation
obligations.

## Persistence boundaries

The three persistent mechanisms have different roles:

| Mechanism | Authoritative state | Required to resume | Safe to rebuild |
|---|---:|---:|---:|
| Checkpoint | Yes | Yes | No |
| Sigma database | No | No | Yes |
| Plan cache | No | No | Yes |

The checkpoint is atomically replaced only at a stable boundary. Sigma writes
are flushed at stable boundaries. Plan entries become visible only after
complete atomic publication.
