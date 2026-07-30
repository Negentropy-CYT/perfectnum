# perfectnum

`perfectnum` is a finite-box constraint-propagation search engine for odd
perfect numbers and Descartes-type spoof candidates. It combines Euler-form
exponent rules, exact factor-chain propagation, additive prime valuations,
finite-window sigma analysis, and sound necessary-condition pruning.

All currently known perfect numbers are even. Whether an odd perfect number
exists remains an open problem.

For a positive integer $n$, let

```math
\sigma(n)=\sum_{d\mid n} d,
\qquad
I(n)=\frac{\sigma(n)}{n}.
```

The function `I` is the abundancy index. A perfect number satisfies
$I(N)=2$, equivalently $\sigma(N)=2N$.

The search is computational evidence inside a configured finite box. Exhausting
that box does **not** prove that odd perfect numbers do not exist.

## Core mathematical model

Euler proved that an odd perfect number must have the form

```math
N=q^\alpha m^2,
\qquad
\gcd(q,m)=1,
\qquad
q\equiv\alpha\equiv1\pmod 4,
```

where `q` is prime. It is commonly called the special prime or Euler prime.
Equivalently, $\alpha=4k+1$ and every other prime exponent is even.

Write $p^a\parallel N$ when $p^a$ is the exact prime-power component of $N$.
The sigma chain, also called a factor chain, follows from multiplicativity:

```math
p^a\parallel N
\quad\Longrightarrow\quad
\sigma(p^a)\mid\sigma(N)=2N
\quad\Longrightarrow\quad
\text{every odd prime divisor of }\sigma(p^a)\text{ divides }N.
```

The engine therefore tracks both assigned exponents in $N$ and the additive
$q$-adic valuations forced by processed sigma factors, where $v_q(x)$ is the
exponent of $q$ in $x$. A branch is impossible when those obligations cannot
be paid inside the exponent and factor limits.

In the default factor-chain mode, the target is an Euler-form OPN candidate.
`PROPAGATE=False` instead explores a restricted positive Descartes-type class:
identities that satisfy the formal perfect-number equation when one derived
factor is treated as a prime. These are research comparison objects, not
established odd perfect numbers.

More precisely, if the selected prime-power components are $p_i^{a_i}$, the
search derives a pseudo-prime factor $r$ satisfying

```math
(r+1)\prod_i\sigma\!\left(p_i^{a_i}\right)
=2r\prod_i p_i^{a_i},
```

as though $\sigma(r)=r+1$. The search equation itself does not assert that
`r` is composite; compositeness is what distinguishes a classical spoof from
an actual prime-factor candidate. Descartes' classical example is

```math
3^2\cdot7^2\cdot11^2\cdot13^2\cdot22021,
\qquad
22021=19^2\cdot61,
```

where $22021$ is treated as if it were prime.

## Safe quick start

Requirements:

- Python 3.10 or newer;
- `gmpy2`;
- NumPy;
- `psutil`.

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Before the first run, edit the configuration block at the top of
`opn_core.py`. A short demonstration configuration is:

```python
MAX_PRIME = 100_000
MAX_FACTORS = 12
MAX_EXP = 6
```

Then run:

```bash
python opn_main.py
```

The checked-in configuration is a large search:

```python
MAX_PRIME = 20_000_000_000
MAX_FACTORS = 61
MAX_EXP = 35
```

Do not start it as a demonstration. Large windows require substantial memory,
disk space, and uninterrupted runtime. See
[Running the search](docs/RUNNING.md) first.

## What the bounds mean

In Euler-form OPN mode, the three primary limits define:

- `MAX_PRIME`: largest permitted odd prime divisor of `N`;
- `MAX_FACTORS`: maximum value of $\omega(N)$, the number of distinct prime
  divisors;
- `MAX_EXP`: maximum permitted exponent.

An outside-window sigma factor is a contradiction only within this box. A
complete run establishes that no candidate satisfying the active search rules
exists inside the recorded bounds; it makes no claim outside them.

`PROPAGATE=True` selects the Euler-form factor-chain search. Setting
`PROPAGATE=False` selects the independent-prime DFS used for Descartes-type
spoof exploration. In spoof mode the limits apply to the explicitly selected
prime bases; the derived `r` is not drawn from the prime pool and is not
counted by `MAX_FACTORS`. The run manifest records the active mode and bounds.

The mathematical scope and principal sound pruning conditions are documented
in [Mathematical Correctness](MATHEMATICAL_CORRECTNESS.md).

## Default mathematical constraints

The checked-in OPN policy applies the following necessary conditions:

| Constraint | Role |
|---|---|
| q=3 LTE prepool check | Uses the lifting-the-exponent lemma to compute the exact `v_3(sigma(p^a))` and rejects an impossible 3-adic obligation before a pool scan |
| Touchard congruence | Constrains whether 3 must divide `N` from the Euler prime and `N mod 12/36` |
| Sigma outside-window certificate | Rejects a branch whose mandatory odd sigma factor lies beyond `MAX_PRIME` |
| Additive valuation ledger | Rejects excluded, overrun, or exponent-budget obligations |
| Factor-slot ratio bounds | Proves that the remaining prime slots cannot reach the target abundancy index |
| Maximum-prime capacity | Bounds the exponent when a candidate is guaranteed to be the largest prime factor |

The q=3 prepool check is enabled by `Q3_PREPOOL_MODE="enforce"`. The optional
Fermat-debt bound and domain-ratio bound are disabled in the checked-in
configuration. Configuration and shadow-mode semantics are described in
[Running the search](docs/RUNNING.md); derivations, stated soundness
dependencies, and applicability conditions are in
[Mathematical Correctness](MATHEMATICAL_CORRECTNESS.md).

## Current search pipeline

At a high level, the engine:

1. generates the complete odd-prime pool through `MAX_PRIME`;
2. applies the exact q=3 LTE valuation check before pool work;
3. computes exact cyclotomic components of each required $\sigma(p^a)$;
4. uses necessary residue/order filters and hierarchical GCD scans to find all
   in-window factors;
5. classifies the remaining cofactor as exact or outside the finite window;
6. propagates complete valuation maps through the factor chain;
7. applies conservative arithmetic, ratio, exponent, and congruence bounds;
8. records mathematical structure separately from engineering performance.

The SQLite sigma database and persistent plan directory contain validated,
rebuildable derived data. Neither is a checkpoint or a source of mathematical
truth. Invalid or incompatible entries are treated as misses.

Runtime and memory are platform- and cache-dependent. Performance comparisons
must identify the hardware and distinguish cold, warm, and expansion runs;
the generated performance reports provide the measurements for each run.

## Runtime files

Every run creates `runs/<run_id>/` containing:

- `summary.txt`: compact run index and outcome;
- `manifest.json`: exact configuration and source identity;
- `structure.txt` and `structure.json`: mathematical search structure;
- `performance.txt` and `performance.json`: timing, cache, GCD, and memory data;
- `performance_samples.csv`: sampled progress and process resources.

True-OPN factor-chain runs also record productive partial states satisfying
$0 < 2-\sigma(S)/S \le 10^{-2}$ in a streamed JSONL file, with a compact CSV
index and a detailed text report for the 100 smallest positive gaps. These
records are observability output and do not participate in pruning.

Long-running state is saved atomically in `checkpoint_merged.pkl`. One
`Ctrl+C` requests a stable stop and checkpoint; a second interrupt stops
immediately and retains the most recent complete checkpoint when one exists.

See [Outputs and Recovery](docs/OUTPUTS_AND_RECOVERY.md) before moving,
deleting, comparing, or resuming run artifacts. Checkpoints are trusted local
pickle files; the security implications are described in
[Security Policy](SECURITY.md).

## Project layout

```text
opn_main.py        process lifecycle, signals, sampling, and reports
opn_core.py        configuration, arithmetic, prime pool, sigma analysis
opn_search.py      DFS / best-first search and factor-chain propagation
opn_state.py       search states and assignment operations
opn_metrics.py     structural and performance data models
opn_io.py          checkpoints, solutions, and factor-graph export
opn_reports.py     text, JSON, and CSV report writers
opn_runtime.py     background process-resource sampler
opn_abundancy_capture.py  small positive abundancy-gap state capture
opn_sigma_db.py    validated persistent sigma-analysis cache
opn_plan_cache.py  validated persistent hierarchical-plan cache
test_opn.py        mathematical, recovery, cache, and search regression tests
legacy/            archived reference implementations
```

Further documentation:

- [Running the search](docs/RUNNING.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Outputs and Recovery](docs/OUTPUTS_AND_RECOVERY.md)
- [Validation](docs/VALIDATION.md)
- [Mathematical Correctness](MATHEMATICAL_CORRECTNESS.md)
- [Security Policy](SECURITY.md)

## Tests

Install the test dependency and run:

```bash
python -m pip install "pytest>=7"
python -m pytest -q
```

The repository CI runs the non-slow suite on Python 3.10, 3.11, and 3.12.
Correctness tests include independent arithmetic oracles, sigma-component
identities, filter completeness, cache corruption fallback, checkpoint
continuity, and deterministic search-structure comparisons.

## References

- Descartes, R. (1638). Letter to Mersenne. The example
  `3^2 * 7^2 * 11^2 * 13^2 * 22021` is reproduced in *Oeuvres de Descartes*,
  Vol. II.
- BYU Computational Number Theory Group (2022). *Odd, spoof perfect
  factorizations*. J. Number Theory 234, 31–47.
  [doi:10.1016/j.jnt.2021.07.028](https://doi.org/10.1016/j.jnt.2021.07.028)
- Nielsen, P. P. (2007). *Odd perfect numbers have at least nine distinct
  prime factors*. Math. Comp. 76, 2109–2126.
  [doi:10.1090/S0025-5718-07-01990-4](https://doi.org/10.1090/S0025-5718-07-01990-4)
- Nielsen, P. P. (2015). *Odd perfect numbers, Diophantine equations, and
  upper bounds*. Math. Comp. 84, 2549–2567.
  [doi:10.1090/S0025-5718-2015-02941-X](https://doi.org/10.1090/S0025-5718-2015-02941-X)
- Ochem, P. and Rao, M. (2012). *Odd perfect numbers are greater than
  10^1500*. Math. Comp. 81, 1869–1877.
  [doi:10.1090/S0025-5718-2012-02563-4](https://doi.org/10.1090/S0025-5718-2012-02563-4)
- Thackeray, H. R. (2024). *Each friend of 10 has at least 10 nonidentical
  prime factors*. Indag. Math. 35(3), 595–607.
  [doi:10.1016/j.indag.2024.04.011](https://doi.org/10.1016/j.indag.2024.04.011);
  [arXiv:2310.15900](https://arxiv.org/abs/2310.15900)
- Touchard, J. (1953). *On prime numbers and perfect numbers*. Scripta Math.
  19, 35–39.

## License and author

MIT License. See [LICENSE](LICENSE).

Chengyuan Tang · chengyuantang37@gmail.com

This project was developed with AI-assisted tooling (Codex, Claude Code).
All algorithms, mathematical derivations, and code architecture were reviewed
and validated by the author.
