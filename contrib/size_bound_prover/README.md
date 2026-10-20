# Size-bound prover: "no odd perfect number below B"

A small, self-contained companion to the window search in this repository.
Where `opn_main.py` exhausts a **window** (all prime factors ≤ P, at most F
distinct primes, exponents ≤ E) and must reject any branch whose σ-chain
leaves the window, this tool exhausts a **size bound**: it proves, by
exhaustive branching, that no odd perfect number N ≤ B exists, with no
restriction on how large the primes or exponents may be. A branch that
produces a huge prime is not a loss here; it is the normal way a branch
closes, because the forced part of N jumps past B.

The method is the factor-chain search of Brent, Cohen and te Riele
(*Math. Comp.* 57, 1991), the same family of arguments behind the published
record N > 10^1500 (Ochem and Rao, *Math. Comp.* 81, 2012). It is a
re-implementation written from the papers' ideas, not a port of any
published code (none appears to be public).

## What it proves and how

Search state: `S` (primes with fixed exponent, `p^a || N`), `U` (primes known
to divide N, exponent open), `lo` (every prime of N outside `S ∪ U` is ≥ lo),
the Euler prime once fixed, the exact abundancy of `S`, and for every prime
the valuation already forced by the processed σ-values.

Each node either branches on the exponent of `max(U)` or, when `U` is empty,
on the smallest prime of the remaining cofactor `M = N / ∏S`, whose abundancy
must equal `2 / A_S` exactly. Both branchings are finite for fixed B:

- exponents stop as soon as the forced part exceeds B, and before any
  factoring happens the inequality `N ≥ K·σ(p^a)/gcd(σ(p^a), 2K)` (because
  `σ(p^a) | 2N`) already closes most of them, so the numbers that do need
  factoring have about half the digits of B;
- the smallest new prime is bounded because `ω(M)` is bounded by B and
  `∏ r/(r−1)` over the primes of M has to exceed `2 / A_S`.

σ(p^a) is split into cyclotomic values `Φ_d(p)`, `d | a+1`, which are
factored with trial division, bounded Pollard p−1 / rho / ECM (sympy), a
lookup of the Cunningham-project tables on factordb.com, and finally trial
division by candidates ≡ 1 (mod d). **Every database answer is verified
locally** (the product must equal the number and every claimed prime must pass
`isprime`); the database is a hint source, never an authority.

Exponent parity is enforced throughout: a prime that cannot be the Euler
prime and is forced to appear once must appear squared, which is what makes
the forced-size pruning bite at the boundary.

If a composite cofactor resists all of that, the branch is still closed
rigorously when `K · c' > B`, where `c'` is the cofactor stripped of primes
already known (its remaining primes are new and divide N at least as often as
they divide `c'`); when the Euler prime is already fixed and `c'` is not a
square one more factor `> L` is available. Anything that survives is reported
as an **unresolved roadblock** and the proof is explicitly incomplete for
that bound. A run only prints `PROVED` when there are none.

## Correctness check

The same code with `--mode all` treats 2 like any other prime and must then
find exactly the even perfect numbers. It does: below 10^40 it returns

```
6, 28, 496, 8128, 33550336, 8589869056, 137438691328,
2305843008139952128, 2658455991569831744654692615953842176
```

with their factorisations, no false positives, in well under a second. This
exercises every rule (σ-chain propagation, forced valuations, abundancy
bounds, the ω bound, the smallest-prime branching) on a case with a known
answer.

## Results on one desktop (Ryzen 9 9950X3D, 32 threads, Python 3.13)

| Bound B | Mode | Nodes | Wall time | Unresolved roadblocks | Result |
|--------:|------|------:|----------:|---:|--------|
| 10^20 | serial | 157 | < 1 s | 0 | PROVED |
| 10^30 | serial | 326 | < 1 s | 0 | PROVED |
| 10^50 | serial | 322 | 0.2 s | 0 | PROVED |
| 10^100 | serial | ~1,500 | 45 s (32 factordb lookups) | 0 | PROVED |

A 10^200 run with the parallel driver was started (its 195 level-2
cyclotomic values all came back fully factored from the database, verified
locally) but was stopped before completion, so nothing above 10^100 is
claimed here. Only runs that end with `PROVED` and zero unresolved
roadblocks count.

For comparison, the window search report in this repository
(`P80B_F200_E50_report.md`) retained 35.6 million nodes and rejected 452
million branches, 90.9 % of them only because a σ-factor left the window.
Under a size bound those branches close for free, which is why a few
thousand nodes suffice: the two searches answer different questions, and
this one is the one the literature's lower bounds are stated in.

None of this is new mathematics. The published record is 10^1500; reaching
it with this code would need far better factoring (the boundary cases at
10^1500 involve 750-digit cyclotomic values) and the additional ideas of
Ochem and Rao. A lower-bound search of any size cannot settle existence.

## Running

```
pip install sympy gmpy2
python size_bound_prover.py --B 1e50 --mode odd --cache cache.json
python size_bound_prover.py --B 1e40 --mode all           # self-test
python size_bound_prover_parallel.py --B 1e200 --mode odd --cache cache.json --workers 30
```

`--effort 1..4` sets the bounded ECM budget per stubborn composite; the
parallel driver raises it automatically while roadblocks remain. The cache
file stores verified factorisations of the cyclotomic values and is reused
across runs; `factor_cache.json` here holds the ones from the runs above.
Set `FDB_ENABLED = False` in the module to run fully offline.

Primality is sympy's `isprime` (deterministic below 3.3·10^24, BPSW above),
so the proofs are as rigorous as BPSW; nothing else is probabilistic.

Provenance: written in October 2026 with Claude Code (Anthropic) on David's
machine, in a session that also re-checked the literature (no published or
preprint bound above 10^1500 was found; the 10^1600 figure appears only on
Ochem's slides). Licensed MIT like the rest of the repository.
