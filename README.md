# opn-search — Odd Perfect Number Constraint Engine

A constraint-propagation factor-chain search engine for exploring
structured subspaces of odd perfect numbers and Descartes-type
pseudo-candidates.

$$N = q^{4k+1} \prod p_i^{2a_i} \qquad\text{(Euler form)}$$
$$\sigma(N) = 2N \qquad\text{(perfect number condition)}$$

---

## Quick Start

```bash
pip install gmpy2
python opn_main.py
```

Default configuration searches for **pseudo-OPN candidates** (composite
"Euler factor") with primes ≤ 500, up to 8 distinct factors, exponent 2.
A 12-digit Descartes-type pseudo-candidate is found in ~7 minutes.

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

### Pseudo-OPN (Descartes-type) Candidates

If we relax the requirement that the "Euler factor" be a single prime
power and instead allow a composite *r* satisfying

$$(r+1) \prod \sigma\!\left(p_i^{a_i}\right) = 2r \prod p_i^{a_i}$$

we obtain *spoofs* or *pseudo-candidates*.  The smallest known example,
due to Descartes, has

$$N = 3^{2} \cdot 7^{2} \cdot 11^{2} \cdot 13^{2} \cdot 22021 \qquad (r = 22021 = 19^{2} \cdot 61)$$

where 22021 is treated *as if* it were prime — σ(22021) is replaced
by 22021 + 1 in the perfect-number equation.

This program searches for both true OPN (Euler-prime) candidates **and**
Descartes-type pseudo-candidates.

### Factor Chains

A key structural constraint exploited by modern OPN research:

$$p^{a} \mid N \;\Longrightarrow\; \sigma(p^{a}) \mid \sigma(N) = 2N$$
$$\Longrightarrow\; \text{every prime factor of }\sigma(p^{a})\text{ must also divide }2N$$

Since N is odd, prime factors of σ(p^a) (excluding 2) must themselves
appear in N.  This creates **forced chains** — including one prime
forces others.  The search engine propagates these constraints
additively, tracking q-adic valuations.

---

## Project Structure

```
opn_main.py        Entry point (configuration + main loop)
opn_core.py        Arithmetic engine
                     · prime generation (sieve)
                     · Brent Pollard-Rho factorisation
                     · σ(p^a) computation (cached)
                     · factor / power / sigma caches
                     · σ-factor-set precomputation
                     · ratio upper/lower bounds
                     · all user-configurable constants
opn_state.py       Search state & constraint propagation
                     · State dataclass (slots)
                     · assign_prime() — Euler constraints,
                       resonance score, additive valuation,
                       factor-chain propagation
                     · pending-queue dedup helpers
opn_search.py      Search engine
                     · search_opn() — generator yielding candidates
                     · DFS (stack) for pseudo-solution mode
                     · best-first (heap) for factor-chain mode
                     · pending-prime processing
                     · true-OPN & pseudo-solution checks
opn_io.py          Display, checkpoint, file I/O
                     · display_solution()
                     · factor-chain trace
                     · atomic pickle checkpoint save/load
                     · human-readable solutions file
```

**Dependency graph** (no cycles):

```
opn_core   ← gmpy2, math, random
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

An early factor-chain prototype is also retained at the project root:

| file | description |
|------|-------------|
| `opn_factor_chain.py` | first factor-chain engine prototype |

---

## Comparison: Legacy vs Current Engine

### Search Space

| | Legacy (`legacy/`) | Current (`opn_*.py`) |
|---|---|---|
| **Candidate form** | $N = r \prod p_i^{2}$ | $N = q^{4k+1} \prod p_i^{2a_i}$ |
| **Exponents** | fixed: all $a_i = 1$ | variable: $a_i \in \{2, 4, 6, 8, 10\}$ |
| **Euler prime** | folded into composite $r$ | explicitly tracked ($q \equiv 1 \pmod{4}$, exponent $\equiv 1 \pmod{4}$) |
| **Factor coupling** | none — primes are independent | factor chains propagate via $\sigma(p^{a})$ factorisation |
| **Search strategy** | DFS (stack, fixed order) | DFS for pseudo-solution; best-first heap for true OPN |
| **Pseudo-solutions** | primary output (composite $r$) | found in `propagate=False` mode |

### Performance

| | Legacy | Current (DFS mode) | Current (factor chain) |
|---|---|---|---|
| **Max rate** | ~1.5M states/s | ~350K states/s | ~140K states/s |
| **Per-state cost** | $O(1)$ — stack push + mpz multiply | $O(1)$ — same core operations | $O(\log d)$ — Brent-Rho factorisation of $\sigma(p^{a})$ |
| **Memory** | ~1 MB (stack only) | ~10 MB (stack + caches) | ~50 MB (heap + factor/sigma/power caches) |
| **Time to first pseudo-solution** (PRIME=397) | ~7 min | ~7 min | N/A (not applicable) |

### Why the Current Engine Is Slower Per State

1. **Brent Pollard-Rho factorisation** — each $\sigma(p^{a})$ must be fully
   factorised to propagate factor chains.  The legacy engine never factorises
   $\sigma$ values — it only multiplies them into the running product.
2. **State cloning** — the `State` dataclass carries 10 fields (7 collections);
   `clone()` deep-copies all of them.  The legacy engine reuses 5-element tuples.
3. **Resonance computation** — computing σ-factor overlap via set intersections on
   every `assign_prime` call adds measurable overhead.
4. **Best-first heap** — `heapq.heappush`/`heappop` are $O(\log h)$ vs
   $O(1)$ stack operations.  With `HEAP_MAX_SIZE = 200 000`, each push costs
   ∼18 comparisons.

### Why the Current Engine Matters Despite the Slowdown

- **The legacy engine cannot search beyond $a_i = 1$.**  Adding variable exponents
  requires the factor-chain framework — factorising $\sigma(p^{a})$ is mandatory
  to determine which new primes are forced into $N$.
- **Resonance guidance reduces the state count to the first solution.**  In
  factor-chain mode, the best-first heap explores high-resonance branches
  first, often reaching candidates in thousands of states instead of hundreds
  of millions.
- **The additive q-adic valuation** provides a correctness guarantee that the
  legacy `max()` heuristic lacks: tracking $\sum v_q(\sigma(\cdot))$ against
  $v_q(N)$ enables precise contradiction detection.

### When to Use Which

| Goal | Recommended |
|------|-------------|
| Find known Descartes-type spoofs quickly | Legacy (`legacy/main.py`) |
| Explore the full Euler-form search space | Current, `PROPAGATE=True` |
| Verify results against prior work | Legacy (reference implementation) |
| Extend to new exponent ranges or factor-chain depth | Current |

---

## Installation

**Requirements:** Python 3.10+, [gmpy2](https://github.com/aleaxit/gmpy)

```bash
pip install gmpy2
```

**Windows note:** gmpy2 wheels are available via `pip`.  If you
encounter issues, install from [Christoph Gohlke's page](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gmpy2)
or use `conda install -c conda-forge gmpy2`.

---

## Usage

### Basic

```bash
python opn_main.py
```

### Configuration

Edit the constants at the top of `opn_core.py` (or override them
programmatically):

```python
# opn_core.py

MAX_PRIME   = 500        # largest odd prime considered
MAX_FACTORS = 8          # max distinct prime factors in N
MAX_EXP     = 4          # max exponent for any prime
                         #   2 = a_i=1 restriction
                         #   6+ = variable exponents
PROPAGATE   = False      # False → pseudo-solution search
                         # True  → true-OPN factor-chain search
```

### Search Modes

| `PROPAGATE` | Strategy | Description |
|:-----------:|----------|-------------|
| `False` | DFS (stack) | Independent primes. Finds pseudo-candidates via composite‑r formula. Fast per-state. |
| `True` | best-first (heap) | Factor-chain propagation. Resonance-guided priority queue. Searches for genuine Euler-prime OPN. |

### Interpreting Output

**Progress line** (updates in-place every ~100K states):
```
[Progress] States:  884,300,000 | Time:  582.1s | Rate: 1519000/s | |f|=8 ratio=1.8486 reson=-3.42
```
- `|f|` — distinct primes currently assigned
- `ratio` — current $\sigma(N)/N$ (target: $2.0$)
- `reson` — resonance heuristic score (higher = more $\sigma$-factor reuse)

**Pseudo-candidate:**
```
*** Pseudo-OPN Candidate  #1 ***
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

### Checkpoint / Resume

Press `Ctrl+C` at any time — the search state (including the full
heap/stack) is atomically saved to `checkpoint_merged.pkl`.  Running
`python opn_main.py` again will resume from the interruption point.

Delete `checkpoint_merged.pkl` to force a fresh start.

---

## Key Algorithms

### Additive q-adic Valuation

When $p^{a}$ is assigned and $\sigma(p^{a})$ contains factor $q^{e}$, we track

$$\begin{aligned}
\mathrm{req}_v[q] &\mathrel{+}= e \qquad &\text{(total $q$-demand from the $\sigma$ side)} \\
\mathrm{cur}_v[q]  &\mathrel{+}= a_q \qquad &\text{($q$'s exponent in $N$)}
\end{aligned}$$

If `required_v[q] > current_v[q]`, the prime q is **forced** into the
pending queue — it must appear in N with sufficient exponent to
satisfy the valuation balance from σ(N) = 2N.

This additive formulation (summing contributions across all assigned
primes) replaces the weaker `max()` heuristic used in earlier versions.

### Resonance-Guided Best-First Search

For each candidate prime *p* with exponent *a*, the σ(p^a) factor set is
compared against the primes already in N:

$$\begin{aligned}
\mathrm{reuse} &= \bigl|\Sigma_{\sigma} \cap N\bigr| \\[2pt]
\mathrm{newf}  &= \bigl|\Sigma_{\sigma} \setminus N\bigr| \\[4pt]
\mathrm{resonance} &\mathrel{+}= \mathrm{reuse} \times 1.5 - \mathrm{newf} \times 0.7 \\
                 &\qquad - \log_{10}(\mathrm{largest} + 1) \times 0.15
\end{aligned}$$

where $\Sigma_{\sigma}$ is the set of odd prime factors of $\sigma(p^{a})$, and
*reuse* / *newf* count how many of them are already in $N$ versus newly introduced.

States with high resonance (σ-factor recycling, characteristic of
Descartes-type structures) are explored first via a priority heap.

### Early Ratio Pruning

Before cloning a state to assign p^a, the code checks

$$\sigma(p^{a}) \times \mathrm{num} \;\geq\; 2 \times p^{a} \times \mathrm{den}$$

If true, the new state would be immediately pruned (ratio ≥ 2),
avoiding the cost of clone + factorisation.

---

## Author

**Chengyuan Tang**  ·  chengyuantang37@gmail.com

> This project was developed with AI-assisted tooling (Claude Code).
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
- Ochem, P. & Rao, M. (2012).  *Odd perfect numbers are greater than
  10^1500*.  Math. Comp. 81, 1869–1877.
  doi:[10.1090/S0025-5718-2012-02563-4](https://doi.org/10.1090/S0025-5718-2012-02563-4)
- Touchard, J. (1953).  *On prime numbers and perfect numbers*.
  Scripta Math. 19, 35–39.
