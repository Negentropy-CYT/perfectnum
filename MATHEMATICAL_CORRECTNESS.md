# Mathematical Correctness Contract

## Scope

The engine searches a finite box:

- every prime divisor is in the supplied `primes` list;
- the number of distinct prime divisors is at most `max_factors`;
- every exponent is at most `max_exp`.

Exhausting this box is not a proof that odd perfect numbers do not exist.
A prime forced beyond the final prime in the list is a contradiction only
relative to this finite-box contract.

For an Euler-form box, its maximum possible total multiplicity is

```text
E_euler + (F - 1) * E_even
```

where `E_euler` is the largest allowed exponent ≡ 1 mod 4 and `E_even`
the largest allowed even exponent.  The box is determined by the active
values of `MAX_PRIME`, `MAX_FACTORS`, and `MAX_EXP` at the top of
`opn_core.py`.  See the validated-regression configuration in the README
for a concrete example with known search-tree stability.

Live states are never discarded because of heap size or priority. If a future
resource budget is added, reaching that budget must produce an explicit
`UNRESOLVED` result rather than an empty search frontier.

## Sigma-Pool Analysis Contract

The pool analyser (`SigmaPoolAnalyzer`) classifies each σ(p^a) against the
configured odd-prime pool.  Four correctness invariants hold:

1. The prime pool **must** be the complete ordered set of odd primes from 3
   through `prime_limit`.  The analyser validates "starts at 3, odd, strictly
   increasing" but does **not** independently re-verify primality or
   completeness — it trusts the sieve output.

2. `exact=True` means the odd part of σ(p^a) has been fully stripped of
   every prime in the pool; `residual == 1` and the returned `valuations`
   is a complete map that can be written to `_SIG_VALUATIONS`.

3. `exact=False` means the cofactor after removing all in-pool primes
   exceeds 1.  For a **cold** (first-time) analysis, this cofactor is the
   complete residual after exhausting the pool.  For an
   **exact-global-cache** fast path, the residual may be a single
   certified outside-window witness rather than the full cofactor.
   In both cases `residual > 1` is sufficient for the finite-window
   rejection.  The returned `valuations` is **partial** and must **not**
   be used for factor-chain propagation or written to the global exact
   cache.

4. The hierarchical superblock GCD is **semantics-preserving**:
   `gcd(r, S) = 1 ⇒ gcd(r, B_i) = 1` for all child blocks, so no factor
   is ever missed.  The exponent-order filter is also a necessary
   condition (see below).

## Exponent Filter Correctness

For an odd prime q ≠ p dividing σ(p^a), let n = a+1.  From

```
σ(p^a) = (p^n - 1) / (p - 1)
```

and q ∤ (p - 1) (since q is odd and 0 < p - 1 < q for all
non-trivial cases), we have q | (p^n - 1).

**Case 1:** p ≡ 1 (mod q).  Then σ(p^a) ≡ n (mod q), so q | n.

**Case 2:** p ≢ 1 (mod q).  Let d = ord_q(p) > 1.  Then d | n
and d | (q - 1).  Hence gcd(q - 1, n) ≥ d > 1.

Therefore every odd prime factor q of σ(p^a) satisfies

```
q | (a + 1)   or   gcd(q - 1, a + 1) > 1.
```

This is a **necessary** condition — primes failing both tests cannot
divide σ(p^a) and are safely excluded from the pool for exponent a.
The filter is strict conservative: it may include primes that do not
actually divide σ(p^a), but it never excludes a real divisor.

## Factor-Slot Tail Bound

If a state has at most `r` distinct-prime slots left, each future component
satisfies

```text
sigma(p^a) / p^a < p / (p - 1).
```

The expression on the right strictly decreases with `p`. Therefore the
largest relaxed tail abundancy is obtained from the `r` smallest available
primes, not from every prime remaining in the search window.

`ratio_upper_bound()` computes this product exactly. Mandatory pending primes
are included even when they lie before `next_idx`; they consume slots before
optional primes are selected. States with an excluded pending prime, a pending
prime beyond the finite window, or more pending primes than remaining slots
are rejected before the bound is evaluated.

The same construction gives `next_prime_upper_bound()`: after a candidate
prime is fixed, at most `r-1` smallest available later primes form its best
possible tail. No floating-point comparison is used for either proof prune.

The former four arrays of full suffix products were a looser bound and needed
quadratic-size big-integer storage. They are not part of the search anymore.

## Valuation Ledgers

For an odd prime q, the state records:

```text
incoming_v[q] = sum v_q(sigma(p^a)) over processed components
assigned_exp[q] = exponent selected for q in N
target_offset[q] = v_q(target_num) - v_q(target_den)
owed_v[q] = max(assigned_exp[q] + target_offset[q] - incoming_v[q], 0)
```

The current field names are retained for checkpoint compatibility:

```text
required_v == incoming_v
current_v  == assigned_exp
```

`valuation_debts()` is the authoritative conversion to `owed_v`.
For OPN mode every odd-prime target offset is zero. For friend-of-10 mode,
the nonzero offsets are `offset[3]=2` and `offset[5]=-1`, as required by
`5*sigma(N)=9*N`.

## Exact Reverse Valuation

For distinct odd primes p and q, let n=a+1 and d=ord_q(p). Then

```text
v_q(sigma(p^a)) = 0                              if d does not divide n
                 = v_q(n)                        if d = 1
                 = v_q(p^d - 1) + v_q(n/d)       otherwise
```

`sigma_valuation_from_order()` implements this identity using multiplicative
order and LTE. It does not factor `sigma(p^a)`.

Full sigma factor maps are populated lazily only after cheaper state bounds
have passed. Once computed, a map is cached and used for both pre-clone
contradiction checks and mandatory factor-chain propagation.

## Residue-Class Count

Let `g=gcd(n,q-1)` and `t=v_q(n)`. The number of units x modulo q^e for which
q^e divides `1+x+...+x^(n-1)` is

```text
(g - 1) * q^min(t,e-1) + (q^(e-1) if t >= e else 0).
```

`residue_class_count()` implements this formula. A zero count applies to one
source component with the specified exponent. It is not by itself a state
contradiction because several future components may split a valuation debt.

## Fermat-Debt Capacity Prune

> **Note:** this prune is implemented but **disabled by default**
> (`ENABLE_FERMAT_DEBT = False` in `opn_core.py`).  It can be re-enabled
> for controlled experiments.

For each future prime p, the engine computes the maximum valuation that one
allowed component p^a could contribute to a Fermat-prime debt q. If h component
slots remain, the sum of the h largest individual capacities is an upper bound
on what any completion can contribute.

The implementation deliberately relaxes other constraints and even allows more
than one prospective component to use an Euler exponent while calculating this
upper bound. This can make the bound too large and miss a prune, but it cannot
make the bound too small. A state is rejected only when this relaxed capacity
is still less than the outstanding debt.

## Maximum-Prime Exponent Capacity

For the largest prime factor `R` of an odd perfect number `N`, the exponent
`v_R(N)` is bounded by the purely local quantity

```
B(u) = 1/2 * sum_{d|u, d>1} φ(d)²    where   u = oddpart(R-1).
```

This is a necessary-condition theorem that uses only `R` itself — no search
window, exponent cap, or abundance margin.  A Lean formalisation exists in
the author's local development tree but is not currently distributed with
this repository.  The Python implementation should not be treated as
independently machine-checked from this repository alone.

**In the search engine:** `max_prime_capacity(p)` in `opn_core.py` computes
`B(oddpart(p-1))`.  The check fires only when the current expansion candidate
`p` is guaranteed to be the largest prime factor of the completed `N`:

- in DFS mode when one free slot remains (`k_remain == 1`);
- in chain mode when one free slot remains and the pending queue is empty, or
  when the pending prime `q` satisfies `q >= all assigned primes` and
  `q >= all remaining pending primes`.

The guard is conservative: it never rejects an exponent that could be valid
for the maximum prime.  The rounding helpers `euler_max_exp_capacity` and
`even_max_exp_capacity` match the Lean theorems `euler_rounding` and
`nonEuler_rounding`.

**At current search parameters** (small `MAX_EXP`) the existing
`_max_possible_valuation` bound is often tighter.  The capacity bound
becomes the dominant constraint when `MAX_EXP` is raised significantly.

## Spoof-State Expansion

In factor-chain (true-OPN) mode a state that satisfies the Descartes-spoof
formula (`_check_spoof`) is yielded but **not** terminated: the state may
still accept more real prime-power components and evolve into a genuine
`σ(N) = 2N` solution.  The `continue` is guarded by `not use_heap`, i.e.
it stops expansion only in DFS (Descartes-spoof) mode where the spoof is
the intended end product.

## Deferred Constraints

The following ideas are not active proof prunes:

- CRT merging of several debts into one parent prime;
- primitive-order obligations and Hall matching;
- partial-state mod-8 reachability;
- density estimates for primes in residue classes.

They require explicit debt allocation, exception handling, and completion
certificates before they can affect reachability.
