# Mathematical Correctness Contract

## Notation

For a positive integer $n$,

```math
\sigma(n)=\sum_{d\mid n}d,
\qquad
I(n)=\frac{\sigma(n)}{n}
```

are the sum-of-divisors function and abundancy index. For a prime $q$,
$v_q(n)$ is the $q$-adic valuation of $n$. The notation $p^a\parallel N$ means
that $p^a$ is the exact prime-power component of $N$: $p^a\mid N$ but
$p^{a+1}\nmid N$. The function $\omega(N)$ denotes the number of distinct
prime divisors.

$\operatorname{oddpart}(m)=m/2^{v_2(m)}$, and $\varphi$ denotes Euler's
totient function.

## Scope

In Euler-form OPN mode (`PROPAGATE=True`, `SEARCH_MODE=OPN_MODE`), the engine
searches the finite domain in which:

- every prime divisor of `N` is in the supplied `primes` list;
- $\omega(N)\leq F$, where `F = max_factors`;
- every exponent is at most `max_exp`.

Exhausting this box is not a proof that odd perfect numbers do not exist.
A prime forced beyond the final prime in the list is a contradiction only
relative to this finite-box contract.

In Descartes-type DFS mode (`PROPAGATE=False`), the same limits apply to the
explicitly selected, distinct odd-prime bases and their exponents. The derived
pseudo-prime factor `r` is required to be greater than 1 and coprime to those
bases, but it is not drawn from the prime pool, is not counted by
`max_factors`, and is not required by `_check_spoof()` to be composite.
Consequently, completion of a spoof run excludes only candidates in this
restricted selected-base domain.

In friend-of-10 mode (`PROPAGATE=True`,
`SEARCH_MODE=FRIEND_10_MODE`), the selected bases are again odd primes from
the supplied list, all exponents are even, prime 5 is forced with exponent at
least 2, prime 3 is excluded, and the target abundancy index is $9/5$.
The same prime, factor-count, and exponent limits define this separate finite
domain.

For reference, Euler's form is

```math
N=q^\alpha m^2,
\qquad
\gcd(q,m)=1,
\qquad
q\equiv\alpha\equiv1\pmod 4,
```

where `q` is the special prime, also called the Euler prime. This is the
Eulerian form used throughout the OPN literature; see
[Nielsen (2007)](https://doi.org/10.1090/S0025-5718-07-01990-4).

For an Euler-form box, its maximum possible total multiplicity is

```math
E_{\mathrm{Euler}}+(F-1)E_{\mathrm{even}}
```

where $E_{\mathrm{Euler}}$ is the largest allowed exponent congruent to
$1\pmod 4$ and $E_{\mathrm{even}}$ the largest allowed even exponent. The box
is determined by the active
values of `MAX_PRIME`, `MAX_FACTORS`, and `MAX_EXP` at the top of
`opn_core.py`. The exact values and active policy are recorded in each run's
`manifest.json`.

Live states are never discarded because of heap size or priority. If a future
resource budget is added, reaching that budget must produce an explicit
`UNRESOLVED` result rather than an empty search frontier.

## Sigma-Pool Analysis Contract

The pool analyzer (`SigmaPoolAnalyzer`) classifies each $\sigma(p^a)$ against
the configured odd-prime pool. Six correctness invariants hold:

1. The prime pool **must** be the complete ordered set of odd primes from 3
   through `prime_limit`.  The analyzer validates "starts at 3, odd, strictly
   increasing" but does **not** independently re-verify primality or
   completeness — it trusts the sieve output.

2. `exact=True` means the odd part of $\sigma(p^a)$ has been fully stripped of
   every prime in the pool; `residual == 1` and the returned `valuations`
   is a complete map that can be written to `_SIG_VALUATIONS`.

3. `exact=False` means the cofactor after removing all in-pool primes
   exceeds 1.  Both cold scans and exact-global-cache partitioning return
   the **complete** outside-window cofactor with multiplicity, not merely
   one witness.  `residual > 1` is sufficient for the finite-window
   rejection.  The returned `valuations` is **partial** and must **not**
   be used for factor-chain propagation or written to the global exact
   cache.

4. The hierarchical superblock GCD is **semantics-preserving**:
   $\gcd(r,S)=1\Longrightarrow\gcd(r,B_i)=1$ for all child blocks, so no factor
   is ever missed.  Each cyclotomic component filter is also a necessary
   condition (see below).

5. Persistent records are derived cache entries, not unchecked facts.  Before
   reuse, their factor keys are prime-tested, their payload checksum is
   verified, and the exact identity

   ```math
   \operatorname{oddpart}\!\left(\sigma(p^a)\right)
   =\mathrm{residual}\prod_q q^{\mathrm{valuations}[q]}
   ```

   is recomputed.  A partial record additionally carries a SHA-256 digest of
   the complete prime-pool prefix it scanned.  On a larger compatible window,
   the analyzer regenerates every exact cyclotomic component and removes the
   stored valuations across those components. The product of the resulting
   component residuals must equal the stored residual; otherwise the record is
   ignored and a fresh full-window analysis is used. Compatible components are
   scanned against either the new interval or the suffix of their full plans.
   Rounding down to a leaf boundary can only recheck certified primes; it
   cannot skip a new prime.

6. Persistent pool plans do not change the set of eligible primes or any GCD
   identity. Filtered prime arrays are the output of the same two-pass
   component-order filter and are consumed read-only through `numpy.memmap`.
   Superblock products are stored with length framing and SHA-256 checksums,
   then fully reconstructed as `mpz` objects before scanning. The cache key
   binds the complete pool-prefix digest, source interval, integer dtype,
   explicit component filter and order, block size, fanout, format identity,
   and plan semantics. Missing, incompatible, truncated, oversized, or
   checksum-invalid entries are rebuilt or fall back to the ordinary in-memory
   path. Thus disk persistence changes representation and lifetime only; it
   introduces no additional prune.

## Cyclotomic Decomposition and Component Filter

Let $n=a+1$. The analyzer uses the exact integer identity

```math
\sigma(p^a)=\prod_{\substack{d\mid n\\d>1}}\Phi_d(p).
```

$\Phi_d(p)$ is computed by the divisor recurrence

```math
p^d-1=\prod_{c\mid d}\Phi_c(p).
```

Every recurrence division is checked
for zero remainder, and the final component product is checked against the
ordinary integer formula for $\sigma(p^a)$. The 2-part is removed from each
component independently. All components are scanned to completion; repeated
prime factors in different components have their valuations added, and the
complete final residual is the product of all component residuals.

For a prime $q$ dividing $\Phi_d(p)$, the cyclotomic order theorem gives

```math
q\mid d
\qquad\text{or}\qquad
\operatorname{ord}_q(p)=d.
```

In the second case, $d\mid q-1$. Therefore every odd prime divisor of
$\Phi_d(p)$ is contained in

```math
q\mid d
\qquad\text{or}\qquad
q\equiv1\pmod d.
```

The component plan uses exactly this conservative necessary filter. It may
contain false positives, but it cannot omit an actual prime divisor. The
$d=2$ plan is the complete odd-prime pool because every odd prime is
congruent to $1\pmod 2$.

## Factor-Slot Tail Bound

If a state has at most `r` distinct-prime slots left, each future component
satisfies

```math
\frac{\sigma(p^a)}{p^a}<\frac{p}{p-1}.
```

The expression on the right strictly decreases with `p`. Therefore the
largest relaxed tail multiplier is obtained from the `r` smallest available
primes, not from every prime remaining in the search window.

`ratio_upper_bound()` computes this product exactly. Mandatory pending primes
are included even when they lie before `next_idx`; they consume slots before
optional primes are selected. States with an excluded pending prime, a pending
prime beyond the finite window, or more pending primes than remaining slots
are rejected before the bound is evaluated.

The same construction gives `next_prime_upper_bound()`: after a candidate
prime is fixed, at most `r-1` smallest available later primes form its best
possible tail. No floating-point comparison is used for either sound pruning
condition.

The implementation constructs only the factor-slot-aware bound described
above; it does not materialize full suffix-product tables.

## Next-Prime Interval Bounds

Let the current abundancy index be $R$, the target be $T$, and $p$ the next
prime-power base. Since

```math
\frac{\sigma(p^a)}{p^a}\geq\frac{p+1}{p}
```

for every positive exponent, a branch that does not overshoot the target must
satisfy

```math
R\frac{p+1}{p}\leq T.
```

When $R<T$, rearranging gives the necessary lower bound

```math
p\geq\frac{R}{T-R}.
```

`next_prime_lower_bound()` computes the integer ceiling of this expression
with exact integer arithmetic. A zero return means that the current ratio is
already at or above the target, so this particular formula supplies no lower
bound; the ordinary ratio checks handle that state.

For the upper bound, fix a candidate `p` and multiply the current ratio by the
largest relaxed contribution available from the remaining factor slots. As in
the factor-slot tail bound above, this contribution uses the smallest
available later primes because `x/(x-1)` decreases with `x`. If even that
relaxed completion cannot reach `T`, the candidate is too large.
`next_prime_upper_bound()` solves the resulting exact rational inequality for
`p`. The search evaluates this interval only after mandatory pending primes
have been drained; assigned and excluded primes are omitted from the available
tail.

Together these functions restrict only the interval in which the next prime
can occur. They do not assume that the relaxed tail itself is a realizable
factor chain. This is the interval construction used in
[Nielsen (2015), Proposition 3](https://doi.org/10.1090/S0025-5718-2015-02941-X).

## Domain-Aware Mandatory Ratio Lower Bound

For every mandatory pending prime `q`, the valuation ledger supplies a minimum
required exponent. The implementation selects the smallest admissible
exponent at or above that requirement:

- an even exponent for an ordinary component;
- an exponent congruent to 1 modulo 4 when `q` is eligible to be the Euler
  prime.

For a fixed prime, $\sigma(q^a)/q^a$ increases with $a$. Therefore assigning
these minimum admissible exponents gives a lower bound on the ratio contributed
by every completion of the pending obligations.

If the Euler prime is already fixed, every pending prime must use an even
exponent. If one pending prime has no even exponent available, it is forced to
take the Euler role; two such primes make the domain empty. Otherwise the
implementation evaluates the all-even relaxation and each possible pending
Euler assignment, retaining the smallest exact rational product. Allowing an
all-even pending set is conservative because the Euler prime may be supplied
later by an optional component.

A state is rejected only when the resulting lower bound is strictly greater
than the target ratio, or when every admissible exponent assignment is empty.
Equality is not pruned. `DOMAIN_RATIO_MODE="shadow"` evaluates this condition
without changing reachability; `"enforce"` applies it; `"off"` skips it. The
checked-in configuration uses `"off"`.

## Valuation Ledgers

For an odd prime $q$, the state records:

```math
\begin{aligned}
\mathrm{incoming\_v}[q]
  &=\sum_{\text{processed }p^a}v_q\!\left(\sigma(p^a)\right),\\
\mathrm{assigned\_exp}[q]
  &=v_q(N),\\
\mathrm{target\_offset}[q]
  &=v_q(\mathrm{target\_num})-v_q(\mathrm{target\_den}),\\
\mathrm{owed\_v}[q]
  &=\max\!\left(
      \mathrm{assigned\_exp}[q]+\mathrm{target\_offset}[q]
      -\mathrm{incoming\_v}[q],\,0
    \right).
\end{aligned}
```

Here `v_q` is the q-adic valuation defined above. Terms such as valuation
“debt” and “obligation” are project terminology for `owed_v`; they are not
additional number-theoretic assumptions.

The state representation stores these quantities under the field names:

```text
required_v == incoming_v
current_v  == assigned_exp
```

`valuation_debts()` is the authoritative conversion to `owed_v`.
For OPN mode every odd-prime target offset is zero. For friend-of-10 mode,
the nonzero offsets are `offset[3]=2` and `offset[5]=-1`, as required by
$5\sigma(N)=9N$.

## Exact Reverse Valuation

For distinct odd primes $p$ and $q$, let $n=a+1$ and
$d=\operatorname{ord}_q(p)$. Then

```math
v_q\!\left(\sigma(p^a)\right)=
\begin{cases}
0, & d\nmid n,\\
v_q(n), & d=1,\\
v_q(p^d-1)+v_q(n/d), & d\mid n\text{ and }d>1.
\end{cases}
```

`sigma_valuation_from_order()` implements this identity using multiplicative
order and the lifting-the-exponent lemma (LTE). It does not factor
$\sigma(p^a)$.

Full sigma factor maps are populated lazily only after cheaper state bounds
have passed. Once computed, a map is cached and used for both pre-clone
contradiction checks and mandatory factor-chain propagation.

## q=3 LTE Prepool Prune

The standard Euler-form OPN search performs an exact 3-adic valuation check
before consulting the sigma database or scanning a prime-pool plan. Let
$n=a+1$. For an odd prime $p$,

```math
v_3\!\left(\sigma(p^a)\right)=
\begin{cases}
0, & p=3,\\
v_3(n), & p\equiv1\pmod 3,\\
0, & p\equiv-1\pmod 3\text{ and }n\text{ is odd},\\
v_3(p+1)+v_3(n/2),
  & p\equiv-1\pmod 3\text{ and }n\text{ is even}.
\end{cases}
```

For $p\equiv1\pmod 3$, LTE gives

```math
v_3(p^n-1)-v_3(p-1)=v_3(n).
```

For $p\equiv-1\pmod 3$ and even $n$, apply LTE to
$(p^2)^{n/2}-1$. Since $3\nmid p-1$, the result is
$v_3(p+1)+v_3(n/2)$. For odd $n$, the alternating residue sum is 1 modulo 3.
For $p=3$, $\sigma(3^a)$ is also 1 modulo 3. Thus the formula is exact and
does not construct or factor $\sigma(p^a)$.

The incoming value is passed through the same additive valuation rule as a
complete sigma map. With

```text
new_required = required_v[3] + v_3(sigma(p^a)),
```

the branch is contradictory exactly when one of the following holds:

- 3 is excluded but `new_required` exceeds the target offset;
- 3 is already assigned but `new_required` exceeds its assigned exponent plus
  the target offset;
- 3 is unassigned but `new_required` exceeds the largest legal future exponent
  plus the target offset.

`Q3_PREPOOL_MODE="enforce"` applies this contradiction before cloning and
before pool analysis. `"shadow"` records the predicted contradiction but
continues through the ordinary sigma path so the valuation and classification
can be compared. `"off"` disables the precheck. The implementation enables it
only for factor-chain Euler-form searches with target $\sigma(N)/N=2$; it does
not silently generalize the prune to other search targets.

This is an execution shortcut for an exact valuation contradiction, not a new
number-theoretic assumption. It changes where the contradiction is detected,
not which branch is mathematically reachable.

## Touchard Congruence and Forced Prime 3

[Touchard's theorem](https://arxiv.org/abs/1709.05286) requires an odd perfect
number to satisfy

```math
N\equiv1\pmod{12}
\qquad\text{or}\qquad
N\equiv9\pmod{36}.
```

If $3\mid N$, it is not the Euler prime because $3\not\equiv1\pmod 4$;
therefore its exponent is even and at least 2. If `3` does not divide `N`,
Touchard forces $N\equiv1\pmod 3$. Every non-Euler prime power has even
exponent and contributes 1 modulo 3, so an Euler prime congruent to 2 modulo 3
would make $N\equiv2\pmod 3$. Consequently:

- if the Euler prime is 2 modulo 3 and 3 is excluded, the state is impossible;
- if the Euler prime is 2 modulo 3 and 3 is undecided, 3 is mandatory and is
  added to the pending factor chain;
- if 3 is assigned, its exponent must be an allowed even exponent.

The Touchard check constrains the residue class of `N`. The q=3 prepool check
instead constrains the additive valuation contributed by $\sigma(p^a)$. They are
independent necessary conditions even though both involve the prime 3.

## Residue-Class Count

Let $n=a+1$, let $q$ be an odd prime, and let $e\geq1$. Put
$g=\gcd(n,q-1)$ and $t=v_q(n)$. The number of units
$x\in(\mathbb Z/q^e\mathbb Z)^\times$ for which
$q^e\mid 1+x+\cdots+x^{n-1}$ is

```math
(g-1)q^{\min(t,e-1)}
+
\begin{cases}
q^{e-1}, & t\geq e,\\
0, & t<e.
\end{cases}
```

`residue_class_count()` implements this formula. A zero count applies to one
source component with the specified exponent. It is not by itself a state
contradiction because several future components may split a valuation debt.

## Fermat-Debt Capacity Prune

> **Note:** this prune is implemented but **disabled by default**
> (`ENABLE_FERMAT_DEBT = False` in `opn_core.py`).  It can be re-enabled
> for controlled experiments.

For each future prime $p$, the engine computes the maximum valuation that one
allowed component $p^a$ could contribute to a Fermat-prime debt $q$. If $h$
component slots remain, the sum of the $h$ largest individual capacities is
an upper bound on what any completion can contribute.

The implementation deliberately relaxes other constraints and even allows more
than one prospective component to use an Euler exponent while calculating this
upper bound. This can make the bound too large and miss a prune, but it cannot
make the bound too small. A state is rejected only when this relaxed capacity
is still less than the outstanding debt.

## Maximum-Prime Exponent Capacity

For the largest prime factor $R$ of an odd perfect number $N$, the exponent
$v_R(N)$ is bounded by the purely local quantity

```math
B(u)=\frac12\sum_{\substack{d\mid u\\d>1}}\varphi(d)^2,
\qquad
u=\operatorname{oddpart}(R-1).
```

This is a necessary-condition theorem that uses only `R` itself — no search
window, exponent cap, or abundancy headroom. A Lean formalisation exists in
the author's local development tree but is not currently distributed with
this repository. The Python implementation should not be treated as
independently machine-checked from this repository alone.

This theorem is an explicit soundness dependency of the active capacity
prune. Because neither a public proof nor the Lean artifact is included here,
an external audit must treat the theorem as a stated dependency rather than a
locally reproducible formal result.

**In the search engine:** `max_prime_capacity(p)` in `opn_core.py` computes
$B(\operatorname{oddpart}(p-1))$. The check fires only when the current
expansion candidate `p` is guaranteed to be the largest prime factor of the
completed `N`:

- in Euler-form OPN mode when one free slot remains and the pending queue is
  empty, or
  when the pending prime `q` satisfies `q >= all assigned primes` and
  `q >= all remaining pending primes`.

The capacity theorem is not applied in Descartes-type DFS mode. Its derived
pseudo-prime factor `r` can contain prime divisors larger than the explicitly
selected bases, so the required maximum-prime hypothesis would not be
established there.

The guard is conservative: it never rejects an exponent that could be valid
for the maximum prime. The rounding helpers `euler_max_exp_capacity` and
`even_max_exp_capacity` match the Lean theorems `euler_rounding` and
`nonEuler_rounding`.

**At current search parameters** (small `MAX_EXP`) the existing
`_max_possible_valuation` bound is often tighter.  The capacity bound
becomes the dominant constraint when `MAX_EXP` is raised significantly.

## Spoof-State Expansion

In Euler-form OPN mode, a state that satisfies the restricted Descartes-type
equation (`_check_spoof`) is yielded as a candidate but **not** terminated: the
state may
still accept more real prime-power components and evolve into a genuine
$\sigma(N)=2N$ solution.  The `continue` is guarded by `not use_heap`, i.e.
it stops expansion only in DFS mode where that candidate class is the intended
output.

## Deferred Constraints

The following ideas are not active sound pruning rules:

- CRT merging of several debts into one parent prime;
- primitive-order obligations and Hall matching;
- partial-state mod-8 reachability;
- density estimates for primes in residue classes.

They require explicit debt allocation, exception handling, and completion
certificates before they can affect reachability.

`is_prime_infinite()` retains factorization cutoffs from the Mathematica
reference as classification metadata. It is not called by the production
search and never permits the engine to skip sigma analysis or factor-chain
propagation.
