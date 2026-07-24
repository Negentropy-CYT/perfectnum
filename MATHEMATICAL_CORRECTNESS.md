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
max_euler_exponent + (max_factors - 1) * max_even_exponent.
```

With the checked-in values `MAX_FACTORS=10` and `MAX_EXP=9`, this is only
`9 + 9*8 = 81`; `MAX_PRIME=10000` is also a hard finite cutoff. These
settings are therefore an algorithm experiment, not a box capable of
containing an OPN under the standard published lower bounds.

Live states are never discarded because of heap size or priority. If a future
resource budget is added, reaching that budget must produce an explicit
`UNRESOLVED` result rather than an empty search frontier.

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

For each future prime p, the engine computes the maximum valuation that one
allowed component p^a could contribute to a Fermat-prime debt q. If h component
slots remain, the sum of the h largest individual capacities is an upper bound
on what any completion can contribute.

The implementation deliberately relaxes other constraints and even allows more
than one prospective component to use an Euler exponent while calculating this
upper bound. This can make the bound too large and miss a prune, but it cannot
make the bound too small. A state is rejected only when this relaxed capacity
is still less than the outstanding debt.

## Deferred Constraints

The following ideas are not active proof prunes:

- CRT merging of several debts into one parent prime;
- primitive-order obligations and Hall matching;
- partial-state mod-8 reachability;
- density estimates for primes in residue classes.

They require explicit debt allocation, exception handling, and completion
certificates before they can affect reachability.
