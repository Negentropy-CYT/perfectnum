# OPN Search Report: P ≤ 8×10¹⁰, F ≤ 200, E ≤ 50

## 1. Search Parameters

| Parameter | Value |
|-----------|-------|
| Prime bound P | 79,999,999,999 |
| Maximum number of distinct odd prime factors | 200 |
| Maximum exponent | 50 |
| Target abundancy | σ(N)/N = 2 |
| Euler condition | enforced (Euler prime exponent ≡ 1 mod 4) |
| Propagation mode | factor chain (additive valuation constraints) |
| Admissible Euler exponents | {1, 5, 9, 13, 17, 21, 25, 29, 33, 37} |
| Admissible non-Euler exponents | {2, 4, 6, …, 40} |

The search traverses all candidate odd integers N = ∏ p_i^{a_i} under the factor-chain constraint: for every assigned σ(p^a), each odd prime factor q of σ(p^a) must either satisfy q ≤ P or the branch is rejected. A branch rejection due to "q exceeded P" means the current parameter bounds are insufficient to explore the branch further; it does not assert that the branch is mathematically impossible. Only rejections due to valuation contradiction or abundancy overshoot are unconditional.

---

## 2. Result

**The search bounds were exhausted. No N with σ(N)/N = 2 was found.**

| | |
|---|---|
| Status | exhausted |
| Solutions | 0 |
| Intermediate nodes retained | 35,638,533 |
| Branch rejections (total) | 452,159,965 |
| Maximum observed distinct prime factors | 33 |
| Factor-count ceiling (200) reached | no |

---

## 3. Branch Rejection Analysis

Each search branch is either retained for further exploration or rejected. Rejections fall into four categories:

| Reason | Count | Share |
|--------|-------|-------|
| σ(p^a) factor exceeded P | 411,181,729 | 90.9% |
| Valuation contradiction | 40,970,643 | 9.1% |
| Abundancy already above 2 | 7,592 | <0.01% |
| Remaining tail cannot reach 2 | 1 | <0.01% |

The dominant category (90.9%) is **σ(p^a) factor exceeded P**: the sigma expansion of an assignable component produced a prime factor larger than P. These branches are not mathematically impossible—they cannot be continued under the current bound.

The `valuation contradiction` category (9.1%) represents unconditional dead ends: the required p-adic valuation for a prime exceeds what is permitted by the exponent constraints on that prime. These branches cannot lead to an N with σ(N)/N = 2 regardless of P.

The `abundancy overshoot` and `tail-unreachable` categories are negligible, indicating that the ratio-comparison pruning is working correctly and does not generate false positives at scale.

---

## 4. Distribution of |assigned|

Let |assigned| denote the number of distinct odd primes assigned in a retained intermediate node. The distribution is approximately normal with a steep rightward cutoff:

| |assigned|| Nodes retained |
|------|----------------:|
| 1–5 | 10,024 |
| 6–10 | 440,845 |
| 11–15 | 4,381,076 |
| 16–20 | 9,742,418 |
| 21–25 | 17,215,179 |
| 26–30 | 2,760,671 |
| 31–33 | 122,376 |

Peak density occurs at |assigned| = 22 (4,655,719 nodes, 13.1%), with a sharp decline beyond 25: 26 = 1.50M, 27 = 0.54M, and only 4,002 nodes at 33. No node reached 34. The effective ceiling at ~33—far below the configured limit of 200—is a direct consequence of the P-bound rejection pressure: as chains of sigma propagation lengthen, the probability that a newly introduced odd prime factor exceeds P approaches certainty.

---

## 5. Sigma Factor Classification

Before a component p^a can be assigned, σ(p^a) is decomposed against the precomputed prime array (all odd primes ≤ P). The result is classified:

- **Exact**: all odd prime factors of σ(p^a) are ≤ P
- **Outside**: at least one odd prime factor exceeds P

Classification is performed once per distinct (p, exponent) pair.

### 5.1 Totals

| | Distinct (p, a) pairs |
|---|---|
| Exact | 1,667 |
| Outside | 19,866 |
| Total | 21,533 |

Only 7.7% of distinct (p, a) pairs are fully resolvable within P = 8×10¹⁰.

### 5.2 By Exponent

The exact rate drops sharply with increasing exponent:

| Exponent | Exact | Outside | Exact share |
|----------|-------|---------|-------------|
| 1 | 196 | 0 | 100.0% |
| 2 | 536 | 107 | 83.4% |
| 5 | 113 | 32 | 77.9% |
| 4 | 306 | 434 | 41.4% |
| 6 | 139 | 601 | 18.8% |
| 10–50 | 377 | 18,722 | <2% |

Exponents 1 and 2 retain high exact rates because σ(p) = p+1 and σ(p²) = p²+p+1 are small-argument polynomials unlikely to exceed P for moderate p. Starting from exponent 10, over 98% of all (p, a) pairs produce at least one out-of-bounds factor.

---

## 6. Factor-Chain Propagation

When a component p^a is assigned under factor-chain propagation, the valuations from σ(p^a) propagate to target primes in the partial state. Each event (p, a) → q is recorded.

### 6.1 Aggregate Statistics

| | |
|---|---|
| Unique source primes | 620 |
| Unique source components (p, a) | 1,648 |
| Unique (p, a, q) edges | 6,949 |
| Unique (p, q) edges (summed over a) | 5,081 |
| Total edge events (branch-accumulated) | 135,380,696 |

No single edge dominates: the top 1 edge accounts for 0.75% of events, the top 10 for 6.4%, and the top 100 for 33.2%. Propagation is broadly distributed across hundreds of source–target pairs. On average, each retained intermediate node triggers 3.80 propagation edge events, consistent with the typical σ(p^a) producing 3–8 odd prime factors.

For a fixed source component (p, a), all targets q receive identical event counts. This is a structural invariant of the implementation: each assignment of p^a invokes the same σ(p^a) valuation map, and every non-2 target is incremented once. The invariant holds for all 1,648 components in this run, confirming counting consistency.

### 6.2 Leading Source Components

Ordered by total branch-accumulated edge events.

| Rank | Source | Events | Distinct targets |
|------|--------|--------|------------------|
| 1 | 397^10 | 3,163,524 | 7 |
| 2 | 4271^6 | 3,113,408 | 4 |
| 3 | 2467^4 | 3,029,718 | 3 |
| 4 | 211^8 | 2,622,096 | 6 |
| 5 | 41^24 | 2,553,425 | 7 |
| 6 | 540181^2 | 2,541,804 | 3 |
| 7 | 4271^2 | 2,335,056 | 3 |
| 8 | 397^6 | 2,120,500 | 4 |
| 9 | 41^14 | 2,035,602 | 6 |
| 10 | 71^14 | 2,032,968 | 8 |

The prime 397 appears at two exponents (ranks 1 and 8), illustrating that a single base prime can produce structurally distinct propagation patterns at different powers. Components such as 540181^2 and 2467^4 generate exactly three targets each—a minimal propagation pattern typical of σ(p²) = p² + p + 1 and σ(p⁴) = p⁴ + p³ + p² + p + 1, both of which are irreducible polynomials of degree φ(·) that may factor into a small number of large primes.

---

## 7. Near-Target Abundancy States

During the search, intermediate nodes whose current abundancy I(S) = σ(S)/S satisfies 0 < 2 − I(S) ≤ 10⁻² are recorded. These are **not** solutions—they are partial factorizations that happen to lie close to the target. They are retained solely for post-hoc analysis.

### 7.1 Selection Funnel

| Stage | Count |
|-------|-------|
| Nodes with I(S) ∈ [1.99, 2) | 6,646 |
| Rejected: mandatory pending primes would push I(S) above 2 | 5,853 |
| Qualifying (written to disk) | 793 |

Of the 6,646 candidates within the raw gap, 88.1% are eliminated because the minimum contribution of their mandatory pending primes—each contributing at least (p+1)/p—would raise I(S) above 2. The 793 retained records are states where a positive gap persists even under the worst-case pending-prime lower bound.

### 7.2 Euler Prime Distribution

| Euler prime | Records | Share |
|-------------|---------|-------|
| 1093 | 620 | 78.2% |
| 13 | 102 | 12.9% |
| 398581 | 18 | 2.3% |
| 181 | 16 | 2.0% |
| 5390701 | 14 | 1.8% |
| 1741 | 10 | 1.3% |
| (no Euler assigned) | 8 | 1.0% |
| 157 | 4 | 0.5% |
| 5 | 1 | 0.1% |

The dominance of 1093 (78.2%) is not confined to this run—the same Euler prime and similar proportions were observed in the preceding P = 7.5×10¹⁰ search, suggesting it is a structural feature of the search space rather than a statistical fluctuation. The prime 1093 satisfies 1093 ≡ 1 (mod 4) (Euler condition) and 1093 ≡ 1 (mod 3), which is compatible with Touchard's congruence constraint (any OPN must satisfy N ≡ 1 mod 12 or N ≡ 9 mod 36). Its sigma expansion σ(1093) = 1094 = 2 × 547 introduces the small odd prime 547, favouring further productive assignments.

All 793 records correspond to 793 **distinct** assigned factorizations. No two distinct search paths produced identical assigned-prime sets, confirming that the branching strategy explores the space without redundant duplication at the factorization level.

### 7.3 Gap Magnitude

All 793 qualifying records fall within the gap interval [10⁻³, 10⁻²). The minimum raw gap was approximately 4.85 × 10⁻⁴. The minimum adjusted gap—computed after including mandatory pending primes at their minimal contribution—remains positive for all records, confirming that none of the partial factorizations can reach I(S) = 2 under known constraints.

### 7.4 Assigned and Pending Counts

Most near-target nodes carry 14–16 assigned factors and 12–15 pending primes. Nodes with ≤8 assigned factors are rare, indicating that reaching within 1% of the target abundancy typically requires a non-trivial number of prime-power components. The pending-prime cardinality is comparable to the assigned cardinality, reflecting the depth of valuation obligation generated by sigma propagation at moderate-to-high exponents.

---

## 8. Observations

### 8.1 P Is the Binding Constraint, Not the Factor Count

The observed maximum |assigned| was 33 against a configured limit of 200. Every branch reaching moderate depth was terminated by a σ(p^a) factor exceeding P before exhausting the factor-count allowance. The same pattern held at P = 5×10¹⁰ (F = 65) and P = 7.5×10¹⁰ (F = 200): **the prime bound P is the dominant constraint on the search-structure**, and increasing F alone does not expand the effectively traversed search space.

### 8.2 Recurrence of Euler Prime 1093

The prime 1093 is the dominant Euler assignment among near-target abundancy states across independent runs at different parameter scales. The same Euler prime appeared at P = 7.5×10¹⁰ with 620 out of 781 qualifying states (79.4%), an almost identical proportion. This quantitative stability suggests a structural explanation rooted in 1093's congruence properties and the factorisation of its sigma polynomial, rather than a parameter-dependent artefact. A systematic investigation of the σ(1093^a) trajectory across admissible Euler exponents may be warranted.

### 8.3 Search Structure Stability

The depth distribution at P = 8×10¹⁰ (E = 50) is qualitatively similar to that at P = 7.5×10¹⁰ (E = 45), but shifted to the right: the peak moved from 17 to 22, and the maximum observed |assigned| increased from 26 to 33. The shape of the distribution—central plateau with sharp rightward cutoff—is preserved, consistent with P-bound rejection acting as multiplicative attrition with no structural phase transition between these parameter regimes.

---

## 9. Resource Note

**These parameter bounds are at the limit of consumer hardware.** The combined memory footprint (physical working set plus memory-mapped files) exceeds 200 GiB, broken down as:

- 161 GiB: plan cache (`plan_cache/`)
- 37 GiB: prime pool (`prime_pool/`)
- ~24 GiB: in-memory working set (filtered prime array, sigma database, Python heap)

This is a **total address-space commitment in excess of 200 GiB**. If the machine has insufficient physical RAM, the operating system will thrash between the memory-mapped files and the page file, rendering the search impractical.

Scaling to P > 10¹¹ would inflate every component proportionally: the prime pool, the plan cache, and the sigma analysis cost all grow with the number of primes. The total memory commitment could reach 300 GiB or more. **These regimes should not be attempted without server-class hardware and careful configuration of the plan-build policy.** The persistent sigma database (`sigma_pool.sqlite3`) and plan cache (`plan_cache/`) must be preserved between runs; rebuilding from scratch requires re-scanning the full prime array and is both time- and I/O-intensive.

---

## Appendix: Attached Data Files

The following files contain the complete, reproducible search result data. All run identifiers, timestamps, and hardware-dependent performance measurements have been removed. Only the mathematically canonical fields are retained.

| File | Content |
|------|---------|
| `structure.json` | Depth (|assigned|) distribution, rejection reasons, propagation edges, sigma classification by exponent, concentration statistics |
| `conclusions.json` | Provable statements derived from this run |
| `abundancy_gap_summary.json` | Near-target state selection funnel, Euler prime distribution, assigned/pending count distributions |
| `abundancy_gap_states.jsonl` | Canonical near-target state records (793 lines) |
| `report_integrity.json` | Cross-field consistency verification (all checks pass) |
| `manifest.json` | Reproducible search configuration (with identifiers redacted) |

The following standard output files are deliberately excluded:
- `performance.json`, `performance.txt`, `performance_samples.csv`: hardware-dependent timing and memory measurements.
- `summary.txt`: contains elapsed time and machine-specific fields.
- `abundancy_gap_top.txt`, `abundancy_sigma_maps.json`: human-readable derivations; the canonical data in `abundancy_gap_states.jsonl` and `abundancy_gap_summary.json` is complete.
