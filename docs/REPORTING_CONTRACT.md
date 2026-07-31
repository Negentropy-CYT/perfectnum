# Reporting contract — `structure.json` and derived outputs

Each field declares its semantics, statistical unit, measurement point,
canonical/derived status, and exact derivation when not canonical.

---

## structure.json fields

### `productive_states`
- **semantics**: number of cloned-and-passed-validation search-tree nodes
- **unit**: branch-weighted event count (1 per successful `record_productive`)
- **when**: `record_productive()` in `opn_metrics.py:240`
- **canonical**: yes
- **checkpointed**: yes (part of `StructureMetrics` pickle)
- **allows missing**: no
- **derivation**: atomic `+= 1` in `record_productive`

### `depth_histogram`
- **key**: `depth` (int, range 0..MAX_FACTORS)
- **value**: branch-weighted count of productive states at that depth
- **canonical**: yes
- **checkpointed**: yes
- **derivation**: `depth_histogram[depth] += 1` (same call site as `productive_states`)

### `depth_factor_map`
- **key**: `(depth, assigned_count)`
- **value**: branch-weighted count of productive states with that (depth, |assigned|) pair
- **canonical**: yes
- **checkpointed**: yes
- **derivation**: `depth_factor_map[(depth, assigned_count)] += 1`
- **invariant**: `sum(counts) == productive_states`

### `headroom_by_factor`
- **key**: `(assigned_count, bucket)`
- **value**: branch-weighted count of productive states
- **bucket**: one of `{"<1e-6", "1e-6-1e-5", "1e-5-1e-4", "1e-4-1e-3", "1e-3-1e-2", ">1e-2"}`
- **canonical**: yes
- **checkpointed**: yes
- **derivation**: `headroom_by_factor[(assigned_count, bucket)] += 1`
- **invariant**: `sum(counts) == productive_states`

### `ratio_headroom`
- **key**: bucket label (str)
- **value**: branch-weighted count of productive states in that headroom bucket
- **canonical**: yes
- **checkpointed**: yes
- **derivation**: marginal of `headroom_by_factor` over `assigned_count`

### `prune_reasons`
- **key**: `PruneReason` enum value (str)
- **value**: branch-weighted prune event count by mathematical reason
- **canonical**: yes
- **checkpointed**: yes
- **derivation**: `record_prune()` increments both reason and mechanism
- **invariant**: `sum(values) == sum(prune_mechanisms values)` (from performance.json)

### `contradiction_attribution`
- **key**: `(prime, reason_label)`
- **value**: branch-weighted contradiction count
- **canonical**: yes
- **checkpointed**: yes

### `propagation_edges`
- **key**: `(source_prime, target_prime)`
- **value**: branch-weighted propagation event count
- **canonical**: no — derived by collapsing `propagation_exp_edges` over exponent
- **checkpointed**: yes
- **derivation**: `propagation_edges[(p,q)] = Σ_exp propagation_exp_edges[(p,exp,q)]`
- **recorded as**: both counters incremented together at `opn_state.py:716-717`

### `propagation_exp_edges`
- **key**: `(source_prime, exponent, target_prime)`
- **value**: branch-weighted propagation event count
- **canonical**: yes
- **checkpointed**: yes
- **unique candidate count**: no — multiple events with same (p, exp, q) accumulate

### `outside_pool_sources`
- **key**: `(prime, exponent, residual_bits)`
- **value**: count of sigma queries where the source prime lies outside the pool
- **canonical**: yes
- **checkpointed**: yes

### `outside_window_sources`
- **key**: `(prime, exponent, outside_prime)`
- **value**: count of sigma expansions yielding a prime outside the search window
- **canonical**: yes
- **checkpointed**: yes

### `sigma_exact`
- **value**: total count of sigma queries classified as exact (within-window, fully resolved)
- **canonical**: yes
- **checkpointed**: yes
- **invariant**: `sigma_exact == sum(sigma_exact_by_exp)`

### `sigma_outside`
- **value**: total count of sigma queries classified as outside-window
- **canonical**: yes
- **checkpointed**: yes
- **invariant**: `sigma_outside == sum(sigma_outside_by_exp)`

### `sigma_by_exponent`
- **fields per row**: `exp`, `exact`, `outside`
- **exact**: `sigma_exact_by_exp[exp]`
- **outside**: `sigma_outside_by_exp[exp]`
- **canonical**: yes (per-exponent arrays are the canonical source, this is their serialization)
- **checkpointed**: yes (arrays are pickled)

### `valuation_contradictions_by_exponent`
- **fields per row**: `exp`, `excluded` (kind 0), `overrun` (kind 1), `budget` (kind 2), `q3_total`
- **canonical**: yes
- **checkpointed**: yes

### `pending_prime_frequency`
- **key**: prime value (int)
- **value**: branch-weighted count of productive states where that prime was in the pending set
- **canonical**: yes
- **checkpointed**: yes

---

## abundancy_gap_summary.json fields

### `small_gap_states_seen`
- **semantics**: total productive states whose abundancy gap is within configured threshold
- **canonical**: yes
- **checkpointed**: via summary.json commit protocol
- **invariant**: `small_gap_states_seen == qualifying_states + pending_lower_bound_rejections`

### `qualifying_states`
- **semantics**: states within gap threshold that also pass the pending lower-bound check
- **canonical**: yes
- **checkpointed**: via summary.json commit protocol
- **invariant**: `qualifying_states == records_written + dropped_due_to_limit`

### `pending_lower_bound_rejections`
- **semantics**: states within gap threshold but whose pending-primes lower bound already overshoots target
- **canonical**: yes

### `records_written`
- **semantics**: number of JSONL records actually written to disk
- **canonical**: yes
- **checkpointed**: via summary.json commit protocol

### `dropped_due_to_limit`
- **semantics**: qualifying states not written because `max_records` was reached
- **canonical**: yes

---

## abundancy_gap_states.jsonl fields

### `productive_ordinal`
- **semantics**: monotonically increasing 1-based index of the productive state across the entire search
- **invariant**: strictly increasing across lines; no gaps within written range

### `ratio_num` / `ratio_den` (string-encoded integers)
- **semantics**: exact abundancy ratio of the assigned-factor product at capture time

### `gap_num` / `gap_den` (string-encoded integers)
- **semantics**: `target - ratio` as exact rational
- **derivation**: `gap_num = target_num * ratio_den - target_den * ratio_num`, `gap_den = target_den * ratio_den`
- **invariant**: must reconstruct exactly; must be positive and within configured max_gap

### `assigned` (list of `[prime, exponent]` pairs)
- **semantics**: prime factorization of the state at capture time
- **invariant**: primes must be distinct and non-overlapping with `pending`

---

## Consistency invariants (cross-field)

| # | Invariant | Scope |
|---|-----------|-------|
| I1 | `productive_states == Σ depth_factor_map counts` | structure |
| I2 | `productive_states == Σ headroom_by_factor counts` | structure |
| I3 | `propagation_edges(p,q) == Σ_exp propagation_exp_edges(p,exp,q)` | structure |
| I4 | `sigma_exact == Σ sigma_exact_by_exp` | structure |
| I5 | `sigma_outside == Σ sigma_outside_by_exp` | structure |
| I6 | `Σ prune_reasons == Σ prune_mechanisms` | cross (structure + performance) |
| I7 | `small_gap == qualifying + pending_rejected` | gap summary |
| I8 | `qualifying == written + dropped` | gap summary |
| I9 | JSONL ordinal strictly increasing | gap JSONL |
| I10 | JSONL gap matches ratio/target | gap JSONL |
