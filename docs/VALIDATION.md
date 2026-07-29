# Validation

Validation is organized around mathematical equivalence, persistent-data
fallback, interruption safety, and performance reproducibility.

## Running tests

Run the complete local suite:

```bash
python -m pytest -q
```

Run the non-slow suite used by CI:

```bash
python -m pytest test_opn.py -v -m "not slow"
```

CI executes the non-slow suite on Python 3.10, 3.11, and 3.12.

## Mathematical oracles

The tests use independent scalar or direct-integer paths for critical
identities rather than comparing only two calls through the same optimized
implementation. Coverage includes:

- prime generation and compact integer-width boundaries;
- `sigma(p^a)` and cyclotomic product reconstruction;
- component-filter completeness for actual prime divisors;
- q-adic valuations and reverse-valuation identities;
- the q=3 LTE closed form against direct sigma valuation for every odd prime
  through 2000 and every exponent from 1 through 60;
- exact, partial, and outside-window residual semantics;
- repeated factors shared by more than one component;
- ratio and factor-slot bounds;
- deterministic complete searches on small boxes.

The production component scan and a direct prime-division oracle must agree on
`exact`, valuations, residual, and outside classification.

## Search-structure regression

Search validation compares mathematical structure, not just the final solution
count. Stable comparisons cover:

- productive states and prune reasons;
- depth distribution;
- factor-chain propagation edges;
- sigma exact/outside classifications;
- valuation contradictions;
- pending-prime behavior;
- solution sets.

Performance counters are allowed to change after a semantics-preserving
optimization. Mathematical structure is not.

## Persistent-cache validation

Sigma database tests cover:

- exact warm reuse;
- partial record reuse when the window grows;
- checksum and arithmetic-identity failure;
- component reconstruction mismatch;
- invalid-row fallback to a fresh scan;
- deterministic close and flush behavior on Windows.

Plan cache tests cover:

- cold construction and warm loading;
- compatibility-key separation;
- read-only mapped prime arrays;
- checksum and framing corruption;
- interrupted publication;
- lock contention;
- insufficient-disk fallback.

Cache failure must reduce to a miss, rebuild, or in-memory execution. It must
never produce a prune from unvalidated data.

## Interruption and resume

Checkpoint tests compare uninterrupted execution with a stopped and resumed
execution. They validate:

- stable-boundary snapshots;
- heap order and unique tie identifiers;
- counters and elapsed offsets;
- saved prime-pool metadata and regenerated-pool agreement;
- mode and configuration coherence;
- checkpoint-callback failures;
- analyzer cleanup on normal, stopped, interrupted, and exceptional exits.

The resumed structure and solutions must match continuous execution.

## Performance checks

Performance is platform- and cache-dependent and is secondary to mathematical
acceptance. Any comparison must identify its hardware and cold/warm/expansion
conditions, use repeated isolated runs, and establish structural equality
before interpreting elapsed time, RSS, or disk use. Measurements belong in the
generated run records rather than the correctness contract.
