"""
opn_state — search state and constraint propagation.

Defines the ``State`` dataclass and the core ``assign_prime`` function
that enforces Euler constraints, computes σ(p^a), propagates
q-adic valuations, and updates the resonance heuristic score.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from gmpy2 import mpz

from opn_core import (
    _SIG_FACTORS,
    RESONANCE_REUSE_W,
    RESONANCE_NEWF_W,
    RESONANCE_GIANT_W,
    PRIORITY_RESONANCE_W,
    PRIORITY_DEPTH_W,
    factorize,
    power_pa,
    sigma_prime_power,
)


# ── priority helper ───────────────────────────────────────────
def _compute_priority(
    ratio_num: mpz, ratio_den: mpz, resonance: float, n_assigned: int,
) -> float:
    """Heap ordering key — lower value is explored first."""
    ratio = float(ratio_num) / float(ratio_den)
    return (abs(2.0 - ratio)
            - PRIORITY_RESONANCE_W * resonance
            - PRIORITY_DEPTH_W * n_assigned)


# ── State ─────────────────────────────────────────────────────
@dataclass(slots=True)
class State:
    """One node in the constraint-propagation search tree."""

    # ── prime factorisation of the candidate ──
    assigned:    Dict[int, int] = field(default_factory=dict)
    required_v:  Dict[int, int] = field(default_factory=dict)   # Σ v_q from σ side
    current_v:   Dict[int, int] = field(default_factory=dict)   # v_q already in N

    # ── search-space partitioning ──
    excluded:    set[int]              = field(default_factory=set)
    pending:     Deque[int]            = field(default_factory=deque)
    pending_set: set[int]              = field(default_factory=set)

    # ── Euler (special) prime ──
    euler_prime: Optional[int]         = None

    # ── accumulated σ(N)/N ──
    ratio_num:   mpz                   = mpz(1)
    ratio_den:   mpz                   = mpz(1)

    # ── search progress ──
    next_idx:    int                   = 0
    depth:       int                   = 0
    pseudo:      bool                  = False

    # ── heuristic guidance ──
    resonance:   float                 = 0.0
    priority:    float                 = 0.0

    # ──────────────────────────────────────────────────────────
    def clone(self) -> "State":
        """Deep-copy the state (used before branching)."""
        return State(
            assigned=dict(self.assigned),
            required_v=dict(self.required_v),
            current_v=dict(self.current_v),
            excluded=set(self.excluded),
            pending=deque(self.pending),
            pending_set=set(self.pending_set),
            euler_prime=self.euler_prime,
            ratio_num=mpz(self.ratio_num),
            ratio_den=mpz(self.ratio_den),
            next_idx=self.next_idx,
            depth=self.depth + 1,
            pseudo=self.pseudo,
            resonance=self.resonance,
            priority=self.priority,
        )


# ── pending-queue helper ──────────────────────────────────────
def _enqueue_pending(st: State, q: int) -> None:
    """Add *q* to the pending queue if not already there or assigned."""
    if q not in st.pending_set and q not in st.assigned:
        st.pending.append(q)
        st.pending_set.add(q)


# ── constraint propagation ────────────────────────────────────
def assign_prime(
    st: State, p: int, exp: int, *, propagate: bool = True,
) -> Optional[State]:
    """Return a new ``State`` with *p* ^ *exp* assigned, or ``None``.

    Performs (in order):
    1. early ratio pruning (skip clone if ratio already >= 2)
    2. Euler-prime constraint checks
    3. clone + ratio update
    4. resonance-score update via precomputed σ-factor sets
    5. factor-chain propagation (only when *propagate* is True)
    """
    # ── trivial rejection ──
    if p in st.excluded or p in st.assigned:
        return None

    # ── early ratio pruning (before clone) ──
    sig    = sigma_prime_power(p, exp)
    pa     = mpz(power_pa(p, exp))
    new_num = st.ratio_num * sig
    new_den = st.ratio_den * pa
    if new_num >= 2 * new_den:
        return None

    # ── Euler constraints (before clone for early exit) ──
    if exp % 2 == 1:
        if p % 4 != 1 or exp % 4 != 1:
            return None
        if st.euler_prime is not None:
            return None

    # ── clone & apply ──
    ns = st.clone()
    ns.assigned[p]  = exp
    ns.current_v[p]  = ns.current_v.get(p, 0) + exp
    ns.ratio_num     = new_num
    ns.ratio_den     = new_den
    if exp % 2 == 1:
        ns.euler_prime = p

    # ── resonance score ──
    _update_resonance(ns, st, p, exp)

    ns.priority = _compute_priority(
        ns.ratio_num, ns.ratio_den, ns.resonance, len(ns.assigned),
    )

    if not propagate:
        return ns

    # ── factor-chain propagation (additive valuation) ──
    for q, e in factorize(int(sig)):
        if q == 2:
            continue
        if q in ns.excluded:
            return None                     # contradiction

        ns.required_v[q] = ns.required_v.get(q, 0) + e
        if ns.required_v[q] > ns.current_v.get(q, 0):
            _enqueue_pending(ns, q)

    return ns


# ── resonance update ──────────────────────────────────────────
def _update_resonance(
    ns: State, parent: State, p: int, exp: int,
) -> None:
    """Adjust *ns.resonance* based on σ-factor overlap with parent state."""
    if not _SIG_FACTORS:
        return
    factors_set = _SIG_FACTORS.get((p, exp))
    if factors_set is None:
        return

    assigned_keys = parent.assigned.keys()
    reuse = len(factors_set & assigned_keys)
    newf  = len(factors_set - assigned_keys)

    ns.resonance += reuse * RESONANCE_REUSE_W - newf * RESONANCE_NEWF_W

    new_primes = factors_set - assigned_keys
    if new_primes:
        largest = max(new_primes)
        ns.resonance -= math.log10(largest + 1) * RESONANCE_GIANT_W
