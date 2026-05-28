"""
opn_state — polymorphic search states and constraint propagation.

Defines two state classes:
  - DFSState   — minimal (5 fields) for pseudo-solution DFS
  - ChainState — full (14 fields) for factor-chain best-first search

Key improvements over v1 unified State:
  - DFSState.clone() copies 2 collections vs 7 → ~60% less overhead
  - Separate assign_prime_dfs / assign_prime_chain functions
  - Touchard congruence pruning integrated into both paths
  - Additive q-adic valuation contradiction detection (chain mode)
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from gmpy2 import mpz

from opn_core import (
    PRUNE_STATS,
    _SIG_FACTORS,
    MAX_EXP,
    RESONANCE_REUSE_W,
    RESONANCE_NEWF_W,
    RESONANCE_GIANT_W,
    PRIORITY_RESONANCE_W,
    PRIORITY_DEPTH_W,
    check_touchard,
    factorize,
    power_pa,
    sigma_prime_power,
    touchard_force_3,
)


# ── priority helper ──────────────────────────────────────────
def _compute_priority(ratio_num, ratio_den, resonance, n_assigned):
    ratio = float(ratio_num) / float(ratio_den)
    return (abs(2.0 - ratio)
            - PRIORITY_RESONANCE_W * resonance
            - PRIORITY_DEPTH_W * n_assigned)


# ══════════════════════════════════════════════════════════════
# DFSState — minimal state for pseudo-solution DFS
# ══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class DFSState:
    """Minimal state for DFS pseudo-solution search (propagate=False).

    Omits: required_v, current_v, pending, pending_set, resonance, priority.
    Saves 5 collection deep-copies per clone vs the old unified State.
    """

    assigned:    Dict[int, int] = field(default_factory=dict)
    excluded:    set[int]       = field(default_factory=set)
    euler_prime: int | None     = None
    ratio_num:   mpz            = field(default_factory=lambda: mpz(1))
    ratio_den:   mpz            = field(default_factory=lambda: mpz(1))
    next_idx:    int            = 0
    depth:       int            = 0
    pseudo:      bool           = False

    def clone(self) -> "DFSState":
        return DFSState(
            assigned=dict(self.assigned),
            excluded=set(self.excluded),
            euler_prime=self.euler_prime,
            ratio_num=mpz(self.ratio_num),
            ratio_den=mpz(self.ratio_den),
            next_idx=self.next_idx,
            depth=self.depth + 1,
            pseudo=self.pseudo,
        )


# ══════════════════════════════════════════════════════════════
# ChainState — full state for factor-chain best-first search
# ══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ChainState:
    """Full state for factor-chain true-OPN search (propagate=True)."""

    assigned:    Dict[int, int]  = field(default_factory=dict)
    required_v:  Dict[int, int]  = field(default_factory=dict)
    current_v:   Dict[int, int]  = field(default_factory=dict)
    excluded:    set[int]        = field(default_factory=set)
    pending:     Deque[int]      = field(default_factory=deque)
    pending_set: set[int]        = field(default_factory=set)
    euler_prime: int | None      = None
    ratio_num:   mpz             = field(default_factory=lambda: mpz(1))
    ratio_den:   mpz             = field(default_factory=lambda: mpz(1))
    next_idx:    int             = 0
    depth:       int             = 0
    pseudo:      bool            = False
    resonance:   float           = 0.0
    priority:    float           = 0.0

    def clone(self) -> "ChainState":
        return ChainState(
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


# ── prune telemetry helper ──────────────────────────────────

def _reject(reason: str):
    """Increment prune counter and return None (sentinel for pruned branch)."""
    PRUNE_STATS[reason] += 1
    return None


# ── shared helpers ───────────────────────────────────────────

def _euler_ok(p, exp, euler_prime):
    if exp % 2 == 1:
        if p % 4 != 1 or exp % 4 != 1:
            return False
        if euler_prime is not None:
            return False
    return True


def _early_ratio_prune(ratio_num, ratio_den, p, exp):
    sig = sigma_prime_power(p, exp)
    pa = mpz(power_pa(p, exp))
    return ratio_num * sig >= 2 * ratio_den * pa


def _enqueue_pending(st, q):
    if q not in st.pending_set and q not in st.assigned:
        st.pending.append(q)
        st.pending_set.add(q)


def _max_possible_valuation(q, euler_prime, max_exp):
    if q == euler_prime:
        x = max_exp
        while x % 4 != 1 and x > 0:
            x -= 1
        return max(x, 1)
    return max_exp if max_exp % 2 == 0 else max_exp - 1


# ── assign_prime_dfs (lightweight, no propagation) ───────────

def assign_prime_dfs(st: DFSState, p: int, exp: int,
                     max_exp: int = MAX_EXP) -> Optional[DFSState]:
    """Assign p^exp to a DFSState.  No factor-chain propagation."""
    if p in st.excluded or p in st.assigned:
        return _reject("excluded")
    if _early_ratio_prune(st.ratio_num, st.ratio_den, p, exp):
        return _reject("ratio")
    if not _euler_ok(p, exp, st.euler_prime):
        return _reject("euler")

    ns = st.clone()
    ns.assigned[p] = exp
    sig = sigma_prime_power(p, exp)
    pa = mpz(power_pa(p, exp))
    ns.ratio_num = st.ratio_num * sig
    ns.ratio_den = st.ratio_den * pa
    if exp % 2 == 1:
        ns.euler_prime = p

    if not check_touchard(ns.euler_prime, ns.assigned, ns.excluded):
        return _reject("touchard")

    return ns


# ── assign_prime_chain (full factor-chain propagation) ───────

def assign_prime_chain(st: ChainState, p: int, exp: int, *,
                       propagate: bool = True,
                       max_exp: int = MAX_EXP) -> Optional[ChainState]:
    """Assign p^exp to a ChainState with full factor-chain propagation."""
    if p in st.excluded or p in st.assigned:
        return _reject("excluded")
    if _early_ratio_prune(st.ratio_num, st.ratio_den, p, exp):
        return _reject("ratio")
    if not _euler_ok(p, exp, st.euler_prime):
        return _reject("euler")

    ns = st.clone()
    ns.assigned[p] = exp
    ns.current_v[p] = ns.current_v.get(p, 0) + exp
    sig = sigma_prime_power(p, exp)
    pa = mpz(power_pa(p, exp))
    ns.ratio_num = st.ratio_num * sig
    ns.ratio_den = st.ratio_den * pa
    if exp % 2 == 1:
        ns.euler_prime = p

    if not check_touchard(ns.euler_prime, ns.assigned, ns.excluded):
        return _reject("touchard")

    # resonance (chain mode only)
    if propagate:
        _update_resonance(ns, st, p, exp)
        ns.priority = _compute_priority(
            ns.ratio_num, ns.ratio_den, ns.resonance, len(ns.assigned),
        )
    else:
        ns.priority = 0.0

    if not propagate:
        return ns

    # factor-chain propagation (additive valuation)
    for q, e in factorize(int(sig)):
        if q == 2:
            continue
        if q in ns.excluded:
            return _reject("valuation")

        ns.required_v[q] = ns.required_v.get(q, 0) + e

        if q in ns.assigned:
            if ns.required_v[q] > ns.current_v[q]:
                return _reject("valuation")
        else:
            if ns.required_v[q] > _max_possible_valuation(q, ns.euler_prime,
                                                          max_exp):
                return _reject("valuation")

        if ns.required_v[q] > ns.current_v.get(q, 0):
            _enqueue_pending(ns, q)

    return ns


# ── checkpoint validation ────────────────────────────────────

def validate_chain_state(st: ChainState) -> bool:
    """Check internal consistency of a ChainState after deserialisation.

    Returns False if any invariant is violated (silent corruption guard).
    """
    # pending / pending_set must agree
    if set(st.pending) != st.pending_set:
        return False
    # a pending prime must not already be assigned
    for q in st.pending_set:
        if q in st.assigned:
            return False
    # valuations must be non-negative
    for q, rq in st.required_v.items():
        if rq < 0:
            return False
        cq = st.current_v.get(q, 0)
        if cq < 0:
            return False
    return True


# ── resonance update ─────────────────────────────────────────

def _update_resonance(ns, parent, p, exp):
    if not _SIG_FACTORS:
        return
    factors_set = _SIG_FACTORS.get((p, exp))
    if factors_set is None:
        return
    assigned_keys = parent.assigned.keys()
    reuse = len(factors_set & assigned_keys)
    newf = len(factors_set - assigned_keys)
    ns.resonance += reuse * RESONANCE_REUSE_W - newf * RESONANCE_NEWF_W
    new_primes = factors_set - assigned_keys
    if new_primes:
        largest = max(new_primes)
        ns.resonance -= math.log10(largest + 1) * RESONANCE_GIANT_W
