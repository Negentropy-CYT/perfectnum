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
    CASCADE_DEPTH_HIST,
    CLONE_PAYLOAD,
    CLONE_STATS,
    CONTRADICTION_ATTR,
    DEPTH_FACTOR_MAP,
    DEPTH_STATS,
    SEARCH_MODE,
    HEADROOM_BY_FACTOR,
    INFINITE_POWER_LIMIT,
    OBLIGATION_SIGS,
    PENDING_SIZE_HIST,
    PROPAGATION_EDGES,
    PRUNE_STATS,
    RATIO_HEADROOM,
    TARGET_DEN,
    TARGET_NUM,
    _SIG_FACTORS,
    _SIG_VALUATIONS,
    MAX_EXP,
    RESONANCE_REUSE_W,
    RESONANCE_NEWF_W,
    RESONANCE_GIANT_W,
    PRIORITY_RESONANCE_W,
    PRIORITY_DEPTH_W,
    check_fermat_contradiction,
    check_touchard,
    factorize,
    is_prime_infinite,
    power_pa,
    sigma_prime_power,
    touchard_force_3,
)


# ── priority helper ──────────────────────────────────────────
def _compute_priority(ratio_num, ratio_den, resonance, n_assigned):
    target = TARGET_NUM / TARGET_DEN
    ratio = float(ratio_num) / float(ratio_den)
    return (abs(target - ratio)
            - PRIORITY_RESONANCE_W * resonance
            - PRIORITY_DEPTH_W * n_assigned)


# ══════════════════════════════════════════════════════════════
# Feasibility Cache
# ══════════════════════════════════════════════════════════════

class FeasibilityCache:
    """Caches impossible valuation-deficit patterns across search paths.

    Key insight: many different search branches converge to the *same*
    valuation obligation topology (same required_v, same deficits).  If
    one path proves a deficit pattern unsatisfiable, all other paths
    leading to the same pattern can be pruned — regardless of which
    specific primes got them there.

    Contrast with ContradictionCache (in opn_search): that cache requires
    assigned_keys ⊇ cached_assigned, which makes it more precise but less
    general.  FeasibilityCache is purely deficit-based — broader, faster,
    complementary.

    Signature
    ---------
    frozenset of (q, deficit) pairs where deficit = required_v[q] - current_v[q].
    """
    CACHE_MAX_SIZE = 100_000

    def __init__(self):
        self._failed: set = set()

    def _signature(self, required_v, current_v):
        items = []
        for q, req in required_v.items():
            deficit = req - current_v.get(q, 0)
            if deficit > 0:
                items.append((q, deficit))
        return frozenset(items)

    def add(self, required_v, current_v):
        sig = self._signature(required_v, current_v)
        if len(self._failed) >= self.CACHE_MAX_SIZE:
            self._failed.clear()  # aggressive: just reset (rare in practice)
        self._failed.add(sig)

    def contains(self, required_v, current_v):
        return self._signature(required_v, current_v) in self._failed

    def __len__(self):
        return len(self._failed)


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
        CLONE_STATS["total"] += 1
        CLONE_PAYLOAD[len(self.assigned)] += 1
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
        CLONE_STATS["total"] += 1
        CLONE_PAYLOAD[len(self.assigned)] += 1
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
    """Increment prune counter and return None (sentinel for pruned branch).

    Also classifies the prune as pre-clone (saved) or post-clone (wasted)
    for clone-effectiveness telemetry.
    """
    PRUNE_STATS[reason] += 1
    if reason in ("excluded", "ratio", "euler", "valuation_pre"):
        CLONE_STATS["saved"] += 1   # clone was avoided
    else:
        CLONE_STATS["wasted"] += 1  # clone was already paid
    return None


# ── shared helpers ───────────────────────────────────────────

def _euler_ok(p, exp, euler_prime):
    if not SEARCH_MODE.require_euler:
        # non-Euler mode: all exponents must be even
        return exp % 2 == 0
    if exp % 2 == 1:
        if p % 4 != 1 or exp % 4 != 1:
            return False
        if euler_prime is not None:
            return False
    return True


def _early_ratio_prune(ratio_num, ratio_den, p, exp):
    sig = sigma_prime_power(p, exp)
    pa = mpz(power_pa(p, exp))
    return ratio_num * sig * TARGET_DEN > TARGET_NUM * ratio_den * pa


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

    DEPTH_STATS[ns.depth] += 1
    CLONE_STATS["productive"] += 1
    return ns


# ── assign_prime_chain (full factor-chain propagation) ───────

def assign_prime_chain(st: ChainState, p: int, exp: int, *,
                       propagate: bool = True,
                       max_exp: int = MAX_EXP) -> Optional[ChainState]:
    """Assign p^exp to a ChainState with full factor-chain propagation.

    IMPROVEMENT: pre-clone valuation contradiction check using
    _SIG_VALUATIONS.  Previously, the clone was paid first, then σ(p^a)
    was factorised, and only then was a valuation contradiction detected
    (49% wasted clone rate).  Now the precomputed valuation map enables
    checking *before* clone — moving wasted → saved.
    """
    if p in st.excluded or p in st.assigned:
        return _reject("excluded")
    if _early_ratio_prune(st.ratio_num, st.ratio_den, p, exp):
        return _reject("ratio")
    if not _euler_ok(p, exp, st.euler_prime):
        return _reject("euler")
    if check_fermat_contradiction(p, exp, st.assigned, st.excluded):
        return _reject("fermat")

    # ── interval skip (pre-clone, counts as saved) ──
    # Tracked via _reject but actually handled in the expansion loop.
    # The _reject path here is for future per-assign interval checks.

    # ── pre-clone valuation check (uses precomputed _SIG_VALUATIONS) ──
    if propagate:
        pre_vals = _SIG_VALUATIONS.get((p, exp))
        if pre_vals is not None:
            for q, e in pre_vals.items():
                # q is an odd prime factor of σ(p^a) with exponent e
                if q in st.excluded:
                    CONTRADICTION_ATTR[(q, "excluded_pre")] += 1
                    return _reject("valuation_pre")
                new_req = st.required_v.get(q, 0) + e
                if q in st.assigned:
                    if new_req > st.current_v[q]:
                        CONTRADICTION_ATTR[(q, "overrun_pre")] += 1
                        return _reject("valuation_pre")
                else:
                    if new_req > _max_possible_valuation(q, st.euler_prime,
                                                         max_exp):
                        CONTRADICTION_ATTR[(q, "budget_pre")] += 1
                        return _reject("valuation_pre")

    # ── clone & apply ──
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
        DEPTH_STATS[ns.depth] += 1
        CLONE_STATS["productive"] += 1
        PENDING_SIZE_HIST[len(ns.pending)] += 1
        CASCADE_DEPTH_HIST[0] += 1
        _record_productive_telemetry(ns)
        return ns

    # ── infinite-power: skip factorisation for massive p^a ──
    if is_prime_infinite(p, exp):
        DEPTH_STATS[ns.depth] += 1
        CLONE_STATS["productive"] += 1
        PENDING_SIZE_HIST[len(ns.pending)] += 1
        CASCADE_DEPTH_HIST[0] += 1
        _record_productive_telemetry(ns)
        return ns

    # factor-chain propagation (additive valuation)
    cascade_steps = 0
    for q, e in factorize(int(sig)):
        if q == 2:
            continue
        PROPAGATION_EDGES[(p, q)] += 1
        if q in ns.excluded:
            CONTRADICTION_ATTR[(q, "excluded_post")] += 1
            return _reject("valuation_post")

        ns.required_v[q] = ns.required_v.get(q, 0) + e

        if q in ns.assigned:
            if ns.required_v[q] > ns.current_v[q]:
                CONTRADICTION_ATTR[(q, "overrun_post")] += 1
                return _reject("valuation_post")
        else:
            if ns.required_v[q] > _max_possible_valuation(q, ns.euler_prime,
                                                          max_exp):
                CONTRADICTION_ATTR[(q, "budget_post")] += 1
                return _reject("valuation_post")

        if ns.required_v[q] > ns.current_v.get(q, 0):
            _enqueue_pending(ns, q)
            cascade_steps += 1

    DEPTH_STATS[ns.depth] += 1
    CLONE_STATS["productive"] += 1
    PENDING_SIZE_HIST[len(ns.pending)] += 1
    CASCADE_DEPTH_HIST[cascade_steps] += 1
    _record_productive_telemetry(ns)
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


# ── productive telemetry recording ───────────────────────────

def _record_productive_telemetry(ns) -> None:
    """Record ratio headroom and depth×|f| for a productive state."""
    ratio = float(ns.ratio_num) / float(ns.ratio_den)
    headroom = TARGET_NUM / TARGET_DEN - ratio
    if headroom <= 1e-6:       bucket = "<1e-6"
    elif headroom <= 1e-5:     bucket = "1e-6-1e-5"
    elif headroom <= 1e-4:     bucket = "1e-5-1e-4"
    elif headroom <= 1e-3:     bucket = "1e-4-1e-3"
    elif headroom <= 1e-2:     bucket = "1e-3-1e-2"
    else:                      bucket = ">1e-2"
    RATIO_HEADROOM[bucket] += 1
    DEPTH_FACTOR_MAP[(ns.depth, len(ns.assigned))] += 1
    HEADROOM_BY_FACTOR[(len(ns.assigned), bucket)] += 1
    # coarse headroom for signature dedup
    if headroom > 0:
        coarse = int(math.floor(-math.log10(headroom)))
    else:
        coarse = 99
    OBLIGATION_SIGS[(frozenset(ns.pending), len(ns.assigned), coarse)] += 1


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
