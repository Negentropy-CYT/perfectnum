"""
opn_state — polymorphic search states and constraint propagation.

Defines two state classes:
  - DFSState   — minimal (5 fields) for Descartes-spoof DFS
  - ChainState — full (14 fields, 6 collections cloned) for factor-chain best-first search

Key improvements over v1 unified State:
  - DFSState.clone() copies 2 collections vs 7 → ~60% less overhead
  - Separate assign_prime_dfs / assign_prime_chain functions
  - Touchard congruence pruning integrated into both paths
  - Additive q-adic valuation contradiction detection (chain mode)
"""
import math
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Deque, Dict, Optional, Tuple

from gmpy2 import mpz

from opn_core import (
    FERMAT_PRIMES,
    SEARCH_MODE,
    _SIG_FACTORS,
    MAX_EXP,
    RESONANCE_REUSE_W,
    RESONANCE_NEWF_W,
    RESONANCE_GIANT_W,
    PRIORITY_RESONANCE_W,
    PRIORITY_DEPTH_W,
    check_touchard,
    power_pa,
    sigma_valuation_map,
    sigma_valuation_from_order,
    sigma_prime_power,
    touchard_force_3,
    valid_euler_exponents,
    valid_even_exponents,
)
from opn_metrics import (
    CloneEffect,
    PruneMechanism,
    PruneReason,
    RunMetrics,
)


# ── priority helper ──────────────────────────────────────────
def _compute_priority(ratio_num, ratio_den, resonance, n_assigned):
    target = SEARCH_MODE.target_num / SEARCH_MODE.target_den
    ratio = float(ratio_num / ratio_den)
    return (abs(target - ratio)
            - PRIORITY_RESONANCE_W * resonance
            - PRIORITY_DEPTH_W * n_assigned)


# ══════════════════════════════════════════════════════════════
# DFSState — minimal state for Descartes-spoof DFS
# ══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class DFSState:
    """Minimal state for DFS Descartes-spoof search (propagate=False).

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
    spoof:      bool           = False

    def clone(self) -> "DFSState":
        return DFSState(
            assigned=dict(self.assigned),
            excluded=set(self.excluded),
            euler_prime=self.euler_prime,
            ratio_num=mpz(self.ratio_num),
            ratio_den=mpz(self.ratio_den),
            next_idx=self.next_idx,
            depth=self.depth + 1,
            spoof=self.spoof,
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
    spoof:      bool            = False
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
            spoof=self.spoof,
            resonance=self.resonance,
            priority=self.priority,
        )


# ── prune telemetry helper ──────────────────────────────────

def _reject(
    metrics: RunMetrics,
    *,
    reason: PruneReason,
    mechanism: PruneMechanism,
    clone_effect: CloneEffect,
):
    """Record a prune event and return None (sentinel for pruned branch)."""
    metrics.record_prune(
        reason=reason,
        mechanism=mechanism,
        clone_effect=clone_effect,
    )
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
    return ratio_num * sig * SEARCH_MODE.target_den > SEARCH_MODE.target_num * ratio_den * pa


def _enqueue_pending(st, q):
    if q not in st.pending_set and q not in st.assigned:
        st.pending.append(q)
        st.pending_set.add(q)


def _target_valuation_offset(q: int) -> int:
    """Return v_q(target_num) - v_q(target_den)."""
    numerator = SEARCH_MODE.target_num
    denominator = SEARCH_MODE.target_den
    offset = 0
    while numerator % q == 0:
        numerator //= q
        offset += 1
    while denominator % q == 0:
        denominator //= q
        offset -= 1
    return offset


def _max_possible_valuation(q, euler_prime, max_exp):
    even_max = max(valid_even_exponents(2, max_exp), default=0)
    euler_max = max(valid_euler_exponents(1, max_exp), default=0)

    if not SEARCH_MODE.require_euler:
        return even_max
    if q == euler_prime:
        return euler_max
    if euler_prime is None and q % 4 == 1:
        return max(even_max, euler_max)
    return even_max


def valuation_debts(st: ChainState) -> Dict[int, int]:
    """Return odd-prime valuations that future sigma factors must supply.

    ``required_v[q]`` is the incoming valuation already supplied by processed
    components, while ``current_v[q]`` is the exponent chosen for q in N.
    Their positive difference is the reverse-valuation debt.
    """
    debts: Dict[int, int] = {}
    for q, exponent in st.current_v.items():
        debt = (
            exponent
            + _target_valuation_offset(q)
            - st.required_v.get(q, 0)
        )
        if debt > 0:
            debts[q] = debt
    return debts


@lru_cache(maxsize=None)
def _source_valuation_capacity(
    p: int, q: int, max_exp: int, allow_euler: bool,
) -> int:
    """Maximum q-adic valuation one future component p^a could supply."""
    exponents = valid_even_exponents(2, max_exp)
    if allow_euler and p % 4 == 1:
        exponents = exponents + valid_euler_exponents(1, max_exp)
    return max(
        (sigma_valuation_from_order(p, a, q) for a in exponents),
        default=0,
    )


@lru_cache(maxsize=None)
def _capacity_ranking(
    primes: tuple[int, ...],
    q: int,
    max_exp: int,
    allow_euler: bool,
) -> tuple[tuple[int, int], ...]:
    """Rank source primes once by their maximum contribution to q."""
    ranked = (
        (_source_valuation_capacity(p, q, max_exp, allow_euler), p)
        for p in primes
    )
    return tuple(sorted(ranked, reverse=True))


# ponytail: cached per (primes_tuple) — the prime list is constant once search begins
_prime_index_cache: Dict[tuple, Dict[int, int]] = {}

def fermat_debt_capacity(
    st: ChainState,
    primes,
    max_factors: int,
    max_exp: int,
) -> Tuple[bool, Optional[Tuple[int, int, int]]]:
    """Prove whether remaining slots can still pay valuation debts.

    For each future prime p, the exact order/LTE formula gives an upper bound
    on how much q-adic valuation p can supply over all allowed exponents.  The
    sum of the largest ``remaining_slots`` capacities is therefore an upper
    bound on every completion.  We intentionally allow more than one future
    component to use the Euler exponent while forming this bound; that only
    makes the bound larger and keeps the prune conservative.

    All Fermat-prime debts are always checked.  When ``slots > 2`` only the
    top-3 non-Fermat debts are added to limit cost.

    Returns ``(False, (q, debt, capacity))`` only after a rigorous capacity
    shortfall has been proved.
    """
    if (
        not SEARCH_MODE.require_euler
        or SEARCH_MODE.target_num != 2
        or SEARCH_MODE.target_den != 1
    ):
        return True, None

    slots = max_factors - len(st.assigned)
    allow_euler = st.euler_prime is None
    prime_tuple = tuple(primes)
    unavailable = st.assigned.keys() | st.excluded

    # prime index: cached once per primes list
    if prime_tuple not in _prime_index_cache:
        _prime_index_cache[prime_tuple] = {p: i for i, p in enumerate(prime_tuple)}
    prime_idx = _prime_index_cache[prime_tuple]

    all_debts = valuation_debts(st)

    # always check all Fermat debts (preserve existing prune strength)
    fermat_items = [(q, d) for q, d in all_debts.items() if q in FERMAT_PRIMES]

    # selectively check non-Fermat debts
    nonfermat_items = [
        (q, d) for q, d in all_debts.items() if q not in FERMAT_PRIMES
    ]
    if slots > 2:
        nonfermat_items.sort(key=lambda item: -item[1])
        nonfermat_items = nonfermat_items[:3]

    for q, debt in fermat_items + nonfermat_items:
        capacity = 0
        selected = 0
        for contribution, p in _capacity_ranking(
            prime_tuple, q, max_exp, allow_euler,
        ):
            if contribution == 0 or selected >= max(slots, 0):
                break
            if p in unavailable:
                continue

            # With one slot left a prime before the cursor that is not
            # already pending cannot be both introduced and assigned:
            # introducing it via σ(s^a) and then assigning it needs ≥2 slots.
            if (
                slots == 1
                and p not in st.pending_set
                and prime_idx[p] < st.next_idx
            ):
                continue

            capacity += contribution
            selected += 1
        if capacity < debt:
            return False, (q, debt, capacity)
    return True, None


# ── assign_prime_dfs (lightweight, no propagation) ───────────

def assign_prime_dfs(
    st: DFSState,
    p: int,
    exp: int,
    *,
    metrics: RunMetrics,
    max_exp: int = MAX_EXP,
) -> Optional[DFSState]:
    """Assign p^exp to a DFSState.  No factor-chain propagation."""
    if p in st.excluded or p in st.assigned:
        return _reject(
            metrics,
            reason=PruneReason.EXCLUDED_PRIME,
            mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
            clone_effect=CloneEffect.AVOIDED,
        )
    if _early_ratio_prune(st.ratio_num, st.ratio_den, p, exp):
        return _reject(
            metrics,
            reason=PruneReason.RATIO_OVERSHOOT,
            mechanism=PruneMechanism.PROSPECTIVE_RATIO,
            clone_effect=CloneEffect.AVOIDED,
        )
    if not _euler_ok(p, exp, st.euler_prime):
        return _reject(
            metrics,
            reason=PruneReason.EULER_FORM,
            mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
            clone_effect=CloneEffect.AVOIDED,
        )

    ns = st.clone()
    metrics.record_clone(len(st.assigned))
    ns.assigned[p] = exp
    sig = sigma_prime_power(p, exp)
    pa = mpz(power_pa(p, exp))
    ns.ratio_num = st.ratio_num * sig
    ns.ratio_den = st.ratio_den * pa
    if exp % 2 == 1:
        ns.euler_prime = p

    if not check_touchard(ns.euler_prime, ns.assigned, ns.excluded):
        return _reject(
            metrics,
            reason=PruneReason.TOUCHARD,
            mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
            clone_effect=CloneEffect.WASTED,
        )

    metrics.structure.record_productive(
        depth=ns.depth,
        assigned_count=len(ns.assigned),
        pending=(),
        ratio_num=int(ns.ratio_num),
        ratio_den=int(ns.ratio_den),
        target_num=SEARCH_MODE.target_num,
        target_den=SEARCH_MODE.target_den,
    )
    return ns


# ── assign_prime_chain (full factor-chain propagation) ───────

def assign_prime_chain(
    st: ChainState,
    p: int,
    exp: int,
    *,
    metrics: RunMetrics,
    propagate: bool = True,
    max_exp: int = MAX_EXP,
    prime_limit: int | None = None,
    sigma_pool_analyzer=None,
) -> Optional[ChainState]:
    """Assign p^exp to a ChainState with full factor-chain propagation.

    The exact sigma-valuation map is populated on demand before cloning, so
    valuation contradictions avoid both unnecessary clones and global eager
    factorisation of unreachable (p, a) pairs.
    """
    if p in st.excluded or p in st.assigned:
        return _reject(
            metrics,
            reason=PruneReason.EXCLUDED_PRIME,
            mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
            clone_effect=CloneEffect.AVOIDED,
        )
    if _early_ratio_prune(st.ratio_num, st.ratio_den, p, exp):
        return _reject(
            metrics,
            reason=PruneReason.RATIO_OVERSHOOT,
            mechanism=PruneMechanism.PROSPECTIVE_RATIO,
            clone_effect=CloneEffect.AVOIDED,
        )
    if not _euler_ok(p, exp, st.euler_prime):
        return _reject(
            metrics,
            reason=PruneReason.EULER_FORM,
            mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
            clone_effect=CloneEffect.AVOIDED,
        )

    # ── pre-clone valuation check (populates the exact map lazily) ──
    pre_vals = None
    if propagate:
        if sigma_pool_analyzer is not None:
            analysis = sigma_pool_analyzer.analyze(p, exp)
            if not analysis.exact:
                return _reject(
                    metrics,
                    reason=PruneReason.OUTSIDE_WINDOW,
                    mechanism=PruneMechanism.COLD_POOL_CERTIFICATE,
                    clone_effect=CloneEffect.AVOIDED,
                )
            pre_vals = analysis.valuations
        else:
            pre_vals = sigma_valuation_map(p, exp)

        for q, incoming in pre_vals.items():
            offset = _target_valuation_offset(q)
            new_req = st.required_v.get(q, 0) + incoming

            if prime_limit is not None and q > prime_limit and new_req > offset:
                metrics.structure.contradiction_attribution[
                    (q, "maxprime_pre")
                ] += 1
                metrics.structure.outside_window_sources[
                    (p, exp, q)
                ] += 1
                return _reject(
                    metrics,
                    reason=PruneReason.OUTSIDE_WINDOW,
                    mechanism=PruneMechanism.EXACT_FACTOR_OUTSIDE,
                    clone_effect=CloneEffect.AVOIDED,
                )

            if q in st.excluded and new_req > offset:
                metrics.structure.contradiction_attribution[
                    (q, "excluded_pre")
                ] += 1
                return _reject(
                    metrics,
                    reason=PruneReason.VALUATION_CONTRADICTION,
                    mechanism=PruneMechanism.PRECLONE_VALUATION,
                    clone_effect=CloneEffect.AVOIDED,
                )
            if q in st.assigned:
                if new_req > st.current_v[q] + offset:
                    metrics.structure.contradiction_attribution[
                        (q, "overrun_pre")
                    ] += 1
                    return _reject(
                        metrics,
                        reason=PruneReason.VALUATION_CONTRADICTION,
                        mechanism=PruneMechanism.PRECLONE_VALUATION,
                        clone_effect=CloneEffect.AVOIDED,
                    )
            else:
                if (
                    new_req
                    > _max_possible_valuation(q, st.euler_prime, max_exp)
                    + offset
                ):
                    metrics.structure.contradiction_attribution[
                        (q, "budget_pre")
                    ] += 1
                    return _reject(
                        metrics,
                        reason=PruneReason.VALUATION_CONTRADICTION,
                        mechanism=PruneMechanism.PRECLONE_VALUATION,
                        clone_effect=CloneEffect.AVOIDED,
                    )

    # ── clone & apply ──
    ns = st.clone()
    metrics.record_clone(len(st.assigned))
    ns.assigned[p] = exp
    ns.current_v[p] = ns.current_v.get(p, 0) + exp
    sig = sigma_prime_power(p, exp)
    pa = mpz(power_pa(p, exp))
    ns.ratio_num = st.ratio_num * sig
    ns.ratio_den = st.ratio_den * pa
    if exp % 2 == 1:
        ns.euler_prime = p

    if not check_touchard(ns.euler_prime, ns.assigned, ns.excluded):
        return _reject(
            metrics,
            reason=PruneReason.TOUCHARD,
            mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
            clone_effect=CloneEffect.WASTED,
        )

    # resonance (chain mode only)
    if propagate:
        _update_resonance(ns, st, p, exp)
        ns.priority = _compute_priority(
            ns.ratio_num, ns.ratio_den, ns.resonance, len(ns.assigned),
        )
    else:
        ns.priority = 0.0

    if not propagate:
        metrics.structure.record_productive(
            depth=ns.depth,
            assigned_count=len(ns.assigned),
            pending=ns.pending,
            ratio_num=int(ns.ratio_num),
            ratio_den=int(ns.ratio_den),
            target_num=SEARCH_MODE.target_num,
            target_den=SEARCH_MODE.target_den,
        )
        return ns

    # factor-chain propagation (additive valuation)
    cascade_steps = 0
    post_vals = pre_vals if pre_vals is not None else {}
    for q, incoming in post_vals.items():
        if q == 2:
            continue
        metrics.structure.propagation_edges[(p, q)] += 1
        metrics.structure.propagation_exp_edges[(p, exp, q)] += 1
        offset = _target_valuation_offset(q)
        new_req = ns.required_v.get(q, 0) + incoming
        if q in ns.excluded and new_req > offset:
            metrics.structure.contradiction_attribution[
                (q, "excluded_post")
            ] += 1
            return _reject(
                metrics,
                reason=PruneReason.VALUATION_CONTRADICTION,
                mechanism=PruneMechanism.POSTCLONE_VALUATION,
                clone_effect=CloneEffect.WASTED,
            )

        ns.required_v[q] = new_req

        if q in ns.assigned:
            if ns.required_v[q] > ns.current_v[q] + offset:
                metrics.structure.contradiction_attribution[
                    (q, "overrun_post")
                ] += 1
                return _reject(
                    metrics,
                    reason=PruneReason.VALUATION_CONTRADICTION,
                    mechanism=PruneMechanism.POSTCLONE_VALUATION,
                    clone_effect=CloneEffect.WASTED,
                )
        else:
            if (
                ns.required_v[q]
                > _max_possible_valuation(q, ns.euler_prime, max_exp)
                + offset
            ):
                metrics.structure.contradiction_attribution[
                    (q, "budget_post")
                ] += 1
                return _reject(
                    metrics,
                    reason=PruneReason.VALUATION_CONTRADICTION,
                    mechanism=PruneMechanism.POSTCLONE_VALUATION,
                    clone_effect=CloneEffect.WASTED,
                )

        if (
            ns.required_v[q] - offset
            > ns.current_v.get(q, 0)
        ):
            _enqueue_pending(ns, q)
            cascade_steps += 1

    metrics.structure.record_productive(
        depth=ns.depth,
        assigned_count=len(ns.assigned),
        pending=ns.pending,
        ratio_num=int(ns.ratio_num),
        ratio_den=int(ns.ratio_den),
        target_num=SEARCH_MODE.target_num,
        target_den=SEARCH_MODE.target_den,
    )
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
