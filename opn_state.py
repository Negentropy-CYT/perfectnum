"""
opn_state — polymorphic search states and constraint propagation.

Defines a compact DFS state for Descartes-spoof search and a full chain
state for best-first OPN propagation. Assignment functions enforce Euler,
Touchard, ratio, and additive q-adic valuation constraints.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Deque, Dict, Optional, Tuple

from gmpy2 import mpz

from opn_core import (
    FERMAT_PRIMES,
    Q3_PREPOOL_MODE,
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
    sigma_v3_valuation,
    sigma_valuation_map,
    sigma_valuation_from_order,
    sigma_prime_power,
    valid_euler_exponents,
    valid_even_exponents,
)
from opn_metrics import (
    CloneEffect,
    PruneMechanism,
    PruneReason,
    RunMetrics,
    VALUATION_EXCLUDED,
    VALUATION_OVERRUN,
    VALUATION_BUDGET,
)


# ── centralized valuation contradiction recorder ─────────────

def _record_valuation_contradiction(
    metrics: RunMetrics,
    *,
    q: int,
    source_exp: int,
    kind: int,
    phase: str,
) -> None:
    """Record one valuation contradiction with both attribution and
    per-exponent structural counters in a single call site."""
    if kind == VALUATION_EXCLUDED:
        reason_name = f"excluded_{phase}"
    elif kind == VALUATION_OVERRUN:
        reason_name = f"overrun_{phase}"
    elif kind == VALUATION_BUDGET:
        reason_name = f"budget_{phase}"
    else:
        raise ValueError(
            f"unknown valuation contradiction kind: {kind}"
        )

    structure = metrics.structure
    structure.contradiction_attribution[
        (q, reason_name)
    ] += 1

    if source_exp < len(structure.valuation_contradictions_by_exp):
        structure.valuation_contradictions_by_exp[source_exp][kind] += 1
        if q == 3:
            structure.valuation_q3_by_exp[source_exp] += 1


def _valuation_conflict_kind(
    st: "ChainState",
    *,
    q: int,
    incoming: int,
    max_exp: int,
) -> int | None:
    """Return the contradiction kind if *q* owes an unpayable valuation debt."""
    if incoming <= 0:
        return None

    offset = _target_valuation_offset(q)
    new_required = st.required_v.get(q, 0) + incoming

    if q in st.excluded and new_required > offset:
        return VALUATION_EXCLUDED

    if q in st.assigned:
        if new_required > st.current_v[q] + offset:
            return VALUATION_OVERRUN
        return None

    maximum = _max_possible_valuation(
        q,
        st.euler_prime,
        max_exp,
    )

    if new_required > maximum + offset:
        return VALUATION_BUDGET

    return None


def _q3_prepool_conflict(
    st: "ChainState",
    *,
    p: int,
    exp: int,
    max_exp: int,
) -> tuple[int, int] | None:
    """Return (kind, incoming) if the q=3 LTE valuation is contradictory."""
    incoming = sigma_v3_valuation(p, exp)
    kind = _valuation_conflict_kind(
        st,
        q=3,
        incoming=incoming,
        max_exp=max_exp,
    )
    if kind is None:
        return None
    return kind, incoming


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
    p = int(p)  # normalise numpy unsigned → Python int
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
    p = int(p)  # normalise numpy unsigned → Python int (metrics keys, gmpy2)
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

    # ── q=3 prepool valuation (LTE, O(1), before the pool analyser) ──
    q3_shadow = None
    if (
        propagate
        and SEARCH_MODE.target_num == 2
        and SEARCH_MODE.target_den == 1
        and SEARCH_MODE.require_euler
        and Q3_PREPOOL_MODE != "off"
    ):
        q3_shadow = _q3_prepool_conflict(
            st,
            p=p,
            exp=exp,
            max_exp=max_exp,
        )

        if q3_shadow is not None:
            kind, _incoming = q3_shadow

            if Q3_PREPOOL_MODE == "enforce":
                _record_valuation_contradiction(
                    metrics,
                    q=3,
                    source_exp=exp,
                    kind=kind,
                    phase="pre",
                )
                metrics.performance.q3_prepool_prunes += 1
                if exp < len(metrics.performance.q3_prepool_prunes_by_exp):
                    metrics.performance.q3_prepool_prunes_by_exp[exp] += 1
                return _reject(
                    metrics,
                    reason=PruneReason.VALUATION_CONTRADICTION,
                    mechanism=PruneMechanism.ORDER_LTE_PRECHECK,
                    clone_effect=CloneEffect.AVOIDED,
                )

            metrics.performance.q3_prepool_shadow_hits += 1

    # ── pre-clone valuation check (populates the exact map lazily) ──
    pre_vals = None
    if propagate:
        if sigma_pool_analyzer is not None:
            analysis = sigma_pool_analyzer.analyze(p, exp)

            # ── q=3 shadow verification ──
            if q3_shadow is not None:
                predicted_kind, predicted_incoming = q3_shadow
                observed_incoming = analysis.valuations.get(3, 0)
                observed_kind = _valuation_conflict_kind(
                    st,
                    q=3,
                    incoming=observed_incoming,
                    max_exp=max_exp,
                )
                perf = metrics.performance
                if (
                    observed_incoming != predicted_incoming
                    or observed_kind != predicted_kind
                ):
                    perf.q3_prepool_shadow_mismatches += 1
                elif analysis.exact:
                    perf.q3_prepool_shadow_exact += 1
                else:
                    perf.q3_prepool_shadow_outside += 1

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
                _record_valuation_contradiction(
                    metrics,
                    q=q,
                    source_exp=exp,
                    kind=VALUATION_EXCLUDED,
                    phase="pre",
                )
                return _reject(
                    metrics,
                    reason=PruneReason.VALUATION_CONTRADICTION,
                    mechanism=PruneMechanism.PRECLONE_VALUATION,
                    clone_effect=CloneEffect.AVOIDED,
                )
            if q in st.assigned:
                if new_req > st.current_v[q] + offset:
                    _record_valuation_contradiction(
                        metrics,
                        q=q,
                        source_exp=exp,
                        kind=VALUATION_OVERRUN,
                        phase="pre",
                    )
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
                    _record_valuation_contradiction(
                        metrics,
                        q=q,
                        source_exp=exp,
                        kind=VALUATION_BUDGET,
                        phase="pre",
                    )
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
    post_vals = pre_vals if pre_vals is not None else {}
    for q, incoming in post_vals.items():
        if q == 2:
            continue
        metrics.structure.propagation_exp_edges[(p, exp, q)] += 1
        offset = _target_valuation_offset(q)
        new_req = ns.required_v.get(q, 0) + incoming
        if q in ns.excluded and new_req > offset:
            _record_valuation_contradiction(
                metrics,
                q=q,
                source_exp=exp,
                kind=VALUATION_EXCLUDED,
                phase="post",
            )
            return _reject(
                metrics,
                reason=PruneReason.VALUATION_CONTRADICTION,
                mechanism=PruneMechanism.POSTCLONE_VALUATION,
                clone_effect=CloneEffect.WASTED,
            )

        ns.required_v[q] = new_req

        if q in ns.assigned:
            if ns.required_v[q] > ns.current_v[q] + offset:
                _record_valuation_contradiction(
                    metrics,
                    q=q,
                    source_exp=exp,
                    kind=VALUATION_OVERRUN,
                    phase="post",
                )
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
                _record_valuation_contradiction(
                    metrics,
                    q=q,
                    source_exp=exp,
                    kind=VALUATION_BUDGET,
                    phase="post",
                )
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
