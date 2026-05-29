"""
opn_search — constraint-propagation factor-chain search engine.

Exposes ``search_opn()``, a generator that yields State objects for
each valid OPN or pseudo-OPN candidate discovered.

Supports two search strategies:
  - DFS (stack)   — for pseudo-solution (DFSState)
  - best-first    — for factor-chain true-OPN (ChainState)

Polymorphic dispatch on state type; Touchard congruence pruning;
optional contradiction-pattern learning cache.
"""
import gmpy2
import heapq
import time
from typing import Callable, FrozenSet, List, Optional, Tuple, Union

from gmpy2 import mpz

from opn_core import (
    CLONE_STATS,
    EXCLUDE_EXP_4,
    EXP4_FILTER_HITS,
    SEARCH_MODE,
    HEAP_MAX_SIZE,
    MAX_PRIME,
    PROGRESS_INTERVAL,
    PRUNE_STATS,
    TOXIC_SKIP,
    compute_exclude_exp4,
    check_touchard,
    next_prime_lower_bound,
    next_prime_upper_bound,
    precompute_sig_factors,
    precompute_suffix_bounds,
    ratio_lower_bound,
    ratio_upper_bound,
    sigma_prime_power,
    power_pa,
    touchard_force_3,
    valid_euler_exponents,
    valid_even_exponents,
)
from opn_state import (
    ChainState,
    DFSState,
    FeasibilityCache,
    _compute_priority,
    _enqueue_pending,
    _max_possible_valuation,
    assign_prime_chain,
    assign_prime_dfs,
)

State = Union[DFSState, ChainState]
"""Type alias covering both concrete state classes."""


# ══════════════════════════════════════════════════════════════
# Contradiction Learning Cache
# ══════════════════════════════════════════════════════════════

class ContradictionCache:
    """Remembers valuation-deficit patterns that lead to contradiction.

    Two-tier lookup:
      1. Exact-match set — O(1) hash lookup for identical patterns.
      2. Subsumption list — O(n) scan checking whether current state
         is provably worse than a cached contradiction.

    Only caches contradictions at depth >= MIN_CACHE_DEPTH (deeper
    patterns generalise better).  Bounded by CACHE_MAX_SIZE with
    FIFO eviction.  Disabled by default — enable via *use_cache*.
    """

    CACHE_MAX_SIZE  = 50_000
    MIN_CACHE_DEPTH = 3

    def __init__(self):
        self._exact: set = set()
        self._patterns: List[Tuple[FrozenSet[int], FrozenSet[int],
                                     FrozenSet[Tuple[int, int]]]] = []

    def add(self, assigned_keys, excluded_keys, deficit_items):
        entry = (assigned_keys, excluded_keys, deficit_items)
        self._exact.add(entry)
        if len(self._patterns) >= self.CACHE_MAX_SIZE:
            self._patterns.pop(0)   # FIFO eviction
        self._patterns.append(entry)

    def is_subsumed(self, assigned_keys, excluded_keys,
                    required_v, current_v):
        # Tier 1: exact match
        cur_def = _make_deficit_frozenset(required_v, current_v)
        exact = (assigned_keys, excluded_keys, cur_def)
        if exact in self._exact:
            return True

        # Tier 2: subsumption scan
        for a_keys, e_keys, def_items in self._patterns:
            if not a_keys.issubset(assigned_keys):
                continue
            if not e_keys.issubset(excluded_keys):
                continue
            ok = True
            for q, min_def in def_items:
                if required_v.get(q, 0) - current_v.get(q, 0) < min_def:
                    ok = False
                    break
            if ok:
                return True
        return False

    def __len__(self):
        return len(self._patterns)


def _make_deficit_frozenset(required_v, current_v):
    items = []
    for q, req in required_v.items():
        deficit = req - current_v.get(q, 0)
        if deficit > 0:
            items.append((q, deficit))
    return frozenset(items)


# ══════════════════════════════════════════════════════════════
# Verification & pseudo check
# ══════════════════════════════════════════════════════════════

def _verify_solution(st: State) -> bool:
    lhs = mpz(1)
    rhs = mpz(1)
    for p, a in st.assigned.items():
        lhs *= sigma_prime_power(p, a)
        rhs *= mpz(power_pa(p, a))
    return lhs == 2 * rhs


def _check_pseudo(st: State) -> bool:
    if SEARCH_MODE.target_num != 2 or SEARCH_MODE.target_den != 1:
        return False  # pseudo-solutions formula assumes target = 2/1
    if len(st.assigned) < 1 or st.ratio_num * SEARCH_MODE.target_den >= SEARCH_MODE.target_num * st.ratio_den:
        return False
    if 10 * st.ratio_num < 19 * st.ratio_den:
        return False
    denom = 2 * st.ratio_den - st.ratio_num
    if denom <= 0:
        return False
    if not gmpy2.is_divisible(st.ratio_num, denom):
        return False
    r = st.ratio_num // denom
    if r <= 1:
        return False
    for p in st.assigned:
        if r % p == 0:
            return False
    st.pseudo = True
    return True


# ══════════════════════════════════════════════════════════════
# Main search
# ══════════════════════════════════════════════════════════════

def search_opn(
    primes: List[int],
    max_factors: int,
    max_exp: int,
    state_holder: Optional[dict] = None,
    resume_state: Optional[dict] = None,
    *,
    propagate: bool = True,
    progress_callback=None,
    use_cache: bool = False,
):
    """Generator yielding State objects for each candidate found.

    *propagate* selects DFSState (False) or ChainState (True).
    *use_cache* enables the contradiction learning cache (chain mode).
    """
    n = len(primes)
    cache = ContradictionCache() if use_cache else None
    feas_cache = FeasibilityCache() if use_cache else None
    use_heap = propagate

    # ── precompute suffix bounds (O(1) ratio queries) ──
    s_ub_num, s_ub_den, s_lb_num, s_lb_den = precompute_suffix_bounds(primes)

    if propagate:
        precompute_sig_factors(primes, max_exp)
        compute_exclude_exp4(primes, max_exp, MAX_PRIME)

    if resume_state is not None:
        heap = resume_state["heap"]
        total_states = resume_state["total_states"]
        heap_counter = resume_state.get("heap_counter", total_states)
        elapsed_offset = resume_state["elapsed"]
        use_heap = resume_state.get("use_heap", use_heap)
    else:
        if use_heap:
            init_st = ChainState()
            init_st.priority = _compute_priority(mpz(1), mpz(1), 0.0, 0)
            heap = [(init_st.priority, 0, init_st)]
            heap_counter = 1
        else:
            init_st = DFSState()
            heap = [init_st]
            heap_counter = 0
        total_states = 0
        elapsed_offset = 0.0

    # ── search-mode initialisation: forced/excluded primes ──
    if not resume_state:
        st0 = heap[0][2] if use_heap else heap[0]
        st0.excluded.update(SEARCH_MODE.excluded_primes)
        for q, _ in SEARCH_MODE.forced_primes.items():
            if use_heap:
                _enqueue_pending(st0, q)
            else:
                raise NotImplementedError(
                    "DFS mode does not support forced_primes.  Set PROPAGATE=True.")

    # ── push / pop helpers ──
    def _push(container, st):
        nonlocal heap_counter
        if use_heap:
            heapq.heappush(container, (st.priority, heap_counter, st))
            heap_counter += 1
        else:
            container.append(st)

    def _pop(container):
        if use_heap:
            return heapq.heappop(container)[2]
        return container.pop()

    def _snapshot(container):
        if use_heap:
            return [(s.priority, i, s) for i, (_, _, s) in enumerate(container)]
        return list(container)

    t0 = time.time() - elapsed_offset

    if state_holder is not None:
        state_holder["primes"]       = primes
        state_holder["max_factors"]  = max_factors
        state_holder["max_exp"]      = max_exp
        state_holder["heap"]         = _snapshot(heap)
        state_holder["heap_counter"] = heap_counter
        state_holder["total_states"] = total_states
        state_holder["use_heap"]     = use_heap

    # ── initial progress (trigger on the very first state) ──
    _first = True

    # ── main loop ──
    while heap:
        if use_heap and len(heap) > HEAP_MAX_SIZE and total_states % 1000 == 0:
            keep = int(HEAP_MAX_SIZE * 0.6)
            heap = heapq.nsmallest(keep, heap)
            heapq.heapify(heap)

        st = _pop(heap)

        if state_holder is not None:
            front = [(st.priority, heap_counter, st)] if use_heap else [st]
            state_holder["heap"]         = _snapshot(front + heap)
            state_holder["heap_counter"]  = heap_counter
            state_holder["total_states"]  = total_states
            state_holder["elapsed"]       = time.time() - t0

        total_states += 1
        if progress_callback is not None:
            if total_states == 1 or total_states % PROGRESS_INTERVAL == 0:
                progress_callback(total_states, st, time.time() - t0)
                _first = False

        # ── true-OPN check ──
        if (
            st.ratio_num * SEARCH_MODE.target_den == SEARCH_MODE.target_num * st.ratio_den
            and (not SEARCH_MODE.require_euler or st.euler_prime is not None)
            and len(st.assigned) >= 2
            and (not SEARCH_MODE.require_euler or check_touchard(st.euler_prime, st.assigned, st.excluded))
            and _verify_solution(st)
        ):
            st.pseudo = False
            yield st
            continue

        # ── pseudo-solution check ──
        if _check_pseudo(st):
            yield st
            continue

        # ── pruning ──
        if st.ratio_num * SEARCH_MODE.target_den >= SEARCH_MODE.target_num * st.ratio_den:
            continue
        if len(st.assigned) >= max_factors:
            continue

        # feasibility cache check: same deficit pattern failed before?
        if feas_cache is not None and use_heap:
            if feas_cache.contains(st.required_v, st.current_v):
                continue

        lb_num, lb_den = ratio_lower_bound(
            st.ratio_num, st.ratio_den,
            st.pending if use_heap else [],
        )
        if lb_num * SEARCH_MODE.target_den > SEARCH_MODE.target_num * lb_den:
            continue

        ub_num, ub_den = ratio_upper_bound(
            st.ratio_num, st.ratio_den,
            st.assigned, st.excluded, primes,
            next_idx=st.next_idx,
            suffix_ub_num=s_ub_num, suffix_ub_den=s_ub_den,
        )
        if ub_num * SEARCH_MODE.target_den < SEARCH_MODE.target_num * ub_den:
            if cache is not None and len(st.assigned) >= 3:
                cache.add(frozenset(st.assigned.keys()),
                          frozenset(st.excluded), frozenset())
            if feas_cache is not None:
                feas_cache.add(st.required_v, st.current_v)
            continue

        # ── Touchard: force 3 ──
        if use_heap and touchard_force_3(st.euler_prime, st.assigned,
                                         st.excluded):
            _enqueue_pending(st, 3)

        # ── contradiction cache check ──
        if use_heap and cache is not None and len(st.assigned) >= 3:
            if cache.is_subsumed(
                frozenset(st.assigned.keys()),
                frozenset(st.excluded),
                st.required_v,
                st.current_v,
            ):
                continue

        # ── pending (chain mode) ──
        if use_heap:
            if _drain_and_process_pending(
                st, heap, primes, max_exp, _push, cache,
            ):
                continue

        # ── expansion ──
        # interval bounds (chain mode): skip primes provably too small / large
        # Upper bound is only reliable when few primes remain or ratio is close
        # to target.  Otherwise the bound collapses to ~1, blocking all expansion.
        lo = hi = 0
        k_remain = max_factors - len(st.assigned)
        if use_heap:
            lo = next_prime_lower_bound(st.ratio_num, st.ratio_den,
                                        SEARCH_MODE.target_num, SEARCH_MODE.target_den)
            if k_remain <= 4:
                hi = next_prime_upper_bound(st.ratio_num, st.ratio_den,
                                            st.next_idx, SEARCH_MODE.target_num, SEARCH_MODE.target_den,
                                            s_ub_num, s_ub_den, n)
        idx = st.next_idx
        while idx < n:
            p = primes[idx]
            if p in st.assigned or p in st.excluded:
                idx += 1
                continue
            if lo > 0 and p < lo:
                PRUNE_STATS["interval_lo"] += 1
                idx += 1; continue   # too small to reach target
            if hi > 0 and p > hi:
                PRUNE_STATS["interval_hi"] += 1
                break                 # exceeded interval upper bound

            # skip branch
            skip_st = st.clone()
            skip_st.excluded.add(p)
            skip_st.next_idx = idx + 1
            _push(heap, skip_st)

            # Euler-include
            if st.euler_prime is None and p % 4 == 1:
                for e in reversed(valid_euler_exponents(1, max_exp)):
                    ns = _assign(st, p, e, use_heap, propagate, max_exp)
                    if ns is not None:
                        ns.next_idx = idx + 1
                        _push(heap, ns)

            # non-Euler include
            for e in reversed(valid_even_exponents(2, max_exp)):
                if e == 4 and use_heap and p in EXCLUDE_EXP_4:
                    PRUNE_STATS["exp4_filtered"] += 1
                    CLONE_STATS["saved"] += 1
                    EXP4_FILTER_HITS[p] += 1
                    continue
                ns = _assign(st, p, e, use_heap, propagate, max_exp)
                if ns is not None:
                    ns.next_idx = idx + 1
                    _push(heap, ns)

            break

    # ── exhausted ──
    elapsed = time.time() - t0
    print()  # end inline progress line cleanly
    print(f"搜索完成: {total_states:,} states, {elapsed:.1f}s")
    if state_holder is not None:
        state_holder["heap"]         = []
        state_holder["heap_counter"]  = heap_counter
        state_holder["total_states"]  = total_states
        state_holder["elapsed"]       = elapsed


# ── polymorphic assign dispatch ──────────────────────────────

def _assign(st: State, p: int, exp: int, use_heap: bool,
            propagate: bool, max_exp: int) -> Optional[State]:
    """Dispatch to DFSState or ChainState assign function."""
    if use_heap:
        return assign_prime_chain(st, p, exp, propagate=propagate,
                                   max_exp=max_exp)
    else:
        return assign_prime_dfs(st, p, exp, max_exp=max_exp)


# ── pending processing (chain mode) ──────────────────────────

def _drain_and_process_pending(
    st: ChainState, heap, primes, max_exp: int, _push,
    cache,
) -> bool:
    if not st.pending:
        return False

    while st.pending and (
        st.pending[0] in st.assigned
        or st.pending[0] > primes[-1]
    ):
        q = st.pending.popleft()
        st.pending_set.discard(q)
        if q > primes[-1]:
            PRUNE_STATS["maxprime"] += 1
            return True   # forced prime beyond window → prune state

    if not st.pending:
        return False

    q = st.pending.popleft()
    st.pending_set.discard(q)
    lb = max(st.required_v.get(q, 1) - st.current_v.get(q, 0), 1)

    if st.euler_prime is None and q % 4 == 1:
        for e in reversed(valid_euler_exponents(lb, max_exp)):
            ns = assign_prime_chain(st, q, e, propagate=True, max_exp=max_exp)
            if ns is not None:
                _push(heap, ns)

    for e in reversed(valid_even_exponents(lb, max_exp)):
        ns = assign_prime_chain(st, q, e, propagate=True, max_exp=max_exp)
        if ns is not None:
            _push(heap, ns)

    return True
