"""
opn_search — constraint-propagation factor-chain search engine.

Exposes ``search_opn()``, a generator that yields State objects for
each valid OPN or pseudo-OPN candidate discovered.

Supports two search strategies:
  - DFS (stack)   — for pseudo-solution (DFSState)
  - best-first    — for factor-chain true-OPN (ChainState)

Polymorphic dispatch on state type; Touchard congruence pruning;
optional exact-state deduplication.
"""
import gmpy2
import heapq
import time
from typing import List, Optional, Union

from gmpy2 import mpz

from opn_core import (
    CHECKPOINT_INTERVAL_SECONDS,
    CLONE_STATS,
    EXCLUDE_EXP_4,
    EXP4_FILTER_HITS,
    SEARCH_MODE,
    PRUNE_STATS,
    TOXIC_SKIP,
    check_touchard,
    exp4_forced_outside_window,
    next_prime_lower_bound,
    next_prime_upper_bound,
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
    _compute_priority,
    _enqueue_pending,
    _max_possible_valuation,
    _target_valuation_offset,
    assign_prime_chain,
    assign_prime_dfs,
    fermat_debt_capacity,
)

State = Union[DFSState, ChainState]
"""Type alias covering both concrete state classes."""


class SearchStopped(RuntimeError):
    """Raised after a cooperative stop reaches a stable search boundary."""


def _state_signature(st: State) -> tuple:
    """Canonical key for sound exact-state deduplication."""
    common = (
        tuple(sorted(st.assigned.items())),
        frozenset(st.excluded),
        st.euler_prime,
        int(st.ratio_num),
        int(st.ratio_den),
        st.next_idx,
    )
    if isinstance(st, ChainState):
        return common + (
            tuple(sorted(st.required_v.items())),
            tuple(sorted(st.current_v.items())),
            tuple(st.pending),
        )
    return common


def _heap_snapshot(entries) -> list:
    """Copy entries into a valid heap without mutating the live frontier."""
    snapshot = list(entries)
    heapq.heapify(snapshot)
    return snapshot


# ══════════════════════════════════════════════════════════════
# Verification & pseudo check
# ══════════════════════════════════════════════════════════════

def _verify_solution(st: State) -> bool:
    lhs = mpz(1)
    rhs = mpz(1)
    for p, a in st.assigned.items():
        lhs *= sigma_prime_power(p, a)
        rhs *= mpz(power_pa(p, a))
    return lhs * SEARCH_MODE.target_den == SEARCH_MODE.target_num * rhs


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
    checkpoint_callback=None,
    checkpoint_interval_seconds: Optional[float] = CHECKPOINT_INTERVAL_SECONDS,
    stop_requested=None,
    use_cache: bool = False,
):
    """Generator yielding State objects for each candidate found.

    *propagate* selects DFSState (False) or ChainState (True).
    *use_cache* enables sound exact-state deduplication.
    *checkpoint_callback* runs synchronously at stable frontier boundaries.
    *stop_requested* is polled between fully processed states.
    """
    n = len(primes)
    seen_states = set() if use_cache else None
    use_heap = propagate

    print("using exact factor-slot tail bounds", flush=True)

    if propagate:
        EXCLUDE_EXP_4.clear()
        print("sigma-factor maps will be populated lazily")

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

    t0 = time.time() - elapsed_offset

    if state_holder is not None:
        state_holder["primes"]       = primes
        state_holder["max_factors"]  = max_factors
        state_holder["max_exp"]      = max_exp
        state_holder["use_heap"]     = use_heap

    snapshot_id = (
        int(resume_state.get("snapshot_id", 0))
        if resume_state is not None else 0
    )
    last_checkpoint = time.monotonic()

    def _publish_frontier(reason: str) -> None:
        """Expose a coherent frontier while the search loop is paused."""
        nonlocal snapshot_id, last_checkpoint
        if state_holder is None:
            last_checkpoint = time.monotonic()
            return
        snapshot_id += 1
        snapshot_elapsed = time.time() - t0
        # The callback runs synchronously before this live frontier mutates.
        state_holder.update({
            "heap": heap,
            "heap_counter": heap_counter,
            "total_states": total_states,
            "elapsed": snapshot_elapsed,
            "snapshot_id": snapshot_id,
            "snapshot_reason": reason,
            "frontier_size": len(heap),
            "live_total_states": total_states,
            "live_elapsed": snapshot_elapsed,
        })
        if checkpoint_callback is not None:
            checkpoint_callback(state_holder, reason)
        last_checkpoint = time.monotonic()

    _publish_frontier("initial")

    # ── progress (time-based: ~1 Hz, plus first state) ──
    _last_progress = 0.0

    # ── main loop ──
    while heap:
        if stop_requested is not None and stop_requested():
            _publish_frontier("stop")
            raise SearchStopped("search stopped at a stable frontier boundary")

        if (
            checkpoint_interval_seconds is not None
            and checkpoint_interval_seconds > 0
            and time.monotonic() - last_checkpoint >= checkpoint_interval_seconds
        ):
            _publish_frontier("periodic")

        st = _pop(heap)

        if seen_states is not None:
            signature = _state_signature(st)
            if signature in seen_states:
                PRUNE_STATS["exact_duplicate"] += 1
                continue
            seen_states.add(signature)

        total_states += 1
        if state_holder is not None:
            state_holder["live_total_states"] = total_states
            state_holder["live_elapsed"] = time.time() - t0
            state_holder["frontier_size"] = len(heap)
        if progress_callback is not None:
            elapsed = time.time() - t0
            if total_states == 1 or elapsed - _last_progress >= 1.0:
                progress_callback(total_states, st, elapsed)
                _last_progress = elapsed

        # ── true-OPN check ──
        if (
            st.ratio_num * SEARCH_MODE.target_den == SEARCH_MODE.target_num * st.ratio_den
            and (not SEARCH_MODE.require_euler or st.euler_prime is not None)
            and len(st.assigned) >= 2
            and (not SEARCH_MODE.require_euler or check_touchard(st.euler_prime, st.assigned, st.excluded))
            and _verify_solution(st)
        ):
            st.pseudo = False
            _publish_frontier("solution")
            yield st
            continue

        # ── pseudo-solution check ──
        if _check_pseudo(st):
            _publish_frontier("solution")
            yield st
            continue

        # ── pruning ──
        if st.ratio_num * SEARCH_MODE.target_den >= SEARCH_MODE.target_num * st.ratio_den:
            continue
        if len(st.assigned) >= max_factors:
            continue

        k_remain = max_factors - len(st.assigned)

        # Touchard can force 3 even when it lies before next_idx. Enqueue it
        # before both ratio bounds so the mandatory component is never omitted.
        if use_heap and touchard_force_3(st.euler_prime, st.assigned,
                                         st.excluded):
            _enqueue_pending(st, 3)

        live_pending = (
            {q for q in st.pending if q not in st.assigned}
            if use_heap else set()
        )
        if use_heap:
            if any(q > primes[-1] for q in live_pending):
                PRUNE_STATS["maxprime"] += 1
                continue
            if live_pending & st.excluded:
                PRUNE_STATS["pending_excluded"] += 1
                continue
            if len(live_pending) > k_remain:
                PRUNE_STATS["factor_slots"] += 1
                continue

        lb_num, lb_den = ratio_lower_bound(
            st.ratio_num, st.ratio_den,
            live_pending,
        )
        if lb_num * SEARCH_MODE.target_den > SEARCH_MODE.target_num * lb_den:
            continue

        ub_num, ub_den = ratio_upper_bound(
            st.ratio_num, st.ratio_den,
            st.assigned, st.excluded, primes,
            next_idx=st.next_idx,
            remaining_slots=k_remain,
            pending=live_pending,
        )
        if ub_num * SEARCH_MODE.target_den < SEARCH_MODE.target_num * ub_den:
            PRUNE_STATS["ratio_upper"] += 1
            continue

        if use_heap:
            debt_ok, _debt_detail = fermat_debt_capacity(
                st, primes, max_factors, max_exp,
            )
            if not debt_ok:
                PRUNE_STATS["fermat_debt"] += 1
                continue

        # ── pending (chain mode) ──
        if use_heap:
            if _drain_and_process_pending(
                st, heap, primes, max_exp, _push,
            ):
                continue

        # ── expansion ──
        # Interval bounds skip candidates that necessarily overshoot or cannot
        # reach the target even with the best remaining factor slots.
        lo = 0
        if use_heap:
            lo = next_prime_lower_bound(st.ratio_num, st.ratio_den,
                                        SEARCH_MODE.target_num, SEARCH_MODE.target_den)
        idx = st.next_idx
        while idx < n:
            p = primes[idx]
            if p in st.assigned or p in st.excluded:
                idx += 1
                continue
            if lo > 0 and p < lo:
                PRUNE_STATS["interval_lo"] += 1
                idx += 1; continue   # too small to reach target
            if use_heap:
                hi = next_prime_upper_bound(
                    st.ratio_num,
                    st.ratio_den,
                    idx,
                    k_remain,
                    SEARCH_MODE.target_num,
                    SEARCH_MODE.target_den,
                    primes,
                    st.assigned,
                    st.excluded,
                )
                if hi > 0 and p > hi:
                    PRUNE_STATS["interval_hi"] += 1
                    break

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
                if (
                    e == 4
                    and use_heap
                    and exp4_forced_outside_window(p, primes[-1])
                ):
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
    _publish_frontier("complete")


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
    lb = max(
        st.required_v.get(q, 0)
        - _target_valuation_offset(q)
        - st.current_v.get(q, 0),
        1,
    )

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
