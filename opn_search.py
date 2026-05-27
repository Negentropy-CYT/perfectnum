"""
opn_search — constraint-propagation factor-chain search engine.

Exposes ``search_opn()``, a generator that yields ``(State, bool)``
tuples for each valid odd-perfect-number or pseudo-OPN candidate
discovered.

Supports two search strategies, selected by the *propagate* flag:
  - DFS (stack)   — for independent-prime pseudo-solution search
  - best-first    — for factor-chain true-OPN search (guided by
                     resonance score and ratio proximity to 2)
"""

import gmpy2
import heapq
import time
from typing import List, Optional

from gmpy2 import mpz

from opn_core import (
    HEAP_MAX_SIZE,
    PROGRESS_INTERVAL,
    precompute_sig_factors,
    ratio_lower_bound,
    ratio_upper_bound,
    sigma_prime_power,
    power_pa,
    valid_euler_exponents,
    valid_even_exponents,
)
from opn_state import State, assign_prime, _compute_priority


# ── verification ──────────────────────────────────────────────
def _verify_solution(st: State) -> bool:
    """Check σ(N) == 2N by recomputing from scratch."""
    lhs = mpz(1)
    rhs = mpz(1)
    for p, a in st.assigned.items():
        lhs *= sigma_prime_power(p, a)
        rhs *= mpz(power_pa(p, a))
    return lhs == 2 * rhs


# ── pseudo-solution check ─────────────────────────────────────
def _check_pseudo(st: State) -> bool:
    """Test whether *st* admits a composite 'r' satisfying
       (r+1)·∏σ(p^a) = 2r·∏p^a.   Sets ``st.pseudo = True`` on match."""
    if len(st.assigned) < 1 or st.ratio_num >= 2 * st.ratio_den:
        return False
    # threshold: ratio must exceed 1.9 for r to be >= 19
    if 10 * st.ratio_num < 19 * st.ratio_den:
        return False
    denom = 2 * st.ratio_den - st.ratio_num
    if denom <= 0:
        return False
    # use C-level divisibility check (avoids constructing remainder object)
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


# ── main search ───────────────────────────────────────────────
def search_opn(
    primes: List[int],
    max_factors: int,
    max_exp: int,
    state_holder: Optional[dict] = None,
    resume_state: Optional[dict] = None,
    *,
    propagate: bool = True,
    progress_callback=None,
):
    """Generator that yields ``State`` objects for each candidate found.

    Parameters
    ----------
    primes:        sorted list of odd primes constituting the search alphabet.
    max_factors:   maximum number of distinct prime factors in N.
    max_exp:       maximum exponent to consider for any prime.
    state_holder:      mutable dict updated each iteration for checkpoint saves.
    resume_state:      dict from a previous checkpoint (or ``None``).
    propagate:         ``True``  → factor-chain true-OPN search (best-first).
                       ``False`` → independent-prime pseudo-solution search (DFS).
    progress_callback: called as ``f(total_states, state, elapsed)`` each
                       ``PROGRESS_INTERVAL`` iterations, or ``None``.
    """
    n = len(primes)

    # ── precompute σ-factor sets (only needed for factor-chain mode) ──
    if propagate:
        precompute_sig_factors(primes, max_exp)

    # ── init container & stats ──
    use_heap = propagate

    if resume_state is not None:
        heap = resume_state["heap"]
        total_states = resume_state["total_states"]
        heap_counter  = resume_state.get("heap_counter", total_states)
        elapsed_offset = resume_state["elapsed"]
        use_heap = resume_state.get("use_heap", use_heap)
    else:
        init_st = State()
        init_st.priority = _compute_priority(mpz(1), mpz(1), 0.0, 0)
        if use_heap:
            heap = [(init_st.priority, 0, init_st)]
            heap_counter = 1
        else:
            heap = [init_st]
            heap_counter = 0
        total_states   = 0
        elapsed_offset = 0.0

    # ── helpers (heap/stack agnostic) ──
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

    # ── main loop ────────────────────────────────────────────
    while heap:
        # heap-size guard (best-first mode only)
        if use_heap and len(heap) > HEAP_MAX_SIZE:
            heap = heapq.nsmallest(HEAP_MAX_SIZE, heap)
            heapq.heapify(heap)

        st = _pop(heap)

        # checkpoint snapshot
        if state_holder is not None:
            front = [(st.priority, heap_counter, st)] if use_heap else [st]
            state_holder["heap"]         = _snapshot(front + heap)
            state_holder["heap_counter"]  = heap_counter
            state_holder["total_states"]  = total_states
            state_holder["elapsed"]       = time.time() - t0

        total_states += 1
        if total_states % PROGRESS_INTERVAL == 0 and progress_callback is not None:
            progress_callback(total_states, st, time.time() - t0)

        # ── true-OPN check ──────────────────────────────────
        if (
            st.ratio_num == 2 * st.ratio_den
            and st.euler_prime is not None
            and len(st.assigned) >= 2
            and _verify_solution(st)
        ):
            st.pseudo = False
            yield st
            continue

        # ── pseudo-solution check ───────────────────────────
        if _check_pseudo(st):
            yield st
            continue

        # ── pruning ─────────────────────────────────────────
        if st.ratio_num >= 2 * st.ratio_den:
            continue
        if len(st.assigned) >= max_factors:
            continue

        lb_num, lb_den = ratio_lower_bound(
            st.ratio_num, st.ratio_den, st.pending,
        )
        if lb_num > 2 * lb_den:
            continue

        ub_num, ub_den = ratio_upper_bound(
            st.ratio_num, st.ratio_den,
            st.assigned, st.excluded, primes,
        )
        if ub_num < 2 * ub_den:
            continue

        # ── process pending (forced) primes ─────────────────
        if _drain_and_process_pending(
            st, heap, primes, max_exp, propagate, _push,
        ):
            continue

        # ── expansion (branch on next unprocessed prime) ────
        idx = st.next_idx
        while idx < n:
            p = primes[idx]
            if p in st.assigned or p in st.excluded:
                idx += 1
                continue

            # skip branch
            skip_st = st.clone()
            skip_st.excluded.add(p)
            skip_st.next_idx = idx + 1
            skip_st.priority = _compute_priority(
                skip_st.ratio_num, skip_st.ratio_den,
                skip_st.resonance, len(skip_st.assigned),
            )
            _push(heap, skip_st)

            # Euler-include branches
            if st.euler_prime is None and p % 4 == 1:
                for e in reversed(valid_euler_exponents(1, max_exp)):
                    ns = assign_prime(st, p, e, propagate=propagate, max_exp=max_exp)
                    if ns is not None:
                        ns.next_idx = idx + 1
                        _push(heap, ns)

            # non-Euler include branches
            for e in reversed(valid_even_exponents(2, max_exp)):
                ns = assign_prime(st, p, e, propagate=propagate, max_exp=max_exp)
                if ns is not None:
                    ns.next_idx = idx + 1
                    _push(heap, ns)

            break

    # ── search exhausted ────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n搜索完成: {total_states:,} states, {elapsed:.1f}s")
    if state_holder is not None:
        state_holder["heap"]         = []
        state_holder["heap_counter"]  = heap_counter
        state_holder["total_states"]  = total_states
        state_holder["elapsed"]       = elapsed


# ── pending-prime processing ───────────────────────────────────
def _drain_and_process_pending(
    st: State, heap, primes, max_exp: int, propagate: bool, _push,
) -> bool:
    """Pop one pending forced prime and push its exponent branches."""
    if not st.pending:
        return False

    # drain already-assigned or out-of-range entries
    while st.pending and (
        st.pending[0] in st.assigned
        or st.pending[0] > primes[-1]
    ):
        q = st.pending.popleft()
        st.pending_set.discard(q)

    if not st.pending:
        return False

    q = st.pending.popleft()
    st.pending_set.discard(q)
    lb = max(st.required_v.get(q, 1) - st.current_v.get(q, 0), 1)

    # Euler-candidate branches
    if st.euler_prime is None and q % 4 == 1:
        for e in reversed(valid_euler_exponents(lb, max_exp)):
            ns = assign_prime(st, q, e, propagate=propagate, max_exp=max_exp)
            if ns is not None:
                _push(heap, ns)

    # non-Euler branches
    for e in reversed(valid_even_exponents(lb, max_exp)):
        ns = assign_prime(st, q, e, propagate=propagate, max_exp=max_exp)
        if ns is not None:
            _push(heap, ns)

    return True
