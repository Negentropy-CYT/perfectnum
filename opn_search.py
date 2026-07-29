"""
opn_search — constraint-propagation factor-chain search engine.

Exposes ``search_opn()``, a generator that yields State objects for
each valid OPN or Descartes-spoof candidate discovered.

Supports two search strategies:
  - DFS (stack)   — for Descartes-spoof search (DFSState)
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
    DOMAIN_RATIO_MODE,
    ENABLE_FERMAT_DEBT,
    EXCLUDE_EXP_4,
    EXP4_FILTER_HITS,
    POOL_GCD_MODE,
    POOL_PLAN_DISK_MIN_FREE_BYTES,
    POOL_PLAN_CHUNK_PRIMES,
    POOL_SUPERBLOCK_FANOUT,
    SEARCH_MODE,
    SigmaPoolAnalyzer,
    TOXIC_SKIP,
    _sigma_map_perf,
    check_touchard,
    euler_max_exp_capacity,
    even_max_exp_capacity,
    max_prime_capacity,
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
from dataclasses import dataclass

from opn_metrics import (
    CloneEffect,
    PruneMechanism,
    PruneReason,
    RunMetrics,
)


@dataclass(frozen=True, slots=True)
class PendingExponentDomain:
    """Deterministic exponent choices for one pending prime."""
    lower_bound: int
    even_exponents: tuple[int, ...]
    euler_exponents: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.even_exponents) + len(self.euler_exponents)

    @property
    def empty(self) -> bool:
        return self.size == 0

    @property
    def forced_euler(self) -> bool:
        return not self.even_exponents and bool(self.euler_exponents)


def _pending_lower_bound(st: "ChainState", q: int) -> int:
    return max(
        st.required_v.get(q, 0)
        - _target_valuation_offset(q)
        - st.current_v.get(q, 0),
        1,
    )


def _build_pending_domain(
    st: "ChainState",
    q: int,
    *,
    max_exp: int,
    apply_maximum_capacity: bool,
) -> PendingExponentDomain:
    lower = _pending_lower_bound(st, q)
    even = tuple(valid_even_exponents(lower, max_exp))
    euler: tuple[int, ...] = ()
    if (
        st.euler_prime is None
        and q % 4 == 1
        and SEARCH_MODE.require_euler
    ):
        euler = tuple(valid_euler_exponents(lower, max_exp))
    if apply_maximum_capacity:
        raw_capacity = max_prime_capacity(q)
        even_limit = even_max_exp_capacity(raw_capacity)
        euler_limit = euler_max_exp_capacity(raw_capacity)
        even = tuple(e for e in even if e <= even_limit)
        euler = tuple(e for e in euler if e <= euler_limit)
    return PendingExponentDomain(
        lower_bound=lower,
        even_exponents=even,
        euler_exponents=euler,
    )


@dataclass(frozen=True, slots=True)
class MandatoryRatioResult:
    possible: bool
    numerator: mpz
    denominator: mpz
    reason: str = ""


def _first_even_exponent(lower: int, max_exp: int) -> int | None:
    """Smallest valid even exponent ≥ *lower*, or None."""
    exp = max(lower, 2)
    if exp % 2:
        exp += 1
    return exp if exp <= max_exp else None


def _first_euler_exponent(lower: int, max_exp: int) -> int | None:
    """Smallest valid Euler exponent ≥ *lower*, or None."""
    exp = max(lower, 1)
    exp += (1 - exp) % 4
    return exp if exp <= max_exp else None


def _component_ratio(p: int, exp: int) -> tuple[mpz, mpz]:
    return (mpz(sigma_prime_power(p, exp)), mpz(power_pa(p, exp)))


def _domain_ratio_lower_bound(
    st: "ChainState",
    live_pending: set[int],
    *,
    max_exp: int,
) -> MandatoryRatioResult:
    """Compute a relaxed but safe mandatory-ratio lower bound.

    Every pending prime must receive at least its minimum admissible
    exponent (even or Euler, whichever is available).  The product of
    their best-possible σ/ratio multipliers is a lower bound on the
    final ratio — if it already exceeds the target, no completion exists.
    """
    from opn_core import valid_euler_exponents, valid_even_exponents

    entries: list[tuple[int, int | None, int | None]] = []
    forced_euler: int | None = None

    for q in sorted(live_pending):
        lower = _pending_lower_bound(st, q)
        even_exp = _first_even_exponent(lower, max_exp)

        euler_exp = None
        if (
            SEARCH_MODE.require_euler
            and st.euler_prime is None
            and q % 4 == 1
        ):
            euler_exp = _first_euler_exponent(lower, max_exp)

        if even_exp is None:
            if euler_exp is None:
                return MandatoryRatioResult(
                    False, mpz(0), mpz(1), "empty_domain",
                )
            if forced_euler is not None:
                return MandatoryRatioResult(
                    False, mpz(0), mpz(1), "multiple_forced_euler",
                )
            forced_euler = q

        entries.append((q, even_exp, euler_exp))

    # Euler prime already fixed: everything must be even.
    if st.euler_prime is not None:
        num = mpz(st.ratio_num)
        den = mpz(st.ratio_den)
        for _q, ev_exp, _eu_exp in entries:
            if ev_exp is None:
                return MandatoryRatioResult(
                    False, mpz(0), mpz(1), "empty_even_domain",
                )
            c_num, c_den = _component_ratio(_q, ev_exp)
            num *= c_num
            den *= c_den
        return MandatoryRatioResult(True, num, den)

    # Exactly one prime is forced to be Euler.
    if forced_euler is not None:
        num = mpz(st.ratio_num)
        den = mpz(st.ratio_den)
        for _q, ev_exp, eu_exp in entries:
            chosen = eu_exp if _q == forced_euler else ev_exp
            if chosen is None:
                return MandatoryRatioResult(
                    False, mpz(0), mpz(1), "incompatible_forced_euler",
                )
            c_num, c_den = _component_ratio(_q, chosen)
            num *= c_num
            den *= c_den
        return MandatoryRatioResult(True, num, den)

    # All-even assignment is feasible.  Euler may be supplied later by
    # an optional prime, so the all-even case is one valid relaxed
    # lower bound.  We also try giving each candidate the Euler role
    # in turn, keeping the smallest (i.e. hardest-to-exceed) ratio.
    all_even_num = mpz(st.ratio_num)
    all_even_den = mpz(st.ratio_den)
    cache: dict[int, tuple[int, int | None, mpz, mpz]] = {}

    for _q, ev_exp, eu_exp in entries:
        assert ev_exp is not None
        e_num, e_den = _component_ratio(_q, ev_exp)
        cache[_q] = (ev_exp, eu_exp, e_num, e_den)
        all_even_num *= e_num
        all_even_den *= e_den

    best_num = all_even_num
    best_den = all_even_den

    for _q, (ev_exp, eu_exp, e_num, e_den) in cache.items():
        if eu_exp is None:
            continue
        eu_num, eu_den = _component_ratio(_q, eu_exp)
        candidate_num = all_even_num * eu_num * e_den
        candidate_den = all_even_den * eu_den * e_num
        if candidate_num * best_den < best_num * candidate_den:
            best_num = candidate_num
            best_den = candidate_den

    return MandatoryRatioResult(True, best_num, best_den)


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
# Verification & spoof check
# ══════════════════════════════════════════════════════════════

def _verify_solution(st: State) -> bool:
    lhs = mpz(1)
    rhs = mpz(1)
    for p, a in st.assigned.items():
        lhs *= sigma_prime_power(p, a)
        rhs *= mpz(power_pa(p, a))
    return lhs * SEARCH_MODE.target_den == SEARCH_MODE.target_num * rhs


def _check_spoof(st: State) -> bool:
    if SEARCH_MODE.target_num != 2 or SEARCH_MODE.target_den != 1:
        return False  # spoof formula assumes target = 2/1
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
    st.spoof = True
    return True


# ══════════════════════════════════════════════════════════════
# Main search
# ══════════════════════════════════════════════════════════════

def search_opn(
    primes: List[int],
    max_factors: int,
    max_exp: int,
    *,
    metrics: RunMetrics,
    propagate: bool = True,
    state_holder: Optional[dict] = None,
    resume_state: Optional[dict] = None,
    observer = None,
    progress_callback=None,
    checkpoint_callback=None,
    checkpoint_interval_seconds: Optional[float] = CHECKPOINT_INTERVAL_SECONDS,
    stop_requested=None,
    use_cache: bool = False,
    sigma_database_path: str | None = None,
    pool_plan_cache_dir: str | None = None,
    pool_plan_cache_minimum_free_bytes: int =
    POOL_PLAN_DISK_MIN_FREE_BYTES,
    pool_plan_build_policy: str = "eager",
):
    """Generator yielding State objects for each candidate found.

    *metrics* is the single RunMetrics sink for all observability data.
    *observer* is an optional RuntimeSampler for periodic RSS/rate CSV.
    """
    n = len(primes)
    seen_states = set() if use_cache else None
    use_heap = propagate

    # Wire sigma_map performance bridge
    import opn_core
    opn_core._sigma_map_perf = metrics.performance

    # ── pool-aware sigma analyser (OPN chain mode only) ──
    metrics.configure_exponent_telemetry(max_exp)

    sigma_pool_analyzer = None
    if propagate and SEARCH_MODE.target_num == 2 and SEARCH_MODE.target_den == 1:
        sigma_pool_analyzer = SigmaPoolAnalyzer(
            primes,
            block_size=256,
            superblock_fanout=POOL_SUPERBLOCK_FANOUT,
            gcd_mode=POOL_GCD_MODE,
            plan_chunk_primes=POOL_PLAN_CHUNK_PRIMES,
            pool_perf=metrics.performance.pool,
            structure=metrics.structure,
            database_path=sigma_database_path,
            plan_cache_dir=pool_plan_cache_dir,
            plan_cache_minimum_free_bytes=(
                pool_plan_cache_minimum_free_bytes
            ),
            plan_build_policy=pool_plan_build_policy,
        )
        if sigma_pool_analyzer.database_error is not None:
            print(
                "[init] sigma database unavailable; "
                "falling back to pool scans: "
                f"{sigma_pool_analyzer.database_error}",
                flush=True,
            )
        if sigma_pool_analyzer.plan_cache_error is not None:
            print(
                "[init] persistent plan cache unavailable; "
                "falling back to memory plans: "
                f"{sigma_pool_analyzer.plan_cache_error}",
                flush=True,
            )

    print("using exact factor-slot tail bounds", flush=True)

    if propagate:
        EXCLUDE_EXP_4.clear()
        print("sigma-factor maps will be populated lazily")

    if sigma_pool_analyzer is not None:
        required_exponents = list(
            valid_even_exponents(2, max_exp)
        )

        if SEARCH_MODE.require_euler:
            required_exponents.extend(
                valid_euler_exponents(1, max_exp)
            )

        if pool_plan_build_policy == "eager":
            print(
                "[init] prebuilding sigma-pool plans "
                f"for exponents={sorted(set(required_exponents))} ...",
                flush=True,
            )

            if observer is not None:
                observer.set_phase("plan_prebuild")

            plan_started = time.perf_counter()
            sigma_pool_analyzer.configure_plan_build(
                required_exponents
            )
            plan_elapsed = time.perf_counter() - plan_started

            if observer is not None:
                observer.capture_memory_phase(
                    metrics.performance.memory_phases,
                    "after_plan_prebuild",
                )
                observer.set_phase("search")

            print(
                "[init] sigma-pool plans ready: "
                f"filtered={metrics.performance.pool.filtered_plan_count}, "
                f"full={int(metrics.performance.pool.full_plan_built)}, "
                f"time={plan_elapsed:.3f}s",
                flush=True,
            )
        else:
            sigma_pool_analyzer.configure_plan_build(
                required_exponents
            )
            print(
                "[init] sigma-pool plans deferred "
                f"(policy={pool_plan_build_policy})",
                flush=True,
            )

    if observer is not None:
        observer.set_phase("search")

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
    states_started = int(resume_state.get("states_started", 0)) if resume_state else 0
    states_completed = int(resume_state.get("states_completed", 0)) if resume_state else 0
    last_checkpoint = time.monotonic()

    def _publish_frontier(reason: str) -> None:
        """Expose a coherent frontier while the search loop is paused."""
        nonlocal snapshot_id, last_checkpoint
        if state_holder is None:
            if sigma_pool_analyzer is not None:
                sigma_pool_analyzer.flush()
            last_checkpoint = time.monotonic()
            return
        snapshot_id += 1
        snapshot_elapsed = time.time() - t0
        # Write progress counters *before* the callback so the
        # checkpoint always contains the most recent values.
        state_holder.update({
            "heap": heap,
            "heap_counter": heap_counter,
            "total_states": total_states,
            "states_started": states_started,
            "states_completed": states_completed,
            "elapsed": snapshot_elapsed,
            "snapshot_id": snapshot_id,
            "snapshot_reason": reason,
            "frontier_size": len(heap),
            "live_total_states": total_states,
            "live_elapsed": snapshot_elapsed,
        })
        if sigma_pool_analyzer is not None:
            sigma_pool_analyzer.flush()
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
        states_started += 1

        try:
            if seen_states is not None:
                signature = _state_signature(st)
                if signature in seen_states:
                    metrics.record_prune(
                        reason=PruneReason.DUPLICATE_STATE,
                        mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
                        clone_effect=CloneEffect.AVOIDED,
                    )
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
                st.spoof = False
                _publish_frontier("solution")
                yield st
                continue

            # ── spoof check ──
            if _check_spoof(st):
                if use_heap:
                    pass
                else:
                    _publish_frontier("solution")
                yield st
                if not use_heap:
                    continue

            # ── pruning ──
            if st.ratio_num * SEARCH_MODE.target_den >= SEARCH_MODE.target_num * st.ratio_den:
                continue
            if len(st.assigned) >= max_factors:
                continue

            k_remain = max_factors - len(st.assigned)

            if use_heap and touchard_force_3(st.euler_prime, st.assigned,
                                             st.excluded):
                _enqueue_pending(st, 3)

            live_pending = (
                {q for q in st.pending if q not in st.assigned}
                if use_heap else set()
            )
            if use_heap:
                if any(q > primes[-1] for q in live_pending):
                    metrics.record_prune(
                        reason=PruneReason.OUTSIDE_WINDOW,
                        mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
                        clone_effect=CloneEffect.AVOIDED,
                    )
                    continue
                if live_pending & st.excluded:
                    metrics.record_prune(
                        reason=PruneReason.EXCLUDED_PRIME,
                        mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
                        clone_effect=CloneEffect.AVOIDED,
                    )
                    continue
                if len(live_pending) > k_remain:
                    metrics.record_prune(
                        reason=PruneReason.FACTOR_SLOTS,
                        mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
                        clone_effect=CloneEffect.AVOIDED,
                    )
                    continue

            lb_num, lb_den = ratio_lower_bound(
                st.ratio_num, st.ratio_den,
                live_pending,
            )
            if lb_num * SEARCH_MODE.target_den > SEARCH_MODE.target_num * lb_den:
                continue

            if (
                use_heap
                and live_pending
                and DOMAIN_RATIO_MODE != "off"
            ):
                domain_bound = _domain_ratio_lower_bound(
                    st,
                    live_pending,
                    max_exp=max_exp,
                )
                if not domain_bound.possible:
                    reason = (
                        PruneReason.EULER_FORM
                        if domain_bound.reason == "multiple_forced_euler"
                        else PruneReason.VALUATION_CONTRADICTION
                    )
                    if DOMAIN_RATIO_MODE == "enforce":
                        metrics.record_prune(
                            reason=reason,
                            mechanism=PruneMechanism.MANDATORY_RATIO_BOUND,
                            clone_effect=CloneEffect.AVOIDED,
                        )
                        continue
                    metrics.performance.domain_ratio_would_prune += 1
                elif (
                    domain_bound.numerator * SEARCH_MODE.target_den
                    > SEARCH_MODE.target_num * domain_bound.denominator
                ):
                    if DOMAIN_RATIO_MODE == "enforce":
                        metrics.record_prune(
                            reason=PruneReason.RATIO_OVERSHOOT,
                            mechanism=PruneMechanism.MANDATORY_RATIO_BOUND,
                            clone_effect=CloneEffect.AVOIDED,
                        )
                        continue
                    metrics.performance.domain_ratio_would_prune += 1

            _t0 = time.perf_counter_ns()
            ub_num, ub_den = ratio_upper_bound(
                st.ratio_num, st.ratio_den,
                st.assigned, st.excluded, primes,
                next_idx=st.next_idx,
                remaining_slots=k_remain,
                pending=live_pending,
            )
            perf = metrics.performance
            perf.ratio_upper_ns += time.perf_counter_ns() - _t0
            perf.ratio_upper_calls += 1
            if ub_num * SEARCH_MODE.target_den < SEARCH_MODE.target_num * ub_den:
                metrics.record_prune(
                    reason=PruneReason.RATIO_UNREACHABLE,
                    mechanism=PruneMechanism.TAIL_RATIO_BOUND,
                    clone_effect=CloneEffect.AVOIDED,
                )
                continue

            if use_heap and ENABLE_FERMAT_DEBT:
                _t1 = time.perf_counter_ns()
                debt_ok, _debt_detail = fermat_debt_capacity(
                    st, primes, max_factors, max_exp,
                )
                perf.fermat_debt_ns += time.perf_counter_ns() - _t1
                perf.fermat_debt_calls += 1
                if not debt_ok:
                    metrics.record_prune(
                        reason=PruneReason.VALUATION_CONTRADICTION,
                        mechanism=PruneMechanism.PRECLONE_VALUATION,
                        clone_effect=CloneEffect.AVOIDED,
                    )
                    continue

            # ── pending (chain mode) ──
            if use_heap:
                if _drain_and_process_pending(
                    st, heap, primes, max_exp, _push, k_remain,
                    sigma_pool_analyzer=sigma_pool_analyzer,
                    metrics=metrics,
                ):
                    continue

            # ── expansion ──
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
                    metrics.record_prune(
                        reason=PruneReason.RATIO_OVERSHOOT,
                        mechanism=PruneMechanism.INTERVAL_BOUND,
                        clone_effect=CloneEffect.AVOIDED,
                    )
                    idx += 1; continue
                if use_heap:
                    hi = next_prime_upper_bound(
                        st.ratio_num, st.ratio_den, idx, k_remain,
                        SEARCH_MODE.target_num, SEARCH_MODE.target_den,
                        primes, st.assigned, st.excluded,
                    )
                    if hi > 0 and p > hi:
                        metrics.record_prune(
                            reason=PruneReason.RATIO_UNREACHABLE,
                            mechanism=PruneMechanism.INTERVAL_BOUND,
                            clone_effect=CloneEffect.AVOIDED,
                        )
                        break

                capacity_enabled = (
                    SEARCH_MODE.require_euler
                    and SEARCH_MODE.target_num == 2
                    and SEARCH_MODE.target_den == 1
                )
                is_max_candidate = (
                    capacity_enabled
                    and k_remain == 1
                    and (not use_heap or not st.pending)
                    and all(p >= q for q in st.assigned)
                )
                if is_max_candidate:
                    raw_cap = max_prime_capacity(p)
                    euler_cap = euler_max_exp_capacity(raw_cap)
                    even_cap = even_max_exp_capacity(raw_cap)

                # skip branch
                skip_st = st.clone()
                metrics.record_clone(len(st.assigned))
                skip_st.excluded.add(p)
                skip_st.next_idx = idx + 1
                _push(heap, skip_st)

                # Euler-include
                if st.euler_prime is None and p % 4 == 1:
                    for e in reversed(valid_euler_exponents(1, max_exp)):
                        if is_max_candidate and e > euler_cap:
                            metrics.record_prune(
                                reason=PruneReason.CAPACITY_BOUND,
                                mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
                                clone_effect=CloneEffect.AVOIDED,
                            )
                            continue
                        if _terminal_prune(st, p, e, k_remain, metrics):
                            continue
                        new_num = mpz(st.ratio_num) * sigma_prime_power(p, e)
                        new_den = mpz(st.ratio_den) * power_pa(p, e)
                        if new_num * SEARCH_MODE.target_den > SEARCH_MODE.target_num * new_den:
                            metrics.record_prune(
                                reason=PruneReason.RATIO_OVERSHOOT,
                                mechanism=PruneMechanism.PROSPECTIVE_RATIO,
                                clone_effect=CloneEffect.AVOIDED,
                            )
                            continue
                        if sigma_pool_analyzer is not None and sigma_pool_analyzer.is_known_outside(p, e):
                            metrics.record_prune(
                                reason=PruneReason.OUTSIDE_WINDOW,
                                mechanism=PruneMechanism.KNOWN_OUTSIDE_CACHE,
                                clone_effect=CloneEffect.AVOIDED,
                            )
                            continue
                        ns = _assign(st, p, e, use_heap, propagate, max_exp,
                                       prime_limit=primes[-1],
                                       sigma_pool_analyzer=sigma_pool_analyzer,
                                       metrics=metrics)
                        if ns is not None:
                            ns.next_idx = idx + 1
                            _push(heap, ns)

                # non-Euler include
                for e in reversed(valid_even_exponents(2, max_exp)):
                    if is_max_candidate and e > even_cap:
                        metrics.record_prune(
                            reason=PruneReason.CAPACITY_BOUND,
                            mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
                            clone_effect=CloneEffect.AVOIDED,
                        )
                        continue
                    if _terminal_prune(st, p, e, k_remain, metrics):
                        continue
                    new_num = mpz(st.ratio_num) * sigma_prime_power(p, e)
                    new_den = mpz(st.ratio_den) * power_pa(p, e)
                    if new_num * SEARCH_MODE.target_den > SEARCH_MODE.target_num * new_den:
                        metrics.record_prune(
                            reason=PruneReason.RATIO_OVERSHOOT,
                            mechanism=PruneMechanism.PROSPECTIVE_RATIO,
                            clone_effect=CloneEffect.AVOIDED,
                        )
                        continue
                    if sigma_pool_analyzer is not None and sigma_pool_analyzer.is_known_outside(p, e):
                        metrics.record_prune(
                            reason=PruneReason.OUTSIDE_WINDOW,
                            mechanism=PruneMechanism.KNOWN_OUTSIDE_CACHE,
                            clone_effect=CloneEffect.AVOIDED,
                        )
                        continue
                    ns = _assign(st, p, e, use_heap, propagate, max_exp,
                                   prime_limit=primes[-1],
                                   sigma_pool_analyzer=sigma_pool_analyzer,
                                   metrics=metrics)
                    if ns is not None:
                        ns.next_idx = idx + 1
                        _push(heap, ns)

                break
        finally:
            states_completed += 1
            if observer is not None:
                observer.update_progress(
                    states_started=states_started,
                    states_completed=states_completed,
                    frontier_size=len(heap),
                )

    # ── exhausted ──
    elapsed = time.time() - t0
    print()  # end inline progress line cleanly
    print(f"搜索完成: {total_states:,} states, {elapsed:.1f}s")
    _publish_frontier("complete")


# ── terminal-slot pruning ─────────────────────────────────────

def _terminal_prune(st, p, exp, k_remain, metrics: RunMetrics) -> bool:
    """Pre-clone prune for the last factor slot.  Returns True if pruned."""
    if k_remain != 1:
        return False

    new_num = st.ratio_num * sigma_prime_power(p, exp)
    new_den = st.ratio_den * power_pa(p, exp)
    if new_num * SEARCH_MODE.target_den != SEARCH_MODE.target_num * new_den:
        metrics.record_prune(
            reason=PruneReason.TERMINAL_RATIO,
            mechanism=PruneMechanism.TERMINAL_CHECK,
            clone_effect=CloneEffect.AVOIDED,
        )
        return True

    if (
        SEARCH_MODE.require_euler
        and st.euler_prime is None
        and exp % 2 == 0
    ):
        metrics.record_prune(
            reason=PruneReason.TERMINAL_NO_EULER,
            mechanism=PruneMechanism.TERMINAL_CHECK,
            clone_effect=CloneEffect.AVOIDED,
        )
        return True

    return False


# ── polymorphic assign dispatch ──────────────────────────────

def _assign(
    st: State, p: int, exp: int, use_heap: bool,
    propagate: bool, max_exp: int,
    *,
    metrics: RunMetrics,
    prime_limit: int | None = None,
    sigma_pool_analyzer=None,
) -> Optional[State]:
    """Dispatch to DFSState or ChainState assign function."""
    if use_heap:
        return assign_prime_chain(
            st, p, exp,
            metrics=metrics,
            propagate=propagate,
            max_exp=max_exp,
            prime_limit=prime_limit,
            sigma_pool_analyzer=sigma_pool_analyzer,
        )
    else:
        return assign_prime_dfs(st, p, exp, metrics=metrics, max_exp=max_exp)


# ── pending processing (chain mode) ──────────────────────────

def _drain_and_process_pending(
    st: "ChainState", heap, primes, max_exp: int, _push, k_remain: int,
    *,
    sigma_pool_analyzer=None,
    metrics: RunMetrics,
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
            metrics.record_prune(
                reason=PruneReason.OUTSIDE_WINDOW,
                mechanism=PruneMechanism.DIRECT_DOMAIN_CHECK,
                clone_effect=CloneEffect.AVOIDED,
            )
            return True

    if not st.pending:
        return False

    q = st.pending.popleft()
    st.pending_set.discard(q)

    capacity_enabled = (
        SEARCH_MODE.require_euler
        and SEARCH_MODE.target_num == 2
        and SEARCH_MODE.target_den == 1
    )
    is_max_in_pending = (
        capacity_enabled
        and k_remain == 1
        and all(q >= p for p in st.assigned)
        and all(q >= p for p in st.pending)
    )

    domain = _build_pending_domain(
        st,
        q,
        max_exp=max_exp,
        apply_maximum_capacity=is_max_in_pending,
    )

    for e in reversed(domain.euler_exponents):
        if _terminal_prune(st, q, e, k_remain, metrics):
            continue
        if sigma_pool_analyzer is not None and sigma_pool_analyzer.is_known_outside(q, e):
            metrics.record_prune(
                reason=PruneReason.OUTSIDE_WINDOW,
                mechanism=PruneMechanism.KNOWN_OUTSIDE_CACHE,
                clone_effect=CloneEffect.AVOIDED,
            )
            continue
        ns = assign_prime_chain(
            st, q, e,
            metrics=metrics,
            propagate=True,
            max_exp=max_exp,
            prime_limit=primes[-1],
            sigma_pool_analyzer=sigma_pool_analyzer,
        )
        if ns is not None:
            _push(heap, ns)

    for e in reversed(domain.even_exponents):
        if _terminal_prune(st, q, e, k_remain, metrics):
            continue
        if sigma_pool_analyzer is not None and sigma_pool_analyzer.is_known_outside(q, e):
            metrics.record_prune(
                reason=PruneReason.OUTSIDE_WINDOW,
                mechanism=PruneMechanism.KNOWN_OUTSIDE_CACHE,
                clone_effect=CloneEffect.AVOIDED,
            )
            continue
        ns = assign_prime_chain(
            st, q, e,
            metrics=metrics,
            propagate=True,
            max_exp=max_exp,
            prime_limit=primes[-1],
            sigma_pool_analyzer=sigma_pool_analyzer,
        )
        if ns is not None:
            _push(heap, ns)

    return True
