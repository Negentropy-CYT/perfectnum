"""pytest suite for OPN search engine improvements.

Covers:
  - Core arithmetic: prime generation, sigma, factorisation, ratio bounds
  - Interval bounds: lower/upper bound correctness (regression vs known values)
  - Touchard: congruence pruning correctness
  - Pseudo-solution: known Descartes spoof must be found (regression)
  - Early ratio prune: exact-ratio guard (>= → > fix verification)
  - Reverse valuations and Fermat-prime debt capacity
  - Infinite-power: threshold function
  - Friend-of-10: Euler skip, 5-force, 3-exclude
  - Checkpoint: save/restore round-trip
  - Regression: Descartes spoof found in DFS mode

Usage:
    pytest test_opn.py -v
    pytest test_opn.py -v -k "slow"   # only long-running tests
"""

import math
import os
import pickle
import sys
from array import array
from fractions import Fraction
from itertools import combinations

import pytest
from gmpy2 import mpz

# Run from the improvements/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opn_core import (
    MAX_EXP,
    MAX_FACTORS,
    MAX_PRIME,
    SIGMA_CACHE,
    POWER_CACHE,
    FACTOR_CACHE,
    _SIG_FACTORS,
    _SIG_VALUATIONS,
    _TOTIENT_CACHE,
    _CAPACITY_CACHE,
    FRIEND_10_MODE,
    OPN_MODE,
    SEARCH_MODE,
    SigmaPoolAnalyzer,
    _remove_all,
    build_prime_blocks,
    build_prime_superblocks,
    brent_rho,
    mpz,
    check_touchard,
    euler_max_exp_capacity,
    even_max_exp_capacity,
    exp4_forced_outside_window,
    factorize,
    generate_odd_primes,
    is_prime_infinite,
    max_prime_capacity,
    next_prime_lower_bound,
    next_prime_upper_bound,
    power_pa,
    precompute_sig_factors,
    ratio_lower_bound,
    ratio_upper_bound,
    residue_class_count,
    sigma_valuation_map,
    sigma_valuation_from_order,
    sigma_prime_power,
    totient,
    touchard_force_3,
    valid_euler_exponents,
    valid_even_exponents,
)
from opn_search import (
    SearchStopped,
    _check_pseudo,
    _heap_snapshot,
    _verify_solution,
    search_opn,
)
from opn_state import (
    ChainState,
    DFSState,
    _capacity_ranking,
    _compute_priority,
    _early_ratio_prune,
    _euler_ok,
    _max_possible_valuation,
    _source_valuation_capacity,
    assign_prime_chain,
    assign_prime_dfs,
    fermat_debt_capacity,
    valuation_debts,
)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def reference_odd_primes(limit: int) -> list[int]:
    """Small test-only reference sieve."""
    if limit < 3:
        return []
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            start = p * p
            count = ((limit - start) // p) + 1
            sieve[start:limit + 1:p] = b"\x00" * count
    return [p for p in range(3, limit + 1, 2) if sieve[p]]


def analyze_with_plan(p: int, exp: int, plan, scanner):
    """Run a pool analysis using a specific block plan and scanner."""
    from collections import Counter
    residual = mpz(sigma_prime_power(p, exp))
    residual, _ = _remove_all(residual, 2)
    inside = {}
    stats = Counter()
    residual = scanner(residual, inside, plan, stats)
    return residual, inside


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def clear_caches():
    """Clear all module-level caches before each test."""
    SIGMA_CACHE.clear()
    POWER_CACHE.clear()
    FACTOR_CACHE.clear()
    _SIG_FACTORS.clear()
    _SIG_VALUATIONS.clear()
    _TOTIENT_CACHE.clear()
    _CAPACITY_CACHE.clear()
    sigma_valuation_from_order.cache_clear()
    _source_valuation_capacity.cache_clear()
    _capacity_ranking.cache_clear()


@pytest.fixture
def small_primes():
    return generate_odd_primes(50)


# ══════════════════════════════════════════════════════════════
# Core Arithmetic
# ══════════════════════════════════════════════════════════════

class TestPrimes:
    def test_generate_up_to_50(self):
        p = generate_odd_primes(50)
        assert list(p) == [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def test_generate_up_to_100_first_last(self):
        p = generate_odd_primes(100)
        assert int(p[0]) == 3
        assert int(p[-1]) == 97

    def test_no_even_primes(self):
        p = generate_odd_primes(200)
        assert all(int(q) % 2 == 1 for q in p)


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 10, 31, 100, 999, 1_000, 10_000, 100_000])
@pytest.mark.parametrize("segment_odds", [1, 2, 7, 64, 1_000])
def test_segmented_sieve_matches_reference(limit, segment_odds):
    actual = generate_odd_primes(limit, segment_odds=segment_odds)
    expected = reference_odd_primes(limit)
    assert isinstance(actual, array)
    assert list(actual) == expected


def test_segmented_sieve_known_million_count():
    primes = generate_odd_primes(1_000_000)
    assert len(primes) == 78_497
    assert primes[0] == 3
    assert primes[-1] == 999_983


class TestSigma:
    def test_sigma_3_2(self):
        assert sigma_prime_power(3, 2) == 13

    def test_sigma_7_2(self):
        assert sigma_prime_power(7, 2) == 57

    def test_sigma_11_2(self):
        assert sigma_prime_power(11, 2) == 133

    def test_sigma_13_2(self):
        assert sigma_prime_power(13, 2) == 183

    def test_sigma_cached(self):
        a = sigma_prime_power(3, 2)
        b = sigma_prime_power(3, 2)
        assert a == b == 13


class TestFactorize:
    def test_prime(self):
        assert factorize(13) == [(13, 1)]

    def test_composite_small(self):
        assert factorize(57) == [(3, 1), (19, 1)]

    def test_composite_square(self):
        assert factorize(121) == [(11, 2)]

    def test_cached(self):
        a = factorize(183)
        b = factorize(183)
        assert a is b  # same object from cache


class TestPower:
    def test_power_small(self):
        assert power_pa(3, 2) == 9
        assert power_pa(7, 2) == 49


# ══════════════════════════════════════════════════════════════
# Interval Bounds (P0-2 verification)
# ══════════════════════════════════════════════════════════════

class TestIntervalBounds:
    def test_lower_bound_ratio_1(self):
        """At ratio=1, any prime >= 1 can help reach target=2."""
        lo = next_prime_lower_bound(mpz(1), mpz(1), 2, 1)
        assert lo == 1

    def test_lower_bound_ratio_13_9(self):
        """After {3:2}, ratio=13/9≈1.444, p >= ceil(13/(18-13))=3."""
        lo = next_prime_lower_bound(mpz(13), mpz(9), 2, 1)
        assert lo == 3

    def test_lower_bound_already_at_target(self):
        """At ratio=2, no lower bound (already reached target)."""
        lo = next_prime_lower_bound(mpz(2), mpz(1), 2, 1)
        assert lo == 0

    def test_upper_bound_last_prime(self):
        """With the candidate at the last prime, the tail is 1 and
        hi = 2*1*9/(2*9*1 - 13*1) = 18/5 = 3.  This is correct."""
        primes = generate_odd_primes(30)
        hi = next_prime_upper_bound(
            mpz(13), mpz(9), len(primes) - 1, 1,
            2, 1, primes, {}, set(),
        )
        assert hi == 3  # verified by hand: 18//5 = 3

    def test_upper_bound_uses_only_remaining_slots(self):
        """R=13/9, candidate 5, one tail slot gives U=7/6 and hi=6."""
        primes = [3, 5, 7]
        hi = next_prime_upper_bound(
            mpz(13), mpz(9), 1, 2,
            2, 1, primes, {}, set(),
        )
        assert hi == 6

    def test_upper_bound_unbounded(self):
        """When denom ≤ 0, return 0 (no finite upper bound)."""
        primes = generate_odd_primes(30)
        hi = next_prime_upper_bound(
            mpz(1), mpz(1), 0, len(primes),
            2, 1, primes, {}, set(),
        )
        assert hi == 0  # unbounded

    def test_candidate_bound_never_discards_a_reachable_tail(self):
        primes = [3, 5, 7, 11, 13, 17]
        target = Fraction(2, 1)
        current = Fraction(13, 9)

        for candidate_idx in range(len(primes)):
            for slots in range(1, 4):
                hi = next_prime_upper_bound(
                    current.numerator,
                    current.denominator,
                    candidate_idx,
                    slots,
                    target.numerator,
                    target.denominator,
                    primes,
                    {},
                    set(),
                )
                candidate = primes[candidate_idx]
                tail = primes[candidate_idx + 1:]
                best = current * Fraction(candidate, candidate - 1)
                best *= max(
                    (
                        math.prod(Fraction(p, p - 1) for p in subset)
                        for size in range(min(slots - 1, len(tail)) + 1)
                        for subset in combinations(tail, size)
                    ),
                    default=Fraction(1, 1),
                )
                if best >= target:
                    assert hi == 0 or candidate <= hi


class TestFactorSlotUpperBound:
    @staticmethod
    def _brute_tail(primes, start_idx, slots, assigned, excluded, pending):
        mandatory = set(pending) - set(assigned)
        optional = [
            p for p in primes[start_idx:]
            if p not in assigned
            and p not in excluded
            and p not in mandatory
        ]
        choose = min(slots - len(mandatory), len(optional))
        best = Fraction(0, 1)
        for subset in combinations(optional, choose):
            value = math.prod(
                (Fraction(p, p - 1) for p in mandatory | set(subset)),
                start=Fraction(1, 1),
            )
            best = max(best, value)
        return best

    @pytest.mark.parametrize("start_idx", range(4))
    @pytest.mark.parametrize("slots", range(1, 4))
    def test_matches_exhaustive_subset_maximum(self, start_idx, slots):
        primes = [3, 5, 7, 11, 13, 17]
        assigned = {11: 2}
        excluded = {7}
        pending = {3} if start_idx > 0 else set()
        if len(pending) > slots:
            return

        num, den = ratio_upper_bound(
            mpz(1), mpz(1), assigned, excluded, primes,
            next_idx=start_idx,
            remaining_slots=slots,
            pending=pending,
        )
        assert Fraction(int(num), int(den)) == self._brute_tail(
            primes, start_idx, slots, assigned, excluded, pending,
        )

    def test_pending_before_next_idx_is_included(self):
        primes = [3, 5, 7, 11, 13, 17]
        num, den = ratio_upper_bound(
            mpz(1), mpz(1), {13: 2}, set(), primes,
            next_idx=5,
            remaining_slots=2,
            pending={3},
        )
        assert Fraction(int(num), int(den)) == Fraction(3, 2) * Fraction(17, 16)

    def test_pending_consumes_a_factor_slot(self):
        primes = [3, 5, 7, 11, 13]
        num, den = ratio_upper_bound(
            mpz(1), mpz(1), {}, set(), primes,
            next_idx=1,
            remaining_slots=2,
            pending={13},
        )
        assert Fraction(int(num), int(den)) == Fraction(13, 12) * Fraction(5, 4)

    def test_invalid_pending_invariants_are_rejected(self):
        primes = [3, 5, 7]
        with pytest.raises(ValueError):
            ratio_upper_bound(
                mpz(1), mpz(1), {}, {3}, primes,
                next_idx=0, remaining_slots=1, pending={3},
            )
        with pytest.raises(ValueError):
            ratio_upper_bound(
                mpz(1), mpz(1), {}, set(), primes,
                next_idx=0, remaining_slots=1, pending={3, 5},
            )


# ══════════════════════════════════════════════════════════════
# Touchard Congruence
# ══════════════════════════════════════════════════════════════

class TestTouchard:
    def test_case_a_3_in_n_ok(self):
        assert check_touchard(None, {3: 2}, set())

    def test_case_a_3_odd_exponent(self):
        """3 with odd exponent cannot be Euler (3≡3 mod4)."""
        assert not check_touchard(None, {3: 1}, set())

    def test_case_a_3_exp_lt_2(self):
        assert not check_touchard(None, {3: 0}, set())

    def test_case_b_no_3_euler_2_mod_3(self):
        """Euler ≡ 2 mod 3, 3 not in N, 3 not excluded → ok (3 can be forced)."""
        assert check_touchard(5, {7: 2}, set())

    def test_case_b_no_3_euler_2_mod_3_excluded(self):
        """Euler ≡ 2 mod 3, 3 excluded → contradiction."""
        assert not check_touchard(5, {7: 2}, {3})

    def test_force_3(self):
        """touchard_force_3 returns True when 3 must be included."""
        assert touchard_force_3(5, {7: 2}, set())

    def test_no_force_3_already_assigned(self):
        assert not touchard_force_3(5, {3: 2, 7: 2}, set())


# ══════════════════════════════════════════════════════════════
# Reverse valuation and Fermat-prime debt capacity
# ══════════════════════════════════════════════════════════════

class TestReverseValuation:
    def test_known_values(self):
        assert sigma_valuation_from_order(7, 2, 3) == 1
        assert sigma_valuation_from_order(13, 2, 3) == 1
        assert sigma_valuation_from_order(3, 2, 13) == 1
        assert sigma_valuation_from_order(3, 4, 11) == 2

    def test_matches_direct_sigma_valuation(self):
        for p in generate_odd_primes(50):
            for q in generate_odd_primes(50):
                if p == q:
                    continue
                for a in range(1, 9):
                    value = int(sigma_prime_power(p, a))
                    expected = 0
                    while value % q == 0:
                        value //= q
                        expected += 1
                    assert sigma_valuation_from_order(p, a, q) == expected

    def test_residue_count_zero_does_not_forbid_split_debt(self):
        assert residue_class_count(3, 2, 3) == 0
        assert sigma_valuation_from_order(7, 2, 3) == 1
        assert sigma_valuation_from_order(13, 2, 3) == 1

    def test_target_must_be_an_odd_prime(self):
        with pytest.raises(ValueError):
            residue_class_count(9, 1, 3)
        with pytest.raises(ValueError):
            sigma_valuation_from_order(7, 2, 9)


class TestFermatDebt:
    def test_debt_ledger_direction(self):
        st = ChainState(
            current_v={3: 8, 13: 2},
            required_v={3: 3, 13: 2},
        )
        assert valuation_debts(st) == {3: 5}

    def test_split_debt_is_feasible(self):
        st = ChainState(assigned={3: 2}, current_v={3: 2})
        ok, detail = fermat_debt_capacity(
            st, [3, 7, 13], max_factors=3, max_exp=2,
        )
        assert ok
        assert detail is None

    def test_capacity_shortfall_prunes(self):
        st = ChainState(assigned={3: 8}, current_v={3: 8})
        ok, detail = fermat_debt_capacity(
            st, [3, 5, 7, 11, 13], max_factors=2, max_exp=2,
        )
        assert not ok
        assert detail == (3, 8, 1)

    def test_unselected_euler_budget_includes_exp_9(self):
        assert _max_possible_valuation(13, None, 9) == 9
        assert _max_possible_valuation(13, 5, 9) == 8
        assert _max_possible_valuation(3, None, 9) == 8


# ══════════════════════════════════════════════════════════════
# Infinite Power
# ══════════════════════════════════════════════════════════════

class TestInfinitePower:
    def test_small_power(self):
        assert not is_prime_infinite(3, 2)

    def test_piecewise_cutoffs(self):
        assert not is_prime_infinite(3, 100)
        assert is_prime_infinite(3, 500)
        assert not is_prime_infinite(31, 10)
        assert is_prime_infinite(101, 100)
        assert not is_prime_infinite(1009, 10)
        assert is_prime_infinite(10007, 8)


# ══════════════════════════════════════════════════════════════
# State Operations
# ══════════════════════════════════════════════════════════════

class TestDFSState:
    def test_clone(self):
        st = DFSState()
        st.assigned[3] = 2
        st2 = st.clone()
        assert st2.assigned == {3: 2}
        st2.assigned[5] = 2
        assert 5 not in st.assigned  # independent

    def test_assign_prime_dfs(self):
        st = DFSState()
        ns = assign_prime_dfs(st, 3, 2)
        assert ns is not None
        assert ns.assigned == {3: 2}
        assert ns.ratio_num == 13
        assert ns.ratio_den == 9


class TestChainState:
    def test_clone(self):
        st = ChainState()
        st.assigned[3] = 2
        st2 = st.clone()
        assert st2.assigned == {3: 2}
        st2.assigned[5] = 2
        assert 5 not in st.assigned

    def test_assign_prime_chain(self, small_primes):
        precompute_sig_factors(small_primes, 4)
        st = ChainState()
        ns = assign_prime_chain(st, 3, 2, propagate=True, max_exp=4)
        assert ns is not None
        assert ns.assigned == {3: 2}

    def test_sigma_factor_map_is_populated_lazily(self):
        assert (3, 2) not in _SIG_VALUATIONS
        assert sigma_valuation_map(3, 2) == {13: 1}
        assert _SIG_FACTORS[(3, 2)] == {13}

    def test_euler_precompute_respects_prime_congruence(self):
        precompute_sig_factors([3, 5], 5)
        assert (5, 1) in _SIG_VALUATIONS
        assert (5, 5) in _SIG_VALUATIONS
        assert (3, 1) not in _SIG_VALUATIONS
        assert (3, 5) not in _SIG_VALUATIONS

    def test_exp4_filter_rejects_one_out_of_window_factor(self):
        # sigma(5^4) = 11 * 71: 11 is in the window, but mandatory 71 is not.
        assert exp4_forced_outside_window(5, 13)


# ══════════════════════════════════════════════════════════════
# Pending > MAX_PRIME pruning (P0-1 regression)
# ══════════════════════════════════════════════════════════════

class TestMaxprimePrune:
    def test_pending_beyond_window_prunes_state(self, small_primes):
        """verifies P0-1: forced prime > MAX_PRIME must prune the branch."""
        from opn_search import _drain_and_process_pending

        st = ChainState()
        # seed a pending prime far beyond our small prime window
        st.pending.append(999983)      # well above primes[-1] == 47
        st.pending_set.add(999983)
        st.required_v[999983] = 1

        # _drain_and_process_pending should return True → caller prunes
        heap = []
        called = []

        def fake_push(container, item):
            called.append(item)

        result = _drain_and_process_pending(
            st, heap, small_primes, max_exp=2, _push=fake_push, k_remain=5,
        )
        assert result is True, "P0-1: must return True to signal prune"
        assert len(called) == 0, "no branches should be pushed"


# ══════════════════════════════════════════════════════════════
# Early Ratio Prune (P0-3 verification)
# ══════════════════════════════════════════════════════════════

class TestEarlyRatioPrune:
    def test_below_target(self):
        """ratio < target → no prune."""
        assert not _early_ratio_prune(mpz(1), mpz(1), 3, 2)

    def test_above_target(self):
        """ratio > target → prune."""
        assert _early_ratio_prune(mpz(20), mpz(9), 3, 2)  # 20/9 > 2

    def test_exact_target(self):
        """ratio == target → must NOT prune (P0-3 regression lock).

        current_ratio = 18/13, p=3, a=2, σ(3²)=13, 3²=9.
        (18/13) * (13/9) = 234/117 = 2.0 exactly.
        With the old >= check this would be pruned (blocking a valid solution).
        With the fixed > check it must survive.
        """
        assert not _early_ratio_prune(mpz(18), mpz(13), 3, 2)
        # Sanity: verify the math
        assert 18 * 13 == 2 * 13 * 9  # 234 == 234


# ══════════════════════════════════════════════════════════════
# Pseudo-Solution (Descartes spoof regression)
# ══════════════════════════════════════════════════════════════

class TestPseudoSolution:
    def test_descartes_spoof_found_dfs(self, small_primes):
        """Known spoof must be found in DFS mode."""
        found = None
        for st in search_opn(small_primes, max_factors=5, max_exp=2,
                             propagate=False):
            found = st
            break
        assert found is not None, "Descartes spoof not found"
        assert found.assigned == {3: 2, 7: 2, 11: 2, 13: 2}
        assert found.pseudo is True

    def test_descartes_spoof_verified(self, small_primes):
        """The r-value for the Descartes spoof should be 22021."""
        for st in search_opn(small_primes, max_factors=5, max_exp=2,
                             propagate=False):
            assert _verify_solution(st) is False  # pseudo, not true OPN
            denom = 2 * st.ratio_den - st.ratio_num
            r = st.ratio_num // denom
            assert r == 22021
            break


# ══════════════════════════════════════════════════════════════
# Search Engine Smoke Tests
# ══════════════════════════════════════════════════════════════

class TestSearchEngine:
    def test_chain_mode_runs(self, small_primes):
        """Chain mode should not crash on small primes."""
        count = sum(1 for _ in search_opn(small_primes, max_factors=5,
                                          max_exp=4, propagate=True))
        # Just verify it runs to completion
        assert isinstance(count, int)

    def test_dfs_mode_runs(self, small_primes):
        """DFS mode should not crash."""
        count = sum(1 for _ in search_opn(small_primes, max_factors=5,
                                          max_exp=2, propagate=False))
        assert count >= 1  # at least the Descartes spoof

    def test_priority_computed(self):
        p = _compute_priority(mpz(1), mpz(1), 0.0, 0)
        assert p > 0  # |2.0 - 1.0| - 0 = 1.0

    def test_exact_dedup_preserves_solution_set(self, small_primes):
        def signatures(use_cache):
            return [
                (tuple(sorted(st.assigned.items())), st.euler_prime, st.pseudo)
                for st in search_opn(
                    small_primes,
                    max_factors=5,
                    max_exp=2,
                    propagate=False,
                    use_cache=use_cache,
                )
            ]

        assert signatures(use_cache=True) == signatures(use_cache=False)

    def test_heap_snapshot_reestablishes_heap_order(self):
        states = [ChainState(priority=float(value)) for value in [5, 2, 4, 1]]
        snapshot = _heap_snapshot([
            (5.0, 10, states[0]),
            (2.0, 11, states[1]),
            (4.0, 12, states[2]),
            (1.0, 13, states[3]),
        ])
        for child in range(1, len(snapshot)):
            parent = (child - 1) // 2
            assert snapshot[parent][:2] <= snapshot[child][:2]

    def test_state_holder_does_not_copy_frontier_per_state(
        self, small_primes, monkeypatch,
    ):
        import opn_search

        calls = 0

        def counted_snapshot(entries):
            nonlocal calls
            calls += 1
            return list(entries)

        monkeypatch.setattr(opn_search, "_heap_snapshot", counted_snapshot)
        holder = {}
        list(search_opn(
            small_primes,
            max_factors=5,
            max_exp=4,
            state_holder=holder,
            propagate=True,
            checkpoint_interval_seconds=None,
        ))

        assert calls == 0
        assert holder["snapshot_reason"] == "complete"
        assert holder["heap"] == []
        assert holder["live_total_states"] == holder["total_states"]

    def test_periodic_callback_runs_only_at_serializable_boundaries(
        self, small_primes, monkeypatch,
    ):
        import opn_search

        reasons = []
        tick = 0

        def monotonic():
            nonlocal tick
            tick += 1
            return float(tick)

        monkeypatch.setattr(opn_search.time, "monotonic", monotonic)

        def checkpoint_callback(holder, reason):
            pickle.dumps(holder["heap"], pickle.HIGHEST_PROTOCOL)
            reasons.append(reason)

        list(search_opn(
            small_primes,
            max_factors=5,
            max_exp=2,
            state_holder={},
            propagate=True,
            checkpoint_callback=checkpoint_callback,
            checkpoint_interval_seconds=0.5,
        ))

        assert reasons[0] == "initial"
        assert "periodic" in reasons
        assert reasons[-1] == "complete"

    def test_cooperative_stop_and_resume_preserves_dfs_results(
        self, small_primes,
    ):
        def signature(st):
            return (
                tuple(sorted(st.assigned.items())),
                st.euler_prime,
                st.pseudo,
            )

        baseline_holder = {}
        baseline = [
            signature(st)
            for st in search_opn(
                small_primes,
                max_factors=5,
                max_exp=2,
                state_holder=baseline_holder,
                propagate=False,
                checkpoint_interval_seconds=None,
            )
        ]

        checks = 0

        def should_stop():
            nonlocal checks
            checks += 1
            return checks >= 20

        stopped_holder = {}
        partial = []
        with pytest.raises(SearchStopped):
            partial.extend(
                signature(st)
                for st in search_opn(
                    small_primes,
                    max_factors=5,
                    max_exp=2,
                    state_holder=stopped_holder,
                    propagate=False,
                    checkpoint_interval_seconds=None,
                    stop_requested=should_stop,
                )
            )

        assert stopped_holder["snapshot_reason"] == "stop"
        assert stopped_holder["heap"]

        resume_state = {
            key: stopped_holder[key]
            for key in (
                "heap",
                "heap_counter",
                "total_states",
                "elapsed",
                "use_heap",
                "snapshot_id",
            )
        }
        resumed_holder = {}
        resumed = [
            signature(st)
            for st in search_opn(
                small_primes,
                max_factors=5,
                max_exp=2,
                state_holder=resumed_holder,
                resume_state=resume_state,
                propagate=False,
                checkpoint_interval_seconds=None,
            )
        ]

        assert partial + resumed == baseline
        assert resumed_holder["total_states"] == baseline_holder["total_states"]

    def test_solution_boundary_does_not_requeue_reported_solution(
        self, small_primes,
    ):
        holder = {}
        generator = search_opn(
            small_primes,
            max_factors=5,
            max_exp=2,
            state_holder=holder,
            propagate=False,
            checkpoint_interval_seconds=None,
        )
        solution = next(generator)
        solution_signature = (
            tuple(sorted(solution.assigned.items())),
            solution.euler_prime,
            solution.pseudo,
        )
        assert holder["snapshot_reason"] == "solution"

        resume_state = {
            key: holder[key]
            for key in (
                "heap",
                "heap_counter",
                "total_states",
                "elapsed",
                "use_heap",
                "snapshot_id",
            )
        }
        generator.close()
        resumed = [
            (
                tuple(sorted(st.assigned.items())),
                st.euler_prime,
                st.pseudo,
            )
            for st in search_opn(
                small_primes,
                max_factors=5,
                max_exp=2,
                resume_state=resume_state,
                propagate=False,
                checkpoint_interval_seconds=None,
            )
        ]
        assert solution_signature not in resumed


# ══════════════════════════════════════════════════════════════
# Maximum-Prime Capacity Bound
# ══════════════════════════════════════════════════════════════

class TestCapacityBound:
    def test_totient(self):
        assert totient(1) == 1
        assert totient(2) == 1
        assert totient(3) == 2
        assert totient(5) == 4
        assert totient(9) == 6
        assert totient(15) == 8

    def test_trivial_primes(self):
        """p < 3 or u == 1 → capacity 0."""
        assert max_prime_capacity(2) == 0
        assert max_prime_capacity(3) == 0   # u = oddpart(2) = 1
        assert max_prime_capacity(17) == 0  # u = oddpart(16) = 1
        assert max_prime_capacity(5) == 0   # u = oddpart(4) = 1

    def test_known_values(self):
        """Hand-verified against B(u) = ½ Σ φ(d)² formula."""
        # p=7:  u=3, divisors>1={3}, φ(3)=2, B=4/2=2
        assert max_prime_capacity(7) == 2
        # p=13: u=3, same as above
        assert max_prime_capacity(13) == 2
        # p=19: u=9, divisors>1={3,9}, φ(3)=2 φ(9)=6, B=(4+36)/2=20
        assert max_prime_capacity(19) == 20
        # p=31: u=15, divisors>1={3,5,15}, φ=2,4,8, B=(4+16+64)/2=42
        assert max_prime_capacity(31) == 42
        # p=11: u=5, divisors>1={5}, φ(5)=4, B=16/2=8
        assert max_prime_capacity(11) == 8

    def test_cached(self):
        """Second call returns same value."""
        a = max_prime_capacity(31)
        b = max_prime_capacity(31)
        assert a == b == 42

    def test_euler_rounding(self):
        assert euler_max_exp_capacity(0) == 0
        assert euler_max_exp_capacity(1) == 1
        assert euler_max_exp_capacity(2) == 1
        assert euler_max_exp_capacity(5) == 5
        assert euler_max_exp_capacity(6) == 5
        assert euler_max_exp_capacity(9) == 9
        assert euler_max_exp_capacity(10) == 9

    def test_even_rounding(self):
        assert even_max_exp_capacity(0) == 0
        assert even_max_exp_capacity(1) == 0
        assert even_max_exp_capacity(2) == 2
        assert even_max_exp_capacity(3) == 2
        assert even_max_exp_capacity(8) == 8
        assert even_max_exp_capacity(9) == 8

    def test_cache_cleared_by_fixture(self):
        """clear_caches must also clear capacity/totient caches."""
        assert len(_TOTIENT_CACHE) == 0
        assert len(_CAPACITY_CACHE) == 0

    def test_descartes_spoof_still_found(self, small_primes):
        """Capacity bound must not break existing DFS results."""
        found = None
        for st in search_opn(small_primes, max_factors=5, max_exp=2,
                             propagate=False):
            found = st
            break
        assert found is not None
        assert found.assigned == {3: 2, 7: 2, 11: 2, 13: 2}
        assert found.pseudo is True

    def test_chain_mode_finds_results(self, small_primes):
        """Chain mode with capacity bound should run without error."""
        count = sum(1 for _ in search_opn(small_primes, max_factors=5,
                                          max_exp=4, propagate=True))
        assert isinstance(count, int)


# ══════════════════════════════════════════════════════════════
# Checkpoint round-trip
# ══════════════════════════════════════════════════════════════

class TestCheckpoint:
    def test_round_trip(self, small_primes, tmp_path, monkeypatch):
        """Save state_holder, reload, verify keys."""
        import opn_io

        checkpoint = tmp_path / "checkpoint.pkl"
        monkeypatch.setattr(opn_io, "CHECKPOINT_FILE", str(checkpoint))
        holder = {
            "primes": small_primes,
            "max_factors": 5,
            "max_exp": 2,
            "heap": [],
            "heap_counter": 0,
            "total_states": 100,
            "elapsed": 10.0,
            "use_heap": opn_io.PROPAGATE,
        }
        solutions = [({3: 2, 7: 2}, None, True)]
        opn_io.save_checkpoint(holder, solutions)
        chk = opn_io.load_checkpoint()
        assert chk is not None
        assert chk["total_states"] == 100
        assert len(chk["solutions"]) == 1

    def test_checkpoint_flushes_before_atomic_replace(
        self, small_primes, tmp_path, monkeypatch,
    ):
        import opn_io

        checkpoint = tmp_path / "checkpoint.pkl"
        monkeypatch.setattr(opn_io, "CHECKPOINT_FILE", str(checkpoint))
        fsync_calls = []
        monkeypatch.setattr(opn_io.os, "fsync", fsync_calls.append)
        holder = {
            "primes": small_primes,
            "max_factors": 5,
            "max_exp": 2,
            "heap": [],
            "heap_counter": 0,
            "total_states": 1,
            "elapsed": 0.1,
            "use_heap": opn_io.PROPAGATE,
        }

        opn_io.save_checkpoint(holder, [])

        assert checkpoint.exists()
        assert len(fsync_calls) == 1

    def test_failed_replace_preserves_previous_checkpoint(
        self, small_primes, tmp_path, monkeypatch,
    ):
        import opn_io

        checkpoint = tmp_path / "checkpoint.pkl"
        monkeypatch.setattr(opn_io, "CHECKPOINT_FILE", str(checkpoint))
        holder = {
            "primes": small_primes,
            "max_factors": 5,
            "max_exp": 2,
            "heap": [],
            "heap_counter": 0,
            "total_states": 1,
            "elapsed": 0.1,
            "use_heap": opn_io.PROPAGATE,
        }
        opn_io.save_checkpoint(holder, [])
        holder["total_states"] = 2

        def fail_replace(source, destination):
            raise OSError("simulated replace failure")

        monkeypatch.setattr(opn_io.os, "replace", fail_replace)
        with pytest.raises(OSError, match="simulated replace failure"):
            opn_io.save_checkpoint(holder, [])

        with checkpoint.open("rb") as f:
            assert pickle.load(f)["total_states"] == 1

    def test_nonempty_dfs_frontier_round_trip(
        self, small_primes, tmp_path, monkeypatch,
    ):
        import opn_io

        checkpoint = tmp_path / "checkpoint.pkl"
        monkeypatch.setattr(opn_io, "CHECKPOINT_FILE", str(checkpoint))
        monkeypatch.setattr(opn_io, "PROPAGATE", False)
        holder = {
            "primes": small_primes,
            "max_factors": 5,
            "max_exp": 2,
            "heap": [DFSState(), DFSState(next_idx=1)],
            "heap_counter": 0,
            "total_states": 10,
            "elapsed": 0.2,
            "use_heap": False,
        }

        opn_io.save_checkpoint(holder, [])
        chk = opn_io.load_checkpoint()

        assert chk is not None
        assert len(chk["heap"]) == 2
        assert chk["heap_counter"] == 0

    def test_mode_mismatch_is_not_resumed(self, small_primes, tmp_path, monkeypatch):
        import opn_io

        checkpoint = tmp_path / "checkpoint.pkl"
        monkeypatch.setattr(opn_io, "CHECKPOINT_FILE", str(checkpoint))
        holder = {
            "primes": small_primes,
            "max_factors": 5,
            "max_exp": 2,
            "heap": [],
            "heap_counter": 0,
            "total_states": 100,
            "elapsed": 10.0,
            "use_heap": opn_io.PROPAGATE,
        }
        opn_io.save_checkpoint(holder, [])
        monkeypatch.setattr(opn_io, "PROPAGATE", not opn_io.PROPAGATE)
        assert opn_io.load_checkpoint() is None
        assert checkpoint.exists()


# ══════════════════════════════════════════════════════════════
# Friend-of-10 mode verification
# ══════════════════════════════════════════════════════════════

class TestFriendMode:
    def test_euler_skipped_when_friend(self, monkeypatch):
        """When require_euler=False, odd exponents should be rejected."""
        monkeypatch.setattr("opn_state.SEARCH_MODE", FRIEND_10_MODE)
        assert not _euler_ok(5, 1, None)  # odd exp rejected
        assert _euler_ok(5, 2, None)      # even exp ok

    def test_euler_normal_when_not_friend(self):
        """OPN_MODE (default) — normal Euler rules apply."""
        assert _euler_ok(5, 1, None)       # 5%4=1, ok
        assert not _euler_ok(3, 1, None)   # 3%4≠1, rejected

    def test_target_numerator_absorbs_two_factors_of_3(self, monkeypatch):
        monkeypatch.setattr("opn_state.SEARCH_MODE", FRIEND_10_MODE)
        st = ChainState(excluded={3})

        first = assign_prime_chain(st, 7, 2, propagate=True, max_exp=4)
        assert first is not None
        assert first.required_v[3] == 1
        assert 3 not in first.pending_set

        second = assign_prime_chain(first, 13, 2, propagate=True, max_exp=4)
        assert second is not None
        assert second.required_v[3] == 2
        assert 3 not in second.pending_set

        # sigma(19^2) contributes a third factor of 3, exceeding v_3(9)=2.
        assert assign_prime_chain(
            second, 19, 2, propagate=True, max_exp=4,
        ) is None

    def test_target_denominator_reduces_5_debt(self, monkeypatch):
        monkeypatch.setattr("opn_state.SEARCH_MODE", FRIEND_10_MODE)
        st = ChainState(
            assigned={5: 2},
            current_v={5: 2},
        )
        assert valuation_debts(st) == {5: 1}


# ══════════════════════════════════════════════════════════════
# Pool Analyser — window-smoothness stripping
# ══════════════════════════════════════════════════════════════

class TestPoolAnalyzer:
    def test_analyzer_exact_small(self):
        """σ(3²)=13, pool includes 13 → exact"""
        a = SigmaPoolAnalyzer([3, 5, 7, 11, 13])
        r = a.analyze(3, 2)
        assert r.exact
        assert r.valuations == {13: 1}
        assert r.residual == 1

    def test_analyzer_outside_certificate(self):
        """σ(7²)=57=3×19, 19 not in pool → outside certificate"""
        a = SigmaPoolAnalyzer([3, 5, 7, 11, 13])
        r = a.analyze(7, 2)
        assert not r.exact
        assert r.valuations == {3: 1}
        assert r.residual > 1

    def test_analyzer_cache_hit(self):
        """Second call returns identical object."""
        a = SigmaPoolAnalyzer([3, 5, 7, 11, 13])
        r1 = a.analyze(3, 2)
        r2 = a.analyze(3, 2)
        assert r1 is r2

    def test_3511_10_is_outside_certificate(self):
        """Regression lock: (3511,10) must NOT use full factorisation."""
        primes = generate_odd_primes(5000)
        a = SigmaPoolAnalyzer(primes)
        r = a.analyze(3511, 10)
        assert not r.exact, (
            "3511^10 should produce an outside-pool certificate "
            "(both prime factors exceed 5000)"
        )
        assert r.residual > 1

    def test_exact_from_global_cache(self):
        """When _SIG_VALUATIONS already has a complete map, analyzer reuses it."""
        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        # Pre-populate the global cache with an exact map
        sigma_valuation_map(3, 2)  # σ(3²)=13
        a = SigmaPoolAnalyzer([3, 5, 7, 11, 13])
        r = a.analyze(3, 2)
        assert r.exact
        assert r.valuations == {13: 1}
        assert a.stats.get("exact_from_global_cache", 0) == 1

    def test_3511_10_never_calls_full_factorize(self, monkeypatch):
        """Regression lock: SigmaPoolAnalyzer must not call factorize()."""
        import opn_core

        def forbidden_factorize(_value):
            raise AssertionError("SigmaPoolAnalyzer called full factorize()")

        monkeypatch.setattr(opn_core, "factorize", forbidden_factorize)
        a = SigmaPoolAnalyzer(generate_odd_primes(5000))
        r = a.analyze(3511, 10)
        assert not r.exact
        assert r.residual > 1

    def test_analyzer_updates_external_stats(self):
        """External Counter receives stats from analyzer."""
        from collections import Counter
        stats = Counter()
        a = SigmaPoolAnalyzer([3, 5, 7, 11, 13], stats=stats)
        a.analyze(3, 2)
        assert stats["misses"] == 1
        assert stats["exact"] == 1
        assert stats["analysis_ns"] > 0

    def test_blocks_for_exp_matches_full_blocks(self):
        """Filtered blocks produce identical results to full-pool analysis."""
        from opn_core import build_prime_block_plan, _scan_blocks_flat

        primes = generate_odd_primes(1000)
        full_plan = build_prime_block_plan(
            primes, block_size=16, superblock_fanout=4,
            eligible_primes=list(primes), build_superblocks=False,
        )
        a = SigmaPoolAnalyzer(primes, block_size=16, superblock_fanout=4,
                              gcd_mode="flat")
        for p in generate_odd_primes(200):
            for exp in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
                filtered_plan = a.plan_for_exp(exp)
                # Compare full vs filtered using the flat scanner
                r_full = analyze_with_plan(p, exp, full_plan, _scan_blocks_flat)
                r_filt = analyze_with_plan(p, exp, filtered_plan, _scan_blocks_flat)
                assert r_full[0] == r_filt[0]
                assert r_full[1] == r_filt[1]

    def test_even_n_reuses_single_full_plan(self):
        """All even n share the same full-pool plan object."""
        a = SigmaPoolAnalyzer(generate_odd_primes(500), gcd_mode="hierarchical")
        p1 = a.plan_for_exp(1)   # n=2
        p5 = a.plan_for_exp(5)   # n=6
        p9 = a.plan_for_exp(9)   # n=10
        assert p1 is p5 is p9

    def test_flat_plan_does_not_build_superblocks(self):
        """Flat mode must not construct superblocks."""
        a = SigmaPoolAnalyzer(generate_odd_primes(500), gcd_mode="flat")
        plan = a.plan_for_exp(2)  # even n → full plan
        assert plan.superblocks == ()

    def test_prime_blocks_cover_compact_pool_exactly(self):
        primes = generate_odd_primes(1_000, segment_odds=17)
        blocks = build_prime_blocks(primes, block_size=7)
        covered = []
        for block in blocks:
            assert 0 <= block.start < block.stop <= len(primes)
            covered.extend(range(block.start, block.stop))
            expected = mpz(1)
            for idx in range(block.start, block.stop):
                expected *= int(primes[idx])
            assert block.product == expected
        assert covered == list(range(len(primes)))

    def test_filtered_plan_uses_compact_pool(self):
        primes = generate_odd_primes(10_000)
        a = SigmaPoolAnalyzer(primes, block_size=16, superblock_fanout=4,
                              gcd_mode="hierarchical")
        plan = a.plan_for_exp(2)  # n=3, odd → filtered
        assert isinstance(plan.primes, array)
        assert plan.primes.itemsize >= 4

    def test_full_plan_reuses_master_prime_array(self):
        primes = generate_odd_primes(10_000)
        a = SigmaPoolAnalyzer(primes, gcd_mode="hierarchical")
        p1 = a.plan_for_exp(1)  # even n → full plan
        p5 = a.plan_for_exp(5)  # even n → same full plan
        assert p1 is p5
        assert p1.primes is primes


# ══════════════════════════════════════════════════════════════
# Superblock two-level GCD screening
# ══════════════════════════════════════════════════════════════

class TestSuperblockGCD:
    def test_superblocks_cover_every_leaf_once(self):
        primes = generate_odd_primes(1000)
        blocks = build_prime_blocks(primes, block_size=7)
        supers = build_prime_superblocks(blocks, fanout=4)
        covered = []
        for sb in supers:
            covered.extend(range(sb.start, sb.stop))
        assert covered == list(range(len(blocks)))

    def test_superblock_product_matches_children(self):
        primes = generate_odd_primes(1000)
        blocks = build_prime_blocks(primes, block_size=7)
        supers = build_prime_superblocks(blocks, fanout=4)
        for sb in supers:
            expected = mpz(1)
            for idx in range(sb.start, sb.stop):
                expected *= blocks[idx].product
            assert sb.product == expected

    def test_flat_and_hierarchical_match(self):
        primes = generate_odd_primes(500)
        flat = SigmaPoolAnalyzer(primes, block_size=16, superblock_fanout=4,
                                 gcd_mode="flat")
        hier = SigmaPoolAnalyzer(primes, block_size=16, superblock_fanout=4,
                                 gcd_mode="hierarchical")
        for p in generate_odd_primes(200):
            for exp in [2, 4, 6, 8, 10, 12, 1, 5, 9]:
                r1 = flat.analyze(p, exp)
                _SIG_VALUATIONS.clear(); _SIG_FACTORS.clear()
                r2 = hier.analyze(p, exp)
                _SIG_VALUATIONS.clear(); _SIG_FACTORS.clear()
                assert r1.exact == r2.exact
                assert r1.valuations == r2.valuations
                assert int(r1.residual) == int(r2.residual)

    def test_repeated_valuation_preserved(self):
        """σ(5⁵)=3906=2×3²×7×31"""
        a = SigmaPoolAnalyzer(generate_odd_primes(100), block_size=4,
                              superblock_fanout=2, gcd_mode="hierarchical")
        r = a.analyze(5, 5)
        assert r.exact
        assert r.valuations == {3: 2, 7: 1, 31: 1}

    def test_hierarchical_3511_10_stays_outside(self, monkeypatch):
        import opn_core

        def forbidden_factorize(_):
            raise AssertionError("full factorize was called")

        monkeypatch.setattr(opn_core, "factorize", forbidden_factorize)
        a = SigmaPoolAnalyzer(generate_odd_primes(5000), block_size=32,
                              superblock_fanout=8, gcd_mode="hierarchical")
        r = a.analyze(3511, 10)
        assert not r.exact
        assert r.residual > 1
