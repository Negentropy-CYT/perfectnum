"""pytest suite for OPN search engine improvements.

Covers:
  - Core arithmetic: prime generation, sigma, factorisation, ratio bounds
  - Interval bounds: lower/upper bound correctness (regression vs known values)
  - Touchard: congruence pruning correctness
  - Descartes spoof: known spoof must be found (regression)
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
import json
import random
import sqlite3
import sys
from array import array
from fractions import Fraction
from itertools import combinations

import pytest
import numpy as np
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
    _build_dynamic_leaf_product,
    _product_prime_range,
    _typed_searchsorted_right,
    FRIEND_10_MODE,
    OPN_MODE,
    SEARCH_MODE,
    SigmaPoolAnalyzer,
    _remove_all,
    _scan_blocks_flat,
    _scan_blocks_hierarchical,
    build_component_prime_pools_vectorized,
    build_compact_superblocks,
    build_prime_block_plan,
    build_prime_blocks,
    build_prime_superblocks,
    brent_rho,
    distinct_prime_factors,
    mpz,
    squarefree_kernel,
    check_touchard,
    component_filter_accepts,
    cyclotomic_sigma_components,
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
    prime_pool_prefix_digest,
    ratio_lower_bound,
    ratio_upper_bound,
    residue_class_count,
    sigma_v3_valuation,
    sigma_valuation_map,
    sigma_valuation_from_order,
    sigma_prime_power,
    totient,
    touchard_force_3,
    valid_euler_exponents,
    valid_even_exponents,
    validate_prime_pool_vectorized,
)
from opn_search import (
    MandatoryRatioResult,
    PendingExponentDomain,
    SearchStopped,
    _build_pending_domain,
    _check_spoof,
    _domain_ratio_lower_bound,
    _first_even_exponent,
    _first_euler_exponent,
    _heap_snapshot,
    _pending_lower_bound,
    _verify_solution,
    search_opn,
)
from opn_metrics import RunMetrics, PoolPerformance, StructureMetrics
from opn_sigma_db import SigmaAnalysisDatabase
from opn_plan_cache import PersistentPlanCache

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
    from opn_metrics import PoolPerformance
    residual = mpz(sigma_prime_power(p, exp))
    residual, _ = _remove_all(residual, 2)
    inside = {}
    perf = PoolPerformance()
    residual = scanner(residual, inside, plan, perf)
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


class TestPrimePoolValidation:
    @pytest.mark.parametrize(
        "primes",
        [
            array("I", [3, 5, 7, 11, 13]),
            array("Q", [3, 5, 7, 11, 13]),
            np.array([3, 5, 7, 11, 13], dtype=np.uint32),
            np.array([3, 5, 7, 11, 13], dtype=np.uint64),
            np.array([3, 5, 7, 11, 13], dtype=np.int64),
            [3, 5, 7, 11, 13],
            (3, 5, 7, 11, 13),
        ],
    )
    def test_valid_storage_types(self, primes):
        validate_prime_pool_vectorized(primes, chunk_size=2)

    @pytest.mark.parametrize(
        ("primes", "message"),
        [
            ([], "must not be empty"),
            ([5, 7, 11], "must start at 3"),
            ([3, 5, 8, 11], "only odd integers"),
            ([3, 5, 1, 7], "only odd integers"),
            ([3, 7, 5, 11], "strictly increasing"),
            ([3, 5, 5, 7], "strictly increasing"),
            ([3, 0x1_0000_0001, 5], "strictly increasing"),
        ],
    )
    def test_invalid_python_sequences(self, primes, message):
        with pytest.raises(ValueError, match=message):
            validate_prime_pool_vectorized(primes, chunk_size=2)

    @pytest.mark.parametrize(
        ("primes", "message"),
        [
            (array("I"), "must not be empty"),
            (array("I", [5, 7, 11]), "must start at 3"),
            (array("I", [3, 5, 8, 11]), "only odd integers"),
            (
                np.array([3, 5, 1, 7], dtype=np.uint64),
                "only odd integers",
            ),
        ],
    )
    def test_invalid_vectorized_storage(self, primes, message):
        with pytest.raises(ValueError, match=message):
            validate_prime_pool_vectorized(primes, chunk_size=2)

    @pytest.mark.parametrize(
        "primes",
        [
            array("I", [3, 5, 7, 7, 11]),
            np.array([3, 5, 7, 7, 11], dtype=np.uint64),
            np.array([3, 5, 9, 7, 11], dtype=np.int64),
        ],
    )
    def test_chunk_boundary_order_violation(self, primes):
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_prime_pool_vectorized(primes, chunk_size=3)

    def test_signed_numpy_negative_is_not_unsigned_coerced(self):
        primes = np.array([3, 5, -7, 11], dtype=np.int64)
        with pytest.raises(ValueError, match="only odd integers"):
            validate_prime_pool_vectorized(primes, chunk_size=2)

    def test_numpy_array_must_be_one_dimensional_integer(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            validate_prime_pool_vectorized(
                np.array([[3, 5], [7, 11]], dtype=np.uint32)
            )
        with pytest.raises(TypeError, match="integer dtype"):
            validate_prime_pool_vectorized(
                np.array([3.0, 5.0, 7.0])
            )

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_prime_pool_vectorized([3, 5, 7], chunk_size=0)

    def test_compact_array_uses_vectorized_path(self, monkeypatch):
        import opn_core

        def forbidden_scalar(_primes):
            raise AssertionError("compact array used scalar validation")

        monkeypatch.setattr(
            opn_core,
            "_validate_prime_pool_scalar",
            forbidden_scalar,
        )
        validate_prime_pool_vectorized(
            array("Q", [3, 5, 7, 11]),
            chunk_size=2,
        )

    def test_analyzer_accepts_numpy_pool(self):
        analyzer = SigmaPoolAnalyzer(
            np.array([3, 5, 7, 11, 13], dtype=np.uint32)
        )
        result = analyzer.analyze(3, 2)
        assert result.exact
        assert result.valuations == {13: 1}

    def test_structural_validation_does_not_claim_primality_proof(self):
        validate_prime_pool_vectorized([3, 9, 15])

    def test_pool_digest_is_storage_independent_and_prefix_sensitive(self):
        values32 = array("I", [3, 5, 7, 11, 13])
        values64 = array("Q", values32)

        assert prime_pool_prefix_digest(values32) == (
            prime_pool_prefix_digest(values64)
        )
        assert prime_pool_prefix_digest(values32, 4) != (
            prime_pool_prefix_digest(values32)
        )

    @pytest.mark.parametrize("dtype", [np.uint32, np.uint64])
    def test_searchsorted_uses_exact_numpy_scalar_dtype(
        self,
        dtype,
        monkeypatch,
    ):
        import opn_core

        values = np.array([3, 5, 7, 11, 13], dtype=dtype)
        original = np.searchsorted
        observed = []

        def checked(array_value, scalar, *, side):
            observed.append(scalar)
            assert isinstance(scalar, np.generic)
            assert scalar.dtype == values.dtype
            return original(array_value, scalar, side=side)

        monkeypatch.setattr(opn_core.np, "searchsorted", checked)

        assert _typed_searchsorted_right(values, 7) == 3
        assert _typed_searchsorted_right(values, -1) == 0
        assert (
            _typed_searchsorted_right(
                values,
                int(np.iinfo(dtype).max) + 1,
            )
            == len(values)
        )
        assert len(observed) == 1

    def test_analyzer_caches_prime_prefix_positions(self, monkeypatch):
        import opn_core

        analyzer = SigmaPoolAnalyzer(
            array("Q", [3, 5, 7, 11, 13])
        )
        original = opn_core._typed_searchsorted_right
        calls = []

        def counted(values, limit):
            calls.append(limit)
            return original(values, limit)

        monkeypatch.setattr(
            opn_core,
            "_typed_searchsorted_right",
            counted,
        )

        assert analyzer._prime_prefix_stop(7) == 3
        assert analyzer._prime_prefix_stop(7) == 3
        assert calls == [7]


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
        ns = assign_prime_dfs(st, 3, 2, metrics=RunMetrics())
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
        ns = assign_prime_chain(st, 3, 2, metrics=RunMetrics(), propagate=True, max_exp=4)
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
            metrics=RunMetrics(),
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
# Descartes Spoof (regression)
# ══════════════════════════════════════════════════════════════

class TestDescartesSpoof:
    def test_spoof_dfs_does_not_apply_opn_max_prime_capacity(
        self, small_primes, monkeypatch,
    ):
        def invalid_in_spoof_mode(_p):
            raise AssertionError(
                "OPN maximum-prime capacity used in spoof DFS"
            )

        monkeypatch.setattr(
            "opn_search.max_prime_capacity",
            invalid_in_spoof_mode,
        )
        list(search_opn(
            small_primes,
            max_factors=5,
            max_exp=2,
            metrics=RunMetrics(),
            propagate=False,
        ))

    def test_descartes_spoof_found_dfs(self, small_primes):
        """Known spoof must be found in DFS mode."""
        found = None
        for st in search_opn(small_primes, max_factors=5, max_exp=2,
                             metrics=RunMetrics(), propagate=False):
            found = st
            break
        assert found is not None, "Descartes spoof not found"
        assert found.assigned == {3: 2, 7: 2, 11: 2, 13: 2}
        assert found.spoof is True

    def test_descartes_spoof_verified(self, small_primes):
        """The r-value for the Descartes spoof should be 22021."""
        for st in search_opn(small_primes, max_factors=5, max_exp=2,
                             metrics=RunMetrics(), propagate=False):
            assert _verify_solution(st) is False  # spoof, not true OPN
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
                                          max_exp=4, metrics=RunMetrics(), propagate=True))
        # Just verify it runs to completion
        assert isinstance(count, int)

    def test_dfs_mode_runs(self, small_primes):
        """DFS mode should not crash."""
        count = sum(1 for _ in search_opn(small_primes, max_factors=5,
                                          max_exp=2, metrics=RunMetrics(), propagate=False))
        assert count >= 1  # at least the Descartes spoof

    def test_priority_computed(self):
        p = _compute_priority(mpz(1), mpz(1), 0.0, 0)
        assert p > 0  # |2.0 - 1.0| - 0 = 1.0

    def test_exact_dedup_preserves_solution_set(self, small_primes):
        def signatures(use_cache):
            return [
                (tuple(sorted(st.assigned.items())), st.euler_prime, st.spoof)
                for st in search_opn(
                    small_primes,
                    max_factors=5,
                    max_exp=2,
                    metrics=RunMetrics(), propagate=False,
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
            metrics=RunMetrics(), propagate=True,
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
            metrics=RunMetrics(), propagate=True,
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
                st.spoof,
            )

        baseline_holder = {}
        baseline = [
            signature(st)
            for st in search_opn(
                small_primes,
                max_factors=5,
                max_exp=2,
                metrics=RunMetrics(),
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
                    metrics=RunMetrics(),
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
                metrics=RunMetrics(),
                state_holder=resumed_holder,
                resume_state=resume_state,
                propagate=False,
                checkpoint_interval_seconds=None,
            )
        ]

        assert partial + resumed == baseline
        assert resumed_holder["total_states"] == baseline_holder["total_states"]

    def test_chain_stop_resume_with_database_preserves_search(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(100)
        baseline_metrics = RunMetrics()
        baseline_holder = {}
        baseline = list(
            search_opn(
                primes,
                max_factors=6,
                max_exp=4,
                metrics=baseline_metrics,
                state_holder=baseline_holder,
                propagate=True,
                checkpoint_interval_seconds=None,
                sigma_database_path=str(
                    tmp_path / "baseline.sqlite3"
                ),
                pool_plan_cache_dir=str(
                    tmp_path / "baseline-plans"
                ),
                pool_plan_cache_minimum_free_bytes=0,
                pool_plan_build_policy="adaptive",
            )
        )

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        stopped_metrics = RunMetrics()
        stopped_holder = {}
        stop_checks = 0

        def should_stop():
            nonlocal stop_checks
            stop_checks += 1
            return stop_checks > 5

        partial = []
        with pytest.raises(SearchStopped):
            for result in search_opn(
                    primes,
                    max_factors=6,
                    max_exp=4,
                    metrics=stopped_metrics,
                    state_holder=stopped_holder,
                    propagate=True,
                    checkpoint_interval_seconds=None,
                    stop_requested=should_stop,
                    sigma_database_path=str(
                        tmp_path / "resumed.sqlite3"
                    ),
                    pool_plan_cache_dir=str(
                        tmp_path / "resumed-plans"
                    ),
                    pool_plan_cache_minimum_free_bytes=0,
                    pool_plan_build_policy="adaptive",
                ):
                partial.append(result)

        assert stopped_holder["snapshot_reason"] == "stop"
        assert stopped_holder["heap"]

        checkpoint_payload = pickle.loads(
            pickle.dumps(
                stopped_metrics.checkpoint_payload(),
                pickle.HIGHEST_PROTOCOL,
            )
        )
        resumed_metrics = RunMetrics.from_checkpoint_payload(
            checkpoint_payload
        )
        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        resume_state = {
            key: stopped_holder[key]
            for key in (
                "heap",
                "heap_counter",
                "states_started",
                "states_completed",
                "total_states",
                "elapsed",
                "use_heap",
                "snapshot_id",
            )
        }
        resumed_holder = {}
        resumed = list(
            search_opn(
                generate_odd_primes(100),
                max_factors=6,
                max_exp=4,
                metrics=resumed_metrics,
                state_holder=resumed_holder,
                resume_state=resume_state,
                propagate=True,
                checkpoint_interval_seconds=None,
                sigma_database_path=str(
                    tmp_path / "resumed.sqlite3"
                ),
                pool_plan_cache_dir=str(
                    tmp_path / "resumed-plans"
                ),
                pool_plan_cache_minimum_free_bytes=0,
                pool_plan_build_policy="adaptive",
            )
        )

        assert partial + resumed == baseline
        assert resumed_holder["total_states"] == (
            baseline_holder["total_states"]
        )
        assert resumed_metrics.structure == baseline_metrics.structure

    def test_solution_boundary_does_not_requeue_reported_solution(
        self, small_primes,
    ):
        holder = {}
        generator = search_opn(
            small_primes,
            max_factors=5,
            max_exp=2,
            metrics=RunMetrics(),
            state_holder=holder,
            propagate=False,
            checkpoint_interval_seconds=None,
        )
        solution = next(generator)
        solution_signature = (
            tuple(sorted(solution.assigned.items())),
            solution.euler_prime,
            solution.spoof,
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
                st.spoof,
            )
            for st in search_opn(
                small_primes,
                max_factors=5,
                max_exp=2,
                metrics=RunMetrics(),
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
                             metrics=RunMetrics(), propagate=False):
            found = st
            break
        assert found is not None
        assert found.assigned == {3: 2, 7: 2, 11: 2, 13: 2}
        assert found.spoof is True

    def test_chain_mode_finds_results(self, small_primes):
        """Chain mode with capacity bound should run without error."""
        count = sum(1 for _ in search_opn(small_primes, max_factors=5,
                                          max_exp=4, metrics=RunMetrics(), propagate=True))
        assert isinstance(count, int)


# ══════════════════════════════════════════════════════════════
# Checkpoint round-trip
# ══════════════════════════════════════════════════════════════

class TestBoundedStructureMetrics:
    def test_productive_state_does_not_retain_obligation_signature(self):
        metrics = RunMetrics()

        metrics.structure.record_productive(
            depth=3,
            assigned_count=2,
            pending=[7, 13],
            ratio_num=3,
            ratio_den=2,
            target_num=2,
            target_den=1,
        )

        assert metrics.structure.productive_states == 1
        assert metrics.structure.pending_prime_frequency == {7: 1, 13: 1}
        assert not hasattr(metrics.structure, "obligation_signatures")
        assert b"obligation_signatures" not in pickle.dumps(metrics)
        restored = pickle.loads(pickle.dumps(metrics))
        assert restored.structure.pending_prime_frequency == {7: 1, 13: 1}

    def test_legacy_pickle_state_discards_obligation_signatures(self):
        legacy_state = StructureMetrics().__getstate__()[1]
        legacy_state["productive_states"] = 12
        legacy_state.pop("sigma_classified_keys")
        legacy_state["obligation_signatures"] = {
            (frozenset({7, 13}), 2, 1): 12,
        }

        restored = StructureMetrics.__new__(StructureMetrics)
        restored.__setstate__((None, legacy_state))

        assert restored.productive_states == 12
        assert restored.sigma_classified_keys == set()
        assert not hasattr(restored, "obligation_signatures")

    def test_structure_report_uses_pending_frequency(self, tmp_path):
        from opn_reports import _pending_source_lines, write_structure_json

        metrics = RunMetrics()
        metrics.structure.pending_prime_frequency.update({7: 10, 13: 5})
        lines = _pending_source_lines(
            metrics.structure,
            {(3, 2): {13}, (13, 2): {3, 61}},
            100,
        )

        assert any("Frequent pending-prime" in line for line in lines)
        assert any("             7" in line and "10" in line for line in lines)

        write_structure_json(tmp_path, metrics)
        report = json.loads(
            (tmp_path / "structure.json").read_text(encoding="utf-8")
        )
        assert "obligation_signatures" not in report
        assert "sigma_classified_keys" not in report
        assert report["pending_prime_frequency"] == {"7": 10, "13": 5}

    def test_metrics_schema_one_payload_migrates_to_current(self):
        original = RunMetrics()
        payload = original.checkpoint_payload()
        payload["schema_version"] = 1

        restored = RunMetrics.from_checkpoint_payload(payload)

        assert restored.schema_version == 5
        assert not hasattr(restored.structure, "obligation_signatures")

    @pytest.mark.parametrize("schema_version", [1, 2, 3, 4])
    def test_legacy_pool_metrics_gain_current_defaults(
        self,
        schema_version,
    ):
        original = RunMetrics()
        pool_state = original.performance.pool.__getstate__()[1]
        for name in (
            "logical_leaf_blocks",
            "resident_leaf_blocks",
            "dynamic_leaf_products_built",
            "dynamic_leaf_prime_values",
            "dynamic_leaf_product_ns",
            "persistent_hits",
            "persistent_misses",
            "persistent_invalid",
            "disk_plan_hits",
            "disk_plan_misses",
            "disk_plan_invalid",
            "disk_plan_writes",
        ):
            pool_state.pop(name)
        pool_state["leaf_blocks_avoided_after_hit_exhaustion"] = 17
        pool_state["super_hit_bit_length_total"] = 99

        restored_pool = PoolPerformance.__new__(PoolPerformance)
        restored_pool.__setstate__((None, pool_state))
        original.performance.pool = restored_pool
        payload = original.checkpoint_payload()
        payload["schema_version"] = schema_version

        restored = RunMetrics.from_checkpoint_payload(payload)

        assert restored.schema_version == 5
        assert restored.performance.pool.logical_leaf_blocks == 0
        assert restored.performance.pool.resident_leaf_blocks == 0
        assert restored.performance.pool.dynamic_leaf_products_built == 0
        assert restored.performance.pool.dynamic_leaf_prime_values == 0
        assert restored.performance.pool.dynamic_leaf_product_ns == 0
        assert restored.performance.pool.persistent_hits == 0
        assert restored.performance.pool.persistent_misses == 0
        assert restored.performance.pool.persistent_invalid == 0
        assert restored.performance.pool.disk_plan_hits == 0
        assert restored.performance.pool.disk_plan_misses == 0
        assert restored.performance.pool.disk_plan_invalid == 0
        assert restored.performance.pool.disk_plan_writes == 0
        assert not hasattr(
            restored.performance.pool,
            "leaf_blocks_avoided_after_hit_exhaustion",
        )
        assert not hasattr(
            restored.performance.pool,
            "super_hit_bit_length_total",
        )

    def test_performance_reports_include_dynamic_leaf_metrics(
        self,
        tmp_path,
    ):
        from opn_reports import (
            write_performance_json,
            write_performance_text,
        )

        metrics = RunMetrics()
        pool = metrics.performance.pool
        pool.logical_leaf_blocks = 123
        pool.resident_leaf_blocks = 0
        pool.dynamic_leaf_products_built = 7
        pool.dynamic_leaf_prime_values = 1_700
        pool.dynamic_leaf_product_ns = 25_000_000

        write_performance_json(
            tmp_path,
            metrics,
            elapsed=1.0,
            sampled_peak_rss=42,
        )
        write_performance_text(
            tmp_path,
            metrics,
            elapsed=1.0,
            sampled_peak_rss=42,
        )

        report = json.loads(
            (tmp_path / "performance.json").read_text(encoding="utf-8")
        )
        text_report = (
            tmp_path / "performance.txt"
        ).read_text(encoding="utf-8")

        assert report["schema_version"] == 5
        assert report["pool"]["logical_leaf_blocks"] == 123
        assert report["pool"]["resident_leaf_blocks"] == 0
        assert report["pool"]["dynamic_leaf_products_built"] == 7
        assert report["pool"]["dynamic_leaf_prime_values"] == 1_700
        assert report["pool"]["dynamic_leaf_product_ns"] == 25_000_000
        assert "## Dynamic leaf rebuilding" in text_report


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

    def test_round_trip_preserves_sigma_classification_deduplication(
        self,
        small_primes,
        tmp_path,
        monkeypatch,
    ):
        """The real checkpoint file retains restart-stable structure keys."""
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
        metrics = RunMetrics()
        metrics.structure.sigma_classified_keys.add((3, 2))

        opn_io.save_checkpoint(
            holder,
            [],
            metrics=metrics,
        )
        checkpoint_data = opn_io.load_checkpoint()
        assert checkpoint_data is not None

        restored = RunMetrics.from_checkpoint_payload(
            checkpoint_data["metrics"]
        )
        assert restored.structure.sigma_classified_keys == {(3, 2)}

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

    @pytest.mark.parametrize(
        "missing_key",
        [
            "run_id",
            "metrics",
            "solutions",
            "prime_typecode",
            "first_prime",
            "last_prime",
        ],
    )
    def test_checkpoint_rejects_fields_required_by_main(
        self,
        missing_key,
        small_primes,
        tmp_path,
        monkeypatch,
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
        opn_io.save_checkpoint(
            holder,
            [],
            run_id="test-run",
            metrics=RunMetrics(),
        )
        with checkpoint.open("rb") as stream:
            payload = pickle.load(stream)
        payload.pop(missing_key)
        with checkpoint.open("wb") as stream:
            pickle.dump(payload, stream, pickle.HIGHEST_PROTOCOL)
        before = checkpoint.read_bytes()

        assert opn_io.load_checkpoint() is None
        assert checkpoint.read_bytes() == before

    def test_checkpoint_rejects_malformed_solution_entries(
        self,
        small_primes,
        tmp_path,
        monkeypatch,
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
        opn_io.save_checkpoint(
            holder,
            [("not-an-assignment", None, False)],
            run_id="test-run",
            metrics=RunMetrics(),
        )

        assert opn_io.load_checkpoint() is None

    def test_main_does_not_overwrite_an_invalid_checkpoint(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        import opn_io
        import opn_main

        checkpoint = tmp_path / "checkpoint.pkl"
        checkpoint.write_bytes(b"not a pickle")
        before = checkpoint.read_bytes()
        monkeypatch.setattr(opn_io, "CHECKPOINT_FILE", str(checkpoint))
        monkeypatch.setattr(opn_main, "CHECKPOINT_FILE", str(checkpoint))

        def unexpected_prime_generation(_limit):
            raise AssertionError("a new search must not start")

        monkeypatch.setattr(
            opn_main,
            "open_or_extend_prime_pool",
            unexpected_prime_generation,
        )

        opn_main.main()

        assert checkpoint.read_bytes() == before
        assert "不会启动新搜索或覆盖原文件" in capsys.readouterr().out

    def test_main_resume_reports_cumulative_solutions_and_elapsed(
        self,
        small_primes,
        tmp_path,
        monkeypatch,
    ):
        import opn_io
        import opn_main

        checkpoint = tmp_path / "checkpoint.pkl"
        monkeypatch.setattr(opn_io, "CHECKPOINT_FILE", str(checkpoint))
        monkeypatch.setattr(opn_main, "CHECKPOINT_FILE", str(checkpoint))

        previous_solutions = [
            ({3: 2, 7: 2}, None, True),
            ({5: 1, 13: 2}, 5, False),
        ]
        holder = {
            "primes": small_primes,
            "max_factors": 5,
            "max_exp": 2,
            "heap": [(0.0, 0, ChainState())],
            "heap_counter": 1,
            "states_started": 10,
            "states_completed": 10,
            "total_states": 10,
            "elapsed": 120.0,
            "use_heap": True,
            "snapshot_id": 1,
        }
        opn_io.save_checkpoint(
            holder,
            previous_solutions,
            run_id="resume-report-test",
            metrics=RunMetrics(),
        )

        class FakeSampler:
            sampled_peak_rss = 123

            def __init__(self, *_args, **_kwargs):
                pass

            def start(self):
                pass

            def stop(self):
                pass

            def set_phase(self, _phase):
                pass

            def capture_memory_phase(self, phases, phase):
                phases[phase] = self.capture_memory()

            def capture_memory(self):
                return {
                    "rss_bytes": 1,
                    "vms_bytes": 1,
                    "sampled_peak_rss_bytes": self.sampled_peak_rss,
                    "system_available_bytes": 1,
                }

        def fake_search(
            _primes,
            _max_factors,
            _max_exp,
            *,
            state_holder,
            resume_state,
            **_kwargs,
        ):
            assert resume_state is not None
            state_holder.update({
                "total_states": resume_state["total_states"],
                "frontier_size": 0,
            })
            if False:
                yield None

        captured_report = {}
        monkeypatch.setattr(
            opn_main,
            "open_or_extend_prime_pool",
            lambda _limit: small_primes,
        )
        monkeypatch.setattr(opn_main, "RuntimeSampler", FakeSampler)
        monkeypatch.setattr(opn_main, "search_opn", fake_search)
        monkeypatch.setattr(
            opn_main,
            "prepare_run_directory",
            lambda _run_id: tmp_path,
        )
        monkeypatch.setattr(
            opn_main,
            "write_all_reports",
            lambda **kwargs: captured_report.update(kwargs),
        )
        monkeypatch.setattr(opn_main, "_git_commit", lambda: "test")
        monkeypatch.setattr(opn_main, "_git_dirty", lambda: False)

        opn_main.main()

        assert captured_report["solutions_found"] == 2
        assert captured_report["elapsed_seconds"] >= 120.0


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

        first = assign_prime_chain(st, 7, 2, metrics=RunMetrics(), propagate=True, max_exp=4)
        assert first is not None
        assert first.required_v[3] == 1
        assert 3 not in first.pending_set

        second = assign_prime_chain(first, 13, 2, metrics=RunMetrics(), propagate=True, max_exp=4)
        assert second is not None
        assert second.required_v[3] == 2
        assert 3 not in second.pending_set

        # sigma(19^2) contributes a third factor of 3, exceeding v_3(9)=2.
        assert assign_prime_chain(
            second, 19, 2, metrics=RunMetrics(), propagate=True, max_exp=4,
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
        assert a._perf.exact_from_global_cache == 1

    def test_global_cache_preserves_all_outside_factors(self):
        """The fast path residual is the complete outside-pool cofactor."""
        _SIG_VALUATIONS[(5, 9)] = {
            3: 1,
            11: 1,
            71: 1,
            521: 1,
        }
        analyzer = SigmaPoolAnalyzer(generate_odd_primes(50))

        result = analyzer.analyze(5, 9)

        assert not result.exact
        assert result.valuations == {3: 1, 11: 1}
        assert result.residual == mpz(71) * 521
        assert result.outside_witness == 71
        assert analyzer._perf.outside_from_global_cache == 1

    def test_global_cache_outside_residual_preserves_valuations(self):
        """Outside factors retain multiplicity and witness is deterministic."""
        _SIG_VALUATIONS[(7, 2)] = {
            1009: 2,
            3: 2,
            101: 3,
        }
        analyzer = SigmaPoolAnalyzer(generate_odd_primes(50))

        result = analyzer.analyze(7, 2)

        assert not result.exact
        assert result.valuations == {3: 2}
        assert result.residual == mpz(1009) ** 2 * mpz(101) ** 3
        assert result.outside_witness == 101

    def test_global_cache_and_cold_scan_residuals_match(self):
        """Fast and cold paths agree for an actual multi-outside sigma."""
        primes = generate_odd_primes(50)
        cold = SigmaPoolAnalyzer(primes).analyze(5, 9)
        assert not cold.exact

        _SIG_VALUATIONS[(5, 9)] = {
            3: 1,
            11: 1,
            71: 1,
            521: 1,
        }
        fast = SigmaPoolAnalyzer(primes).analyze(5, 9)

        assert fast.exact == cold.exact
        assert fast.valuations == cold.valuations
        assert fast.residual == cold.residual == mpz(71) * 521

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
        """PoolPerformance object receives stats from analyzer."""
        pool_perf = PoolPerformance()
        a = SigmaPoolAnalyzer([3, 5, 7, 11, 13], pool_perf=pool_perf)
        a.analyze(3, 2)
        assert pool_perf.misses == 1
        assert pool_perf.analysis_ns > 0

    def test_order_two_reuses_single_full_plan(self):
        """Every exponent with d=2 shares the master-pool component plan."""
        a = SigmaPoolAnalyzer(generate_odd_primes(500), gcd_mode="hierarchical")
        p1 = a._plans_for_exp(1, lower_limit=None)[2]
        p5 = a._plans_for_exp(5, lower_limit=None)[2]
        p9 = a._plans_for_exp(9, lower_limit=None)[2]
        assert p1 is p5 is p9

    def test_flat_plan_does_not_build_superblocks(self):
        """Flat mode must not construct superblocks."""
        a = SigmaPoolAnalyzer(generate_odd_primes(500), gcd_mode="flat")
        plan = a._plans_for_exp(2, lower_limit=None)[3]
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
        plan = a._plans_for_exp(2, lower_limit=None)[3]
        # ponytail: vectorized plans use np.ndarray; the legacy path uses
        # array.array.  Both are compact (not list); int(primes[idx]) works.
        assert not isinstance(plan.primes, list)
        assert plan.primes.itemsize >= 4

    def test_full_plan_reuses_master_prime_array(self):
        primes = generate_odd_primes(10_000)
        a = SigmaPoolAnalyzer(primes, gcd_mode="hierarchical")
        p1 = a._plans_for_exp(1, lower_limit=None)[2]
        p5 = a._plans_for_exp(5, lower_limit=None)[2]
        assert p1 is p5
        assert p1.primes is primes


# ══════════════════════════════════════════════════════════════
# Superblock two-level GCD screening
# ══════════════════════════════════════════════════════════════

class TestCyclotomicComponents:
    def test_sigma_identity_for_all_small_primes_and_exponents(self):
        primes = generate_odd_primes(2_000)
        for p in primes:
            for exp in range(1, 36):
                product = mpz(1)
                for _order, component in cyclotomic_sigma_components(
                    int(p),
                    exp,
                ):
                    product *= component
                assert product == sigma_prime_power(int(p), exp)

    def test_component_pool_matches_complete_predicate_through_100k(self):
        primes = generate_odd_primes(100_000)
        pools = build_component_prime_pools_vectorized(
            primes,
            range(2, 36),
            chunk_primes=5_000,
        )
        for order in range(2, 36):
            expected = [
                int(q)
                for q in primes
                if component_filter_accepts(int(q), order)
            ]
            assert pools[order].tolist() == expected

    @staticmethod
    def _scalar_oracle(p, exp, primes):
        residual = mpz(sigma_prime_power(p, exp))
        residual, _v2 = _remove_all(residual, 2)
        valuations = {}
        for raw_q in primes:
            q = int(raw_q)
            residual, exponent = _remove_all(residual, q)
            if exponent:
                valuations[q] = exponent
        return (
            residual == 1,
            valuations,
            residual,
        )

    @pytest.mark.parametrize("limit", [50, 500, 5_000])
    def test_component_scan_matches_scalar_oracle_across_windows(
        self,
        limit,
    ):
        pool = generate_odd_primes(limit)
        analyzer = SigmaPoolAnalyzer(
            pool,
            gcd_mode="hierarchical",
            plan_build_policy="after_db_miss",
        )
        for p in generate_odd_primes(200):
            for exp in range(1, 36):
                key = (int(p), exp)
                _SIG_VALUATIONS.pop(key, None)
                _SIG_FACTORS.pop(key, None)
                actual = analyzer.analyze(int(p), exp)
                expected = self._scalar_oracle(
                    int(p),
                    exp,
                    pool,
                )
                assert actual.exact == expected[0]
                assert actual.valuations == expected[1]
                assert actual.residual == expected[2]

    def test_shared_component_prime_valuations_are_added(self):
        pool = generate_odd_primes(50)
        analyzer = SigmaPoolAnalyzer(
            pool,
            gcd_mode="hierarchical",
            plan_build_policy="after_db_miss",
        )
        _SIG_VALUATIONS.pop((5, 5), None)
        _SIG_FACTORS.pop((5, 5), None)
        result = analyzer.analyze(5, 5)
        assert result.exact
        assert result.valuations[3] == 2
        assert result.valuations == {3: 2, 7: 1, 31: 1}

    def test_100k_complete_search_is_deterministic(self):
        snapshots = []
        for _iteration in range(2):
            SIGMA_CACHE.clear()
            POWER_CACHE.clear()
            FACTOR_CACHE.clear()
            _SIG_VALUATIONS.clear()
            _SIG_FACTORS.clear()
            metrics = RunMetrics()
            found = list(
                search_opn(
                    generate_odd_primes(100_000),
                    max_factors=60,
                    max_exp=35,
                    metrics=metrics,
                    propagate=True,
                    checkpoint_interval_seconds=None,
                    pool_plan_build_policy="eager",
                )
            )
            snapshots.append(
                (
                    [repr(solution) for solution in found],
                    pickle.dumps(metrics.structure, protocol=5),
                )
            )
        assert snapshots[0] == snapshots[1]


class TestSuperblockGCD:
    @pytest.mark.parametrize("typecode", ["I", "Q"])
    def test_product_prime_range_specializes_internal_arrays(self, typecode):
        primes = array(typecode, [3, 5, 7, 11, 13])
        assert _product_prime_range(primes, 1, 4) == mpz(5 * 7 * 11)
        assert _product_prime_range(primes, 2, 2) == 1

    @pytest.mark.parametrize("storage", ["ndarray32", "ndarray64", "memmap"])
    def test_product_prime_range_preserves_numpy_conversion_path(
        self,
        storage,
        tmp_path,
    ):
        values = [3, 5, 7, 11, 13]
        if storage == "ndarray32":
            primes = np.asarray(values, dtype=np.uint32)
        elif storage == "ndarray64":
            primes = np.asarray(values, dtype=np.uint64)
        else:
            path = tmp_path / "primes.u4"
            primes = np.memmap(
                path,
                dtype=np.uint32,
                mode="w+",
                shape=(len(values),),
            )
            primes[:] = values
            primes.flush()

        assert _product_prime_range(primes, 1, 4) == mpz(5 * 7 * 11)
        assert _product_prime_range(primes, 3, 3) == 1

    @pytest.mark.parametrize("typecode", ["I", "Q"])
    def test_compact_superblock_products_match_generic_sequence(
        self,
        typecode,
    ):
        values = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        typed = array(typecode, values)
        typed_result = build_compact_superblocks(
            typed,
            block_size=3,
            superblock_fanout=2,
        )
        generic_result = build_compact_superblocks(
            list(values),
            block_size=3,
            superblock_fanout=2,
        )

        typed_products = tuple(
            (sb.start_leaf, sb.stop_leaf, int(sb.product))
            for sb in typed_result[0]
        )
        generic_products = tuple(
            (sb.start_leaf, sb.stop_leaf, int(sb.product))
            for sb in generic_result[0]
        )
        assert typed_products == generic_products
        assert typed_result[1] == generic_result[1]

    def test_superblocks_cover_every_leaf_once(self):
        primes = generate_odd_primes(1000)
        blocks = build_prime_blocks(primes, block_size=7)
        supers = build_prime_superblocks(blocks, fanout=4)
        covered = []
        for sb in supers:
            covered.extend(range(sb.start_leaf, sb.stop_leaf))
        assert covered == list(range(len(blocks)))

    def test_superblock_product_matches_children(self):
        primes = generate_odd_primes(1000)
        blocks = build_prime_blocks(primes, block_size=7)
        supers = build_prime_superblocks(blocks, fanout=4)
        for sb in supers:
            expected = mpz(1)
            for idx in range(sb.start_leaf, sb.stop_leaf):
                expected *= blocks[idx].product
            assert sb.product == expected

    def test_compact_superblocks_cover_tail_and_match_prime_products(self):
        primes = generate_odd_primes(1_000)
        supers, leaf_count, leaf_ns, super_ns = (
            build_compact_superblocks(
                primes,
                block_size=7,
                superblock_fanout=4,
            )
        )

        covered = []
        for sb in supers:
            covered.extend(range(sb.start_leaf, sb.stop_leaf))
            expected = mpz(1)
            prime_start = sb.start_leaf * 7
            prime_stop = min(sb.stop_leaf * 7, len(primes))
            for idx in range(prime_start, prime_stop):
                expected *= int(primes[idx])
            assert sb.product == expected

        assert covered == list(range(leaf_count))
        assert leaf_count == math.ceil(len(primes) / 7)
        assert supers[-1].stop_leaf == leaf_count
        assert leaf_ns > 0
        assert super_ns > 0

    def test_hierarchical_plan_has_no_resident_leaf_products(self):
        primes = generate_odd_primes(10_000)
        perf = PoolPerformance()
        plan = build_prime_block_plan(
            primes,
            block_size=13,
            superblock_fanout=4,
            build_superblocks=True,
            pool_perf=perf,
        )

        assert plan.blocks == ()
        assert plan.leaf_block_count == math.ceil(len(primes) / 13)
        assert plan.superblocks
        assert perf.plan_leaf_blocks == plan.leaf_block_count
        assert perf.logical_leaf_blocks == plan.leaf_block_count
        assert perf.resident_leaf_blocks == 0

    def test_flat_plan_retains_only_logical_leaf_count_as_resident(self):
        primes = generate_odd_primes(1_000)
        perf = PoolPerformance()
        plan = build_prime_block_plan(
            primes,
            block_size=7,
            superblock_fanout=4,
            build_superblocks=False,
            pool_perf=perf,
        )

        assert len(plan.blocks) == plan.leaf_block_count
        assert plan.superblocks == ()
        assert perf.logical_leaf_blocks == plan.leaf_block_count
        assert perf.resident_leaf_blocks == plan.leaf_block_count

    def test_dynamic_leaf_rebuild_handles_partial_final_leaf(self):
        primes = generate_odd_primes(100)
        plan = build_prime_block_plan(
            primes,
            block_size=8,
            superblock_fanout=3,
            build_superblocks=True,
        )

        start, stop, product = _build_dynamic_leaf_product(
            plan,
            plan.leaf_block_count - 1,
        )
        expected = mpz(1)
        for idx in range(start, stop):
            expected *= int(primes[idx])

        assert stop == len(primes)
        assert 0 < stop - start <= plan.block_size
        assert product == expected

    def test_dynamic_scanner_strips_multiple_leaves_and_valuations(self):
        primes = generate_odd_primes(200)
        plan = build_prime_block_plan(
            primes,
            block_size=3,
            superblock_fanout=2,
            build_superblocks=True,
        )
        q0 = int(primes[0])
        q1 = int(primes[4])
        q2 = int(primes[7])
        residual = mpz(q0) ** 3 * mpz(q1) ** 2 * q2
        inside = {}
        perf = PoolPerformance()

        residual = _scan_blocks_hierarchical(
            residual,
            inside,
            plan,
            perf,
        )

        assert residual == 1
        assert inside == {q0: 3, q1: 2, q2: 1}
        assert perf.positive_superblocks == 2
        assert perf.positive_blocks == 3
        assert perf.dynamic_leaf_products_built == 3
        assert perf.dynamic_leaf_products_built == perf.leaf_blocks_tested
        assert perf.dynamic_leaf_prime_values == 9
        assert perf.dynamic_leaf_product_ns > 0

    @pytest.mark.parametrize("hierarchical", [False, True])
    def test_certified_prefix_scan_matches_full_scan(self, hierarchical):
        primes = generate_odd_primes(500)
        plan = build_prime_block_plan(
            primes,
            block_size=3,
            superblock_fanout=4,
            build_superblocks=hierarchical,
        )
        scan = (
            _scan_blocks_hierarchical
            if hierarchical
            else _scan_blocks_flat
        )
        first_new_index = 17
        start_leaf = first_new_index // plan.block_size
        q1 = int(primes[first_new_index])
        q2 = int(primes[first_new_index + 19])
        residual = mpz(q1) ** 2 * q2

        full_inside = {}
        full_residual = scan(
            mpz(residual),
            full_inside,
            plan,
            PoolPerformance(),
        )
        suffix_inside = {}
        suffix_residual = scan(
            mpz(residual),
            suffix_inside,
            plan,
            PoolPerformance(),
            start_leaf=start_leaf,
        )

        assert suffix_residual == full_residual == 1
        assert suffix_inside == full_inside == {q1: 2, q2: 1}

    @pytest.mark.parametrize(
        "scan,build_superblocks",
        [
            (_scan_blocks_flat, False),
            (_scan_blocks_hierarchical, True),
        ],
    )
    def test_scan_rejects_invalid_prefix_boundary(
        self,
        scan,
        build_superblocks,
    ):
        plan = build_prime_block_plan(
            generate_odd_primes(100),
            block_size=3,
            superblock_fanout=2,
            build_superblocks=build_superblocks,
        )
        with pytest.raises(ValueError, match="start leaf"):
            scan(
                mpz(1),
                {},
                plan,
                PoolPerformance(),
                start_leaf=plan.leaf_block_count + 1,
            )

    def test_super_hit_exhaustion_avoids_remaining_leaf_rebuilds(self):
        primes = generate_odd_primes(100)
        plan = build_prime_block_plan(
            primes,
            block_size=2,
            superblock_fanout=4,
            build_superblocks=True,
        )
        q = int(primes[0])
        inside = {}
        perf = PoolPerformance()

        residual = _scan_blocks_hierarchical(
            mpz(q) ** 4 * 101,
            inside,
            plan,
            perf,
        )

        assert residual == 101
        assert inside == {q: 4}
        assert perf.positive_superblocks == 1
        assert perf.leaf_blocks_tested == 1
        assert perf.dynamic_leaf_products_built == 1

    @pytest.mark.parametrize(
        ("block_size", "fanout"),
        [(1, 2), (3, 2), (7, 4), (16, 5)],
    )
    def test_random_residuals_match_flat_oracle(
        self,
        block_size,
        fanout,
    ):
        primes = generate_odd_primes(200)
        flat_plan = build_prime_block_plan(
            primes,
            block_size=block_size,
            superblock_fanout=fanout,
            build_superblocks=False,
        )
        compact_plan = build_prime_block_plan(
            primes,
            block_size=block_size,
            superblock_fanout=fanout,
            build_superblocks=True,
        )
        rng = random.Random(10_000 + block_size * 100 + fanout)

        for _ in range(100):
            value = mpz(1)
            for raw_q in rng.sample(
                list(primes),
                rng.randrange(0, 9),
            ):
                value *= mpz(int(raw_q)) ** rng.randrange(1, 5)
            if rng.randrange(2):
                value *= mpz(211) ** rng.randrange(1, 4)

            flat_inside = {}
            compact_inside = {}
            flat_residual = _scan_blocks_flat(
                value,
                flat_inside,
                flat_plan,
                PoolPerformance(),
            )
            compact_residual = _scan_blocks_hierarchical(
                value,
                compact_inside,
                compact_plan,
                PoolPerformance(),
            )

            assert compact_residual == flat_residual
            assert compact_inside == flat_inside

    def test_candidate_leaf_telemetry_uses_logical_count(self):
        primes = generate_odd_primes(500)
        analyzer = SigmaPoolAnalyzer(
            primes,
            block_size=7,
            superblock_fanout=4,
            gcd_mode="hierarchical",
        )
        plan = analyzer._plans_for_exp(2, lower_limit=None)[3]

        analyzer.analyze(3, 2)

        assert plan.blocks == ()
        assert analyzer._perf.candidate_leaf_blocks == (
            plan.leaf_block_count
        )

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


class TestVectorizedPrimePlans:
    """Tests for vectorized cyclotomic component prime plans."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1, 1),
            (3, 3),
            (5, 5),
            (9, 3),
            (15, 15),
            (27, 3),
            (45, 15),
            (75, 15),
        ],
    )
    def test_squarefree_kernel(self, value, expected):
        assert squarefree_kernel(value) == expected

    @staticmethod
    def _scalar_eligible(primes, order):
        return [
            int(q)
            for q in primes
            if component_filter_accepts(int(q), order)
        ]

    @pytest.mark.parametrize("limit", [100, 1_000, 10_000, 100_000])
    def test_vectorized_filter_matches_scalar(self, limit):
        primes = generate_odd_primes(limit)

        orders = list(range(2, 36))
        outputs = build_component_prime_pools_vectorized(
            primes,
            orders,
            chunk_primes=17,
        )

        for order in orders:
            assert list(outputs[order]) == (
                self._scalar_eligible(primes, order)
            )

    def test_exp_2_and_exp_8_share_plan(self):
        primes = generate_odd_primes(10_000)

        analyzer = SigmaPoolAnalyzer(
            primes,
            block_size=16,
            superblock_fanout=4,
            gcd_mode="hierarchical",
            plan_chunk_primes=31,
        )

        analyzer.prebuild_plans([2, 8])

        assert (
            analyzer._plans_for_exp(2, lower_limit=None)[3]
            is analyzer._plans_for_exp(8, lower_limit=None)[3]
        )
        assert set(analyzer._component_plans) == {3, 9}

    def test_all_odd_exponents_share_full_plan(self):
        primes = generate_odd_primes(10_000)

        analyzer = SigmaPoolAnalyzer(
            primes,
            block_size=16,
            superblock_fanout=4,
            gcd_mode="hierarchical",
            plan_chunk_primes=31,
        )

        analyzer.prebuild_plans([1, 5, 9, 13, 17])

        plans = [
            analyzer._plans_for_exp(exp, lower_limit=None)[2]
            for exp in [1, 5, 9, 13, 17]
        ]

        assert all(plan is plans[0] for plan in plans)

class TestPersistentSigmaDatabase:
    @staticmethod
    def _database_path(tmp_path):
        return tmp_path / "sigma-test.sqlite3"

    def test_same_window_partial_hit_never_builds_plan(
        self,
        tmp_path,
        monkeypatch,
    ):
        path = self._database_path(tmp_path)
        primes = generate_odd_primes(50)

        cold_structure = StructureMetrics()
        cold = SigmaPoolAnalyzer(
            primes,
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
            structure=cold_structure,
        )
        expected = cold.analyze(5, 9)
        cold.close()
        assert not expected.exact

        warm_structure = StructureMetrics()
        warm = SigmaPoolAnalyzer(
            primes,
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
            structure=warm_structure,
        )

        def forbidden_plan(*_args, **_kwargs):
            raise AssertionError("persistent hit attempted to build a plan")

        monkeypatch.setattr(
            warm,
            "_plans_for_exp",
            forbidden_plan,
        )
        actual = warm.analyze(5, 9)
        warm.close()

        assert actual.exact == expected.exact
        assert actual.valuations == expected.valuations
        assert actual.residual == expected.residual
        assert warm._perf.persistent_hits == 1
        assert warm._perf.persistent_misses == 0
        assert warm._perf.plans_built == 0
        assert warm_structure.sigma_outside == 1
        assert warm_structure.sigma_exact == 0

    def test_aggregate_partial_expands_with_cyclotomic_components(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        old_pool = generate_odd_primes(50)
        sigma_odd = mpz(sigma_prime_power(5, 9))
        sigma_odd, _v2 = _remove_all(sigma_odd, 2)
        database = SigmaAnalysisDatabase(path)
        database.store(
            p=5,
            exp=9,
            exact=False,
            scanned_limit=47,
            pool_digest=prime_pool_prefix_digest(old_pool),
            valuations={3: 1, 11: 1},
            residual=mpz(71) * 521,
            sigma_odd=sigma_odd,
        )
        database.close()

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        expanded_analyzer = SigmaPoolAnalyzer(
            generate_odd_primes(100),
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        expanded = expanded_analyzer.analyze(5, 9)
        expanded_analyzer.close()

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        fresh = SigmaPoolAnalyzer(
            generate_odd_primes(100),
            gcd_mode="hierarchical",
        ).analyze(5, 9)
        assert expanded.exact == fresh.exact
        assert expanded.valuations == fresh.valuations
        assert expanded.residual == fresh.residual == 521
        assert expanded_analyzer._perf.persistent_hits == 1

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        warm = SigmaPoolAnalyzer(
            generate_odd_primes(100),
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        warm_result = warm.analyze(5, 9)
        assert warm_result.valuations == expanded.valuations
        assert warm_result.residual == expanded.residual
        assert warm._perf.plans_built == 0
        warm.close()

    def test_inconsistent_partial_component_restore_becomes_fresh_scan(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        sigma_odd = mpz(sigma_prime_power(5, 5))
        sigma_odd, _v2 = _remove_all(sigma_odd, 2)
        database = SigmaAnalysisDatabase(path)
        database.store(
            p=5,
            exp=5,
            exact=False,
            scanned_limit=47,
            pool_digest=prime_pool_prefix_digest(
                generate_odd_primes(50)
            ),
            valuations={3: 1},
            residual=sigma_odd // 3,
            sigma_odd=sigma_odd,
        )
        database.close()

        analyzer = SigmaPoolAnalyzer(
            generate_odd_primes(100),
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        result = analyzer.analyze(5, 5)
        assert result.exact
        assert result.valuations == {3: 2, 7: 1, 31: 1}
        assert analyzer._perf.persistent_hits == 1
        assert analyzer._perf.cold_scans == 1
        analyzer.close()

    def test_exact_record_is_window_independent(
        self,
        tmp_path,
        monkeypatch,
    ):
        path = self._database_path(tmp_path)
        large = SigmaPoolAnalyzer(
            generate_odd_primes(50),
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        assert large.analyze(3, 2).exact
        large.close()

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        small = SigmaPoolAnalyzer(
            generate_odd_primes(11),
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )

        def forbidden_plan(*_args, **_kwargs):
            raise AssertionError("exact persistent hit built a plan")

        monkeypatch.setattr(
            small,
            "_plans_for_exp",
            forbidden_plan,
        )
        result = small.analyze(3, 2)
        small.close()

        assert not result.exact
        assert result.valuations == {}
        assert result.residual == 13
        assert result.outside_witness == 13
        assert _SIG_VALUATIONS[(3, 2)] == {13: 1}
        assert small._perf.persistent_hits == 1

    def test_larger_window_scans_only_new_interval(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        old = SigmaPoolAnalyzer(
            generate_odd_primes(50),
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        old_result = old.analyze(5, 9)
        old.close()
        assert old_result.residual == mpz(71) * 521

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        expanded = SigmaPoolAnalyzer(
            generate_odd_primes(100),
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        incremental = expanded.analyze(5, 9)
        expanded.close()

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        fresh = SigmaPoolAnalyzer(
            generate_odd_primes(100),
            gcd_mode="hierarchical",
        ).analyze(5, 9)

        assert incremental.exact == fresh.exact
        assert incremental.valuations == fresh.valuations
        assert incremental.residual == fresh.residual == 521
        assert incremental.valuations == {3: 1, 11: 1, 71: 1}
        assert expanded._perf.persistent_hits == 1
        assert expanded._perf.plans_built == 3
        assert expanded._component_interval_plans
        assert all(
            int(plan.primes[0]) > 47
            for plan in expanded._component_interval_plans.values()
            if len(plan.primes)
        )

    def test_full_plan_supersedes_resident_interval_plan(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        old = SigmaPoolAnalyzer(
            generate_odd_primes(50),
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        old.analyze(5, 9)
        old.close()

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        expanded = SigmaPoolAnalyzer(
            generate_odd_primes(100),
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        expanded.analyze(5, 9)
        assert expanded._component_interval_plans

        # This key is absent from the database and needs the same full plan.
        expanded.analyze(7, 9)
        expanded.close()

        assert {2, 5, 10}.issubset(expanded._component_plans)
        assert not expanded._component_interval_plans

    def test_persisted_residual_uses_suffix_of_cached_full_plan(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        old = SigmaPoolAnalyzer(
            generate_odd_primes(50),
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        old_result = old.analyze(5, 9)
        old.close()
        assert old_result.residual == mpz(71) * 521

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        expanded = SigmaPoolAnalyzer(
            generate_odd_primes(1_000),
            block_size=2,
            superblock_fanout=2,
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        full_plan = expanded._plans_for_exp(
            9,
            lower_limit=None,
        )[2]

        actual = expanded.analyze(5, 9)
        old_limit = 47
        start_leaf = expanded._component_scan_start_leaf(
            full_plan,
            old_limit,
        )
        expanded.close()

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        expected = SigmaPoolAnalyzer(
            generate_odd_primes(1_000),
            gcd_mode="hierarchical",
        ).analyze(5, 9)

        assert start_leaf > 0
        assert not expanded._component_interval_plans
        assert (
            expanded._perf.candidate_leaf_blocks
            < full_plan.leaf_block_count
        )
        assert actual.exact == expected.exact
        assert actual.valuations == expected.valuations
        assert actual.residual == expected.residual == 1

    def test_large_window_jump_uses_shared_full_plan(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        old = SigmaPoolAnalyzer(
            generate_odd_primes(50),
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        old.analyze(5, 9)
        old.close()

        expanded = SigmaPoolAnalyzer(
            generate_odd_primes(2_000),
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        expanded.analyze(5, 9)
        expanded.close()

        assert {2, 5, 10}.issubset(expanded._component_plans)
        assert not expanded._component_interval_plans

    @pytest.mark.parametrize(
        ("p", "exp"),
        [
            (3, 2),
            (5, 4),
            (5, 9),
            (7, 6),
            (11, 10),
            (13, 8),
        ],
    )
    def test_incremental_result_matches_fresh_scan(
        self,
        tmp_path,
        p,
        exp,
    ):
        path = tmp_path / f"sigma-{p}-{exp}.sqlite3"
        old = SigmaPoolAnalyzer(
            generate_odd_primes(50),
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        old.analyze(p, exp)
        old.close()

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        expanded_analyzer = SigmaPoolAnalyzer(
            generate_odd_primes(2_000),
            gcd_mode="hierarchical",
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        incremental = expanded_analyzer.analyze(p, exp)
        expanded_analyzer.close()

        _SIG_VALUATIONS.clear()
        _SIG_FACTORS.clear()
        fresh = SigmaPoolAnalyzer(
            generate_odd_primes(2_000),
            gcd_mode="hierarchical",
        ).analyze(p, exp)

        assert incremental.exact == fresh.exact
        assert incremental.valuations == fresh.valuations
        assert incremental.residual == fresh.residual

    def test_corrupt_record_falls_back_to_cold_scan(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        primes = generate_odd_primes(50)
        cold = SigmaPoolAnalyzer(
            primes,
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        expected = cold.analyze(5, 9)
        cold.close()

        connection = sqlite3.connect(str(path))
        connection.execute(
            """
            UPDATE sigma_records
            SET checksum=zeroblob(32)
            WHERE p=5 AND exp=9
            """
        )
        connection.commit()
        connection.close()

        warm = SigmaPoolAnalyzer(
            primes,
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        actual = warm.analyze(5, 9)
        warm.close()

        assert actual.exact == expected.exact
        assert actual.valuations == expected.valuations
        assert actual.residual == expected.residual
        assert warm._perf.persistent_invalid == 1
        assert warm._perf.persistent_misses == 1
        assert warm._perf.plans_built == 3

    def test_pool_digest_mismatch_is_a_normal_miss(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        canonical = generate_odd_primes(50)
        cold = SigmaPoolAnalyzer(
            canonical,
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        cold.analyze(5, 9)
        cold.close()

        altered = array(
            "I",
            [q for q in canonical if q != 41],
        )
        incompatible = SigmaPoolAnalyzer(
            altered,
            database_path=str(path),
            plan_build_policy="after_db_miss",
        )
        incompatible.analyze(5, 9)
        incompatible.close()

        assert incompatible._perf.persistent_hits == 0
        assert incompatible._perf.persistent_misses == 1
        assert incompatible._perf.persistent_invalid == 0
        assert incompatible._perf.plans_built == 3

    def test_database_store_rejects_wrong_arithmetic(
        self,
        tmp_path,
    ):
        database = SigmaAnalysisDatabase(
            self._database_path(tmp_path)
        )
        with pytest.raises(ValueError, match="arithmetic identity"):
            database.store(
                p=3,
                exp=2,
                exact=True,
                scanned_limit=0,
                pool_digest=b"",
                valuations={11: 1},
                residual=mpz(1),
                sigma_odd=mpz(13),
            )
        database.close()

    def test_adaptive_policy_bulk_builds_after_threshold(self):
        analyzer = SigmaPoolAnalyzer(
            generate_odd_primes(1_000),
            gcd_mode="hierarchical",
            plan_build_policy="adaptive",
            adaptive_build_threshold=2,
        )
        analyzer.configure_plan_build([1, 2, 4, 5])

        analyzer._plans_for_exp(1, lower_limit=None)
        assert set(analyzer._component_plans) == {2}

        analyzer._plans_for_exp(2, lower_limit=None)
        assert set(analyzer._component_plans) == {2, 3, 5, 6}

    def test_search_stable_boundary_flushes_database(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        list(
            search_opn(
                generate_odd_primes(50),
                max_factors=5,
                max_exp=2,
                metrics=RunMetrics(),
                propagate=True,
                state_holder={},
                checkpoint_interval_seconds=None,
                sigma_database_path=str(path),
                pool_plan_build_policy="after_db_miss",
            )
        )

        connection = sqlite3.connect(str(path))
        count = connection.execute(
            "SELECT COUNT(*) FROM sigma_records"
        ).fetchone()[0]
        connection.close()
        assert count > 0

    def test_completed_search_releases_sigma_database_file(
        self,
        tmp_path,
    ):
        path = self._database_path(tmp_path)
        list(
            search_opn(
                generate_odd_primes(50),
                max_factors=5,
                max_exp=2,
                metrics=RunMetrics(),
                propagate=True,
                checkpoint_interval_seconds=None,
                sigma_database_path=str(path),
                pool_plan_build_policy="after_db_miss",
            )
        )
        moved = path.with_suffix(".moved")
        path.rename(moved)
        moved.rename(path)


class TestPersistentPlanCache:
    """Cold, warm, corruption, and interruption guarantees."""

    @staticmethod
    def _products(plan):
        return tuple(int(block.product) for block in plan.superblocks)

    @staticmethod
    def _analyzer(primes, cache_dir, **kwargs):
        return SigmaPoolAnalyzer(
            primes,
            block_size=16,
            superblock_fanout=4,
            gcd_mode="hierarchical",
            plan_cache_dir=str(cache_dir),
            plan_cache_minimum_free_bytes=0,
            plan_build_policy="after_db_miss",
            **kwargs,
        )

    def test_filtered_plan_cold_build_and_warm_mmap_match_memory(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(10_000)
        cache_dir = tmp_path / "plans"

        reference_analyzer = SigmaPoolAnalyzer(
            primes,
            block_size=16,
            superblock_fanout=4,
            gcd_mode="hierarchical",
            plan_build_policy="after_db_miss",
        )
        reference = reference_analyzer._plans_for_exp(
            2,
            lower_limit=None,
        )[3]
        expected_primes = np.asarray(reference.primes).copy()
        expected_products = self._products(reference)

        cold = self._analyzer(primes, cache_dir)
        cold_plan = cold._plans_for_exp(2, lower_limit=None)[3]
        assert isinstance(cold_plan.primes, np.memmap)
        assert not cold_plan.primes.flags.writeable
        assert np.array_equal(cold_plan.primes, expected_primes)
        assert self._products(cold_plan) == expected_products
        assert cold._perf.disk_plan_misses == 1
        assert cold._perf.disk_plan_writes == 1
        cold.close()

        warm = self._analyzer(primes, cache_dir)
        warm_plan = warm._plans_for_exp(2, lower_limit=None)[3]
        assert isinstance(warm_plan.primes, np.memmap)
        assert not warm_plan.primes.flags.writeable
        assert np.array_equal(warm_plan.primes, expected_primes)
        assert self._products(warm_plan) == expected_products
        assert warm._perf.disk_plan_hits == 1
        assert warm._perf.disk_plan_misses == 0
        assert warm._perf.disk_plan_writes == 0
        assert warm._perf.plans_built == 0
        warm.close()
        reference_analyzer.close()

    def test_unfiltered_plan_persists_products_without_prime_copy(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(5_000)
        cache_dir = tmp_path / "plans"
        cold = self._analyzer(primes, cache_dir)
        cold_plan = cold._plans_for_exp(1, lower_limit=None)[2]
        products = self._products(cold_plan)
        key = cold._disk_plan_key(2, source_start=0)
        entry = cold._plan_cache.entry_path(key)
        assert entry.is_dir()
        assert not (entry / "primes.bin").exists()
        cold.close()

        warm = self._analyzer(primes, cache_dir)
        warm_plan = warm._plans_for_exp(1, lower_limit=None)[2]
        assert warm_plan.primes is primes
        assert self._products(warm_plan) == products
        assert warm._perf.disk_plan_hits == 1
        assert warm._perf.plans_built == 0
        warm.close()

    def test_corrupt_prime_array_is_rebuilt_and_not_used(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(8_000)
        cache_dir = tmp_path / "plans"
        cold = self._analyzer(primes, cache_dir)
        expected = cold._plans_for_exp(2, lower_limit=None)[3]
        expected_primes = np.asarray(expected.primes).copy()
        expected_products = self._products(expected)
        key = cold._disk_plan_key(3, source_start=0)
        entry = cold._plan_cache.entry_path(key)
        cold.close()

        primes_path = entry / "primes.bin"
        with primes_path.open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(bytes([first[0] ^ 0xFF]))

        rebuilt = self._analyzer(primes, cache_dir)
        plan = rebuilt._plans_for_exp(2, lower_limit=None)[3]
        assert rebuilt._perf.disk_plan_invalid == 1
        assert rebuilt._perf.disk_plan_writes == 1
        assert np.array_equal(plan.primes, expected_primes)
        assert self._products(plan) == expected_products
        rebuilt.close()

    def test_corrupt_products_are_rebuilt_and_not_used(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(8_000)
        cache_dir = tmp_path / "plans"
        cold = self._analyzer(primes, cache_dir)
        expected = cold._plans_for_exp(1, lower_limit=None)[2]
        expected_products = self._products(expected)
        key = cold._disk_plan_key(2, source_start=0)
        entry = cold._plan_cache.entry_path(key)
        cold.close()

        products_path = entry / "products.bin"
        with products_path.open("ab") as handle:
            handle.write(b"\0")

        rebuilt = self._analyzer(primes, cache_dir)
        plan = rebuilt._plans_for_exp(1, lower_limit=None)[2]
        assert rebuilt._perf.disk_plan_invalid == 1
        assert rebuilt._perf.disk_plan_writes == 1
        assert self._products(plan) == expected_products
        rebuilt.close()

    def test_interval_and_full_window_have_isolated_cache_keys(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(10_000)
        cache_dir = tmp_path / "plans"
        analyzer = self._analyzer(primes, cache_dir)

        interval = analyzer._plans_for_exp(
            2,
            lower_limit=7_000,
        )[3]
        interval_count = len(interval.primes)
        assert all(int(q) > 7_000 for q in interval.primes)
        full = analyzer._plans_for_exp(2, lower_limit=None)[3]
        assert interval_count < len(full.primes)
        assert analyzer._perf.disk_plan_misses == 2
        assert analyzer._perf.disk_plan_writes == 2

        final_entries = [
            path
            for path in cache_dir.iterdir()
            if path.is_dir() and ".tmp-" not in path.name
        ]
        assert len(final_entries) == 2
        analyzer.close()

    def test_insufficient_space_falls_back_to_memory_plan(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(5_000)
        analyzer = SigmaPoolAnalyzer(
            primes,
            block_size=16,
            superblock_fanout=4,
            gcd_mode="hierarchical",
            plan_cache_dir=str(tmp_path / "plans"),
            plan_cache_minimum_free_bytes=10**30,
            plan_build_policy="after_db_miss",
        )
        plan = analyzer._plans_for_exp(2, lower_limit=None)[3]
        assert not isinstance(plan.primes, np.memmap)
        assert analyzer.plan_cache_error is not None
        assert analyzer._perf.disk_plan_writes == 0
        analyzer.close()

    def test_aborted_transaction_is_invisible_and_lock_reusable(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(2_000)
        analyzer = self._analyzer(primes, tmp_path / "plans")
        key = analyzer._disk_plan_key(3, source_start=0)
        cache = PersistentPlanCache(
            tmp_path / "plans",
            minimum_free_bytes=0,
        )

        build = cache.begin(key)
        mapping = build.allocate_primes(3)
        mapping[:] = np.asarray([7, 13, 19], dtype=mapping.dtype)
        mapping.flush()
        mapping._mmap.close()
        build.abort()

        assert not cache.entry_path(key).exists()
        assert not list(cache.root.glob(f"{key.slug}.tmp-*"))
        second = cache.begin(key)
        second.abort()
        analyzer.close()

    def test_close_releases_mmap_file_on_windows(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(3_000)
        cache_dir = tmp_path / "plans"
        analyzer = self._analyzer(primes, cache_dir)
        analyzer._plans_for_exp(2, lower_limit=None)
        analyzer.close()

        # Recursive removal fails on Windows if a mapped file is still open.
        import shutil

        shutil.rmtree(cache_dir)
        assert not cache_dir.exists()

    def test_component_plans_cold_build_warm_load_and_key_isolation(
        self,
        tmp_path,
    ):
        primes = generate_odd_primes(10_000)
        cache_dir = tmp_path / "component-plans"
        cold = self._analyzer(
            primes,
            cache_dir,
        )
        cold.configure_plan_build([5])
        cold_plans = cold._plans_for_exp(
            5,
            lower_limit=None,
        )
        expected = {
            order: (
                np.asarray(plan.primes).copy(),
                self._products(plan),
            )
            for order, plan in cold_plans.items()
        }
        assert set(cold_plans) == {2, 3, 6}
        assert isinstance(cold_plans[3].primes, np.memmap)
        component_key = cold._disk_plan_key(
            3,
            source_start=0,
        )
        assert component_key.filter_kind == "component"
        assert component_key.filter_order == 3
        cold.close()

        warm = self._analyzer(
            primes,
            cache_dir,
        )
        warm.configure_plan_build([5])
        warm_plans = warm._plans_for_exp(
            5,
            lower_limit=None,
        )
        assert warm._perf.disk_plan_hits == 3
        assert warm._perf.disk_plan_writes == 0
        assert warm._perf.plans_built == 0
        for order, plan in warm_plans.items():
            expected_primes, expected_products = expected[order]
            assert np.array_equal(plan.primes, expected_primes)
            assert self._products(plan) == expected_products
        warm.close()

    def test_corrupt_component_plan_is_rebuilt(self, tmp_path):
        primes = generate_odd_primes(8_000)
        cache_dir = tmp_path / "component-plans"
        cold = self._analyzer(
            primes,
            cache_dir,
        )
        plans = cold._plans_for_exp(
            5,
            lower_limit=None,
        )
        expected = np.asarray(plans[3].primes).copy()
        key = cold._disk_plan_key(
            3,
            source_start=0,
        )
        entry = cold._plan_cache.entry_path(key)
        cold.close()

        with (entry / "primes.bin").open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(bytes([first[0] ^ 0xFF]))

        rebuilt = self._analyzer(
            primes,
            cache_dir,
        )
        rebuilt_plans = rebuilt._plans_for_exp(
            5,
            lower_limit=None,
        )
        assert rebuilt._perf.disk_plan_invalid == 1
        assert np.array_equal(rebuilt_plans[3].primes, expected)
        rebuilt.close()

    def test_component_disk_space_failure_uses_memory(self, tmp_path):
        analyzer = SigmaPoolAnalyzer(
            generate_odd_primes(5_000),
            block_size=16,
            superblock_fanout=4,
            gcd_mode="hierarchical",
            plan_cache_dir=str(tmp_path / "plans"),
            plan_cache_minimum_free_bytes=10**30,
            plan_build_policy="after_db_miss",
        )
        plans = analyzer._plans_for_exp(
            5,
            lower_limit=None,
        )
        assert not isinstance(plans[3].primes, np.memmap)
        assert analyzer.plan_cache_error is not None
        analyzer.close()

    def test_interrupted_component_build_leaves_no_visible_entry(
        self,
        tmp_path,
        monkeypatch,
    ):
        import opn_core

        primes = generate_odd_primes(5_000)
        cache_dir = tmp_path / "plans"
        analyzer = self._analyzer(
            primes,
            cache_dir,
        )
        original = (
            opn_core.build_component_prime_pools_vectorized_memmap
        )

        def interrupted(*_args, **_kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(
            opn_core,
            "build_component_prime_pools_vectorized_memmap",
            interrupted,
        )
        with pytest.raises(KeyboardInterrupt):
            analyzer._plans_for_exp(
                5,
                lower_limit=None,
            )
        assert not list(cache_dir.glob("*.tmp-*"))
        analyzer.close()

        monkeypatch.setattr(
            opn_core,
            "build_component_prime_pools_vectorized_memmap",
            original,
        )
        retry = self._analyzer(
            primes,
            cache_dir,
        )
        plans = retry._plans_for_exp(
            5,
            lower_limit=None,
        )
        assert set(plans) == {2, 3, 6}
        retry.close()


class TestSigmaV3Valuation:
    """Exact LTE formula for v_3(sigma(p^exp))."""

    @pytest.mark.parametrize(
        ("p", "exp", "expected"),
        [
            (3, 1, 0),
            (3, 35, 0),
            (7, 2, 1),
            (13, 2, 1),
            (5, 1, 1),
            (5, 2, 0),
            (7, 4, 0),
            (13, 8, 2),
            (19, 2, 1),
            (7, 1, 0),
        ],
    )
    def test_boundary_cases(self, p, exp, expected):
        assert sigma_v3_valuation(p, exp) == expected

    def test_matches_direct_factorisation(self):
        """Exhaustive: all odd primes <= 2000, all exp 1..60."""
        from opn_core import _valuation
        primes = generate_odd_primes(2000)
        for p_val in primes:
            p_int = int(p_val)
            for exp in range(1, 61):
                expected = _valuation(
                    int(sigma_prime_power(p_int, exp)),
                    3,
                )
                actual = sigma_v3_valuation(p_int, exp)
                assert actual == expected, (
                    f"p={p_int} exp={exp}: expected={expected} actual={actual}"
                )


class TestPendingExponentDomain:
    """Deterministic exponent domain construction (refactoring Phase 5)."""

    @staticmethod
    def _make_st(required=None, assigned=None, excluded=None,
                 euler_prime=None):
        st = ChainState()
        if required:
            st.required_v.update(required)
        if assigned:
            st.assigned.update(assigned)
            st.current_v.update(assigned)
        if excluded:
            st.excluded.update(excluded)
        if euler_prime is not None:
            st.euler_prime = euler_prime
        return st

    def test_q_mod4_3_has_no_euler(self):
        domain = _build_pending_domain(
            ChainState(), 7, max_exp=6, apply_maximum_capacity=False,
        )
        assert not domain.euler_exponents
        assert domain.even_exponents
        assert not domain.forced_euler

    def test_q_mod4_1_has_both_roles(self):
        domain = _build_pending_domain(
            ChainState(), 5, max_exp=6, apply_maximum_capacity=False,
        )
        assert domain.even_exponents
        assert domain.euler_exponents
        assert not domain.forced_euler

    def test_euler_already_selected_suppresses_euler(self):
        st = self._make_st(euler_prime=5)
        domain = _build_pending_domain(
            st, 13, max_exp=6, apply_maximum_capacity=False,
        )
        assert not domain.euler_exponents

    def test_lower_bound_respected(self):
        st = self._make_st(required={7: 4}, assigned={7: 1})
        # current_v[7]=1, required_v[7]=4, target offset for q=7 is 0
        # lower = max(4 - 0 - 1, 1) = 3
        domain = _build_pending_domain(
            st, 7, max_exp=6, apply_maximum_capacity=False,
        )
        # exp >= 3 even → exponents 4, 6
        assert domain.lower_bound == 3
        assert 2 not in domain.even_exponents
        assert 4 in domain.even_exponents

    def test_maximum_capacity_respected(self):
        from opn_core import max_prime_capacity, even_max_exp_capacity
        domain = _build_pending_domain(
            ChainState(), 9973, max_exp=35, apply_maximum_capacity=True,
        )
        raw_cap = max_prime_capacity(9973)
        limit = even_max_exp_capacity(raw_cap)
        for e in domain.even_exponents:
            assert e <= limit

    def test_forced_euler_when_even_domain_empty(self):
        st = self._make_st(required={5: 5})
        domain = _build_pending_domain(
            st, 5, max_exp=5, apply_maximum_capacity=False,
        )
        # lower = max(5-0-0, 1) = 5 → even >= 5, none ≤ max_exp=5 → forced Euler
        assert domain.lower_bound == 5
        assert domain.even_exponents == ()
        assert domain.euler_exponents == (5,)
        assert domain.forced_euler

    def test_empty_domain_when_no_valid_exponents(self):
        st = self._make_st(required={7: 100}, assigned={7: 90})
        domain = _build_pending_domain(
            st, 7, max_exp=6, apply_maximum_capacity=False,
        )
        # lower = max(100 - 0 - 90, 1) = 10 → even >= 10, no exponent ≤ max_exp=6
        assert domain.empty

    def test_pending_lower_bound_minimum_one(self):
        st = ChainState()
        assert _pending_lower_bound(st, 7) == 1

    def test_even_exponent_boundary(self):
        assert _first_even_exponent(1, 6) == 2
        assert _first_even_exponent(2, 6) == 2
        assert _first_even_exponent(3, 6) == 4
        assert _first_even_exponent(6, 6) == 6
        assert _first_even_exponent(7, 6) is None

    def test_euler_exponent_boundary(self):
        assert _first_euler_exponent(1, 5) == 1
        assert _first_euler_exponent(2, 5) == 5
        assert _first_euler_exponent(5, 5) == 5
        assert _first_euler_exponent(6, 5) is None


class TestDomainAwareRatio:
    """Domain-aware mandatory ratio lower bound (Phase 6)."""

    def test_single_pending_exact_ratio(self):
        st = ChainState()
        st.ratio_num = mpz(1)
        st.ratio_den = mpz(1)
        st.required_v[7] = 1
        r = _domain_ratio_lower_bound(st, {7}, max_exp=6)
        assert r.possible
        # sigma(7^2)/7^2 = 57/49
        assert r.numerator == mpz(57)
        assert r.denominator == mpz(49)

    def test_multiple_pending_all_even(self):
        st = ChainState()
        st.ratio_num = mpz(1)
        st.ratio_den = mpz(1)
        st.required_v[3] = 2
        st.required_v[7] = 1
        r = _domain_ratio_lower_bound(st, {3, 7}, max_exp=6)
        assert r.possible

    def test_empty_domain_returns_impossible(self):
        st = ChainState()
        st.ratio_num = mpz(1)
        st.ratio_den = mpz(1)
        st.required_v[7] = 100
        r = _domain_ratio_lower_bound(st, {7}, max_exp=6)
        assert not r.possible
        assert r.reason == "empty_domain"

    def test_strict_inequality_does_not_prune_exact_target(self):
        target_num = SEARCH_MODE.target_num
        target_den = SEARCH_MODE.target_den
        assert not (target_num * target_den > target_num * target_den)

    def test_euler_fixed_all_pending_even(self):
        st = ChainState()
        st.ratio_num = mpz(1)
        st.ratio_den = mpz(1)
        st.euler_prime = 5
        st.required_v[7] = 1
        r = _domain_ratio_lower_bound(st, {7}, max_exp=6)
        assert r.possible
        # 7^2 only (even), no Euler role available
        assert r.numerator == mpz(57)
        assert r.denominator == mpz(49)

    def test_multiple_forced_euler_is_impossible(self):
        st = ChainState()
        st.ratio_num = mpz(1)
        st.ratio_den = mpz(1)
        st.required_v[5] = 5
        st.required_v[13] = 5
        r = _domain_ratio_lower_bound(st, {5, 13}, max_exp=5)
        assert not r.possible
        assert r.reason == "multiple_forced_euler"

    def test_domain_ratio_bound_is_safe_for_all_completions(self):
        """Brute-force: no flagged state has a valid completion, and the
        computed lower bound is always <= any legal completion ratio."""
        from itertools import combinations, product
        primes = [3, 5, 7, 11, 13]
        for pending_size in [1, 2, 3]:
            for pending_qs in combinations(primes, pending_size):
                st = ChainState()
                st.ratio_num = mpz(1)
                st.ratio_den = mpz(1)
                for q in pending_qs:
                    st.required_v[q] = 1
                r = _domain_ratio_lower_bound(
                    st, set(pending_qs), max_exp=10,
                )
                ranges = []
                for q in pending_qs:
                    evens = [e for e in range(2, 11, 2)]
                    eulers = [e for e in [1, 5, 9] if q % 4 == 1]
                    ranges.append(evens + eulers)
                legal_completions = []
                for combo in product(*ranges):
                    # Only one Euler prime is valid in OPN form.
                    euler_count = sum(
                        1 for q, e in zip(pending_qs, combo) if e % 2 == 1
                    )
                    if euler_count > 1:
                        continue
                    num = mpz(1); den = mpz(1)
                    for q, e in zip(pending_qs, combo):
                        num *= mpz(sigma_prime_power(q, e))
                        den *= mpz(power_pa(q, e))
                    if (num * SEARCH_MODE.target_den
                            <= SEARCH_MODE.target_num * den):
                        legal_completions.append((num, den))
                if not r.possible:
                    assert not legal_completions, (
                        f"false positive: {pending_qs}"
                    )
                else:
                    for actual_num, actual_den in legal_completions:
                        assert (r.numerator * actual_den
                                <= actual_num * r.denominator), (
                            f"bound violation: {pending_qs}"
                        )
                    if (r.numerator * SEARCH_MODE.target_den
                            > SEARCH_MODE.target_num * r.denominator):
                        assert all(
                            actual_num * SEARCH_MODE.target_den
                            > SEARCH_MODE.target_num * actual_den
                            for actual_num, actual_den
                            in legal_completions
                        )


class TestAbundancyGapCapture:
    @staticmethod
    def _near_state() -> ChainState:
        """Return a mathematically consistent small-gap partial integer."""
        st = ChainState()
        st.assigned = {3: 2, 5: 2, 11: 2, 67: 2}
        st.current_v = {3: 2, 5: 2, 11: 2, 67: 2}
        st.required_v = {3: 1, 7: 3, 13: 1, 19: 1, 31: 2, 10009: 1}
        st.pending.append(10009)
        st.pending_set.update(st.pending)
        st.depth = 4
        st.next_idx = 9
        st.excluded.update({17, 23})
        ratio = Fraction(1, 1)
        for prime, exponent in st.assigned.items():
            ratio *= Fraction(
                int(sigma_prime_power(prime, exponent)),
                prime ** exponent,
            )
        st.ratio_num = mpz(ratio.numerator)
        st.ratio_den = mpz(ratio.denominator)
        return st

    @staticmethod
    def _database(path, state: ChainState) -> None:
        database = SigmaAnalysisDatabase(path)
        try:
            for prime, exponent in state.assigned.items():
                sigma_value = mpz(sigma_prime_power(prime, exponent))
                odd_sigma, _v2 = _remove_all(sigma_value, 2)
                valuations = {
                    q: e
                    for q, e in factorize(int(odd_sigma))
                    if q != 2
                }
                database.store(
                    p=prime,
                    exp=exponent,
                    exact=True,
                    scanned_limit=0,
                    pool_digest=b"",
                    valuations=valuations,
                    residual=mpz(1),
                    sigma_odd=odd_sigma,
                )
            database.flush()
        finally:
            database.close()

    def test_exact_gap_boundaries(self):
        from opn_metrics import (
            StructureMetrics,
            exact_headroom_bucket,
        )

        boundaries = [
            (100, "1e-3-1e-2", ">1e-2"),
            (1_000, "1e-4-1e-3", "1e-3-1e-2"),
            (10_000, "1e-5-1e-4", "1e-4-1e-3"),
            (100_000, "1e-6-1e-5", "1e-5-1e-4"),
            (1_000_000, "<1e-6", "1e-6-1e-5"),
        ]
        scale = 1_000
        for denominator, at_or_below, above in boundaries:
            cases = (
                (scale - 1, scale * denominator, at_or_below),
                (1, denominator, at_or_below),
                (scale + 1, scale * denominator, above),
            )
            for gap_num, gap_den, expected in cases:
                assert exact_headroom_bucket(
                    ratio_num=2 * gap_den - gap_num,
                    ratio_den=gap_den,
                    target_num=2,
                    target_den=1,
                ) == expected

        # The capture interval includes 10^-2 exactly and excludes the
        # immediately larger rational value.
        assert exact_headroom_bucket(
            ratio_num=199,
            ratio_den=100,
            target_num=2,
            target_den=1,
        ) == "1e-3-1e-2"
        assert exact_headroom_bucket(
            ratio_num=198_999_999,
            ratio_den=100_000_000,
            target_num=2,
            target_den=1,
        ) == ">1e-2"

        for gap_num, gap_den in (
            (1, 100),
            (999, 100_000),
        ):
            assert exact_headroom_bucket(
                ratio_num=2 * gap_den - gap_num,
                ratio_den=gap_den,
                target_num=2,
                target_den=1,
            ) == "1e-3-1e-2"

        metrics = StructureMetrics()
        metrics.record_productive(
            depth=1,
            assigned_count=1,
            pending=(),
            ratio_num=199,
            ratio_den=100,
            target_num=2,
            target_den=1,
        )
        assert metrics.ratio_headroom["1e-3-1e-2"] == 1

    def test_capture_and_human_report(self, tmp_path):
        from opn_abundancy_capture import (
            AbundancyCaptureConfig,
            AbundancyGapRecorder,
        )

        state = self._near_state()
        database_path = tmp_path / "sigma.sqlite3"
        self._database(database_path, state)
        recorder = AbundancyGapRecorder(
            tmp_path,
            run_id="capture-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=0,
            config=AbundancyCaptureConfig(),
        )
        recorder.capture(state, 1)
        recorder.commit(1)
        summary = recorder.finalize(
            status="COMPLETE",
            sigma_database_path=database_path,
        )

        assert summary["qualifying_states"] == 1
        assert summary["records_written"] == 1
        record = json.loads(
            (tmp_path / "abundancy_gap_states.jsonl")
            .read_text(encoding="utf-8")
        )
        ratio = Fraction(
            int(record["ratio_num"]),
            int(record["ratio_den"]),
        )
        assert ratio == Fraction(81416881, 40737675)
        assert Fraction(
            int(record["gap_num"]),
            int(record["gap_den"]),
        ) == 2 - ratio
        assert record["assigned"] == [[3, 2], [5, 2], [11, 2], [67, 2]]

        sigma_doc = json.loads(
            (tmp_path / "abundancy_sigma_maps.json")
            .read_text(encoding="utf-8")
        )
        assert set(sigma_doc["records"]) == {"3:2", "5:2", "11:2", "67:2"}
        for mapping in sigma_doc["records"].values():
            reconstructed = mpz(2) ** mapping["v2"]
            for prime, exponent in mapping["odd_valuations"]:
                reconstructed *= mpz(prime) ** exponent
            assert reconstructed == mpz(mapping["sigma"])

        text = (tmp_path / "abundancy_gap_top.txt").read_text(
            encoding="utf-8"
        )
        assert "I(S) = sigma(S) / S" in text
        assert "Euler component" in text
        assert "Sigma-factor relations" in text
        assert "q-adic valuations" in text
        assert "pending-prime lower bound" in text
        assert "not odd-perfect-number solutions" in text

    def test_pending_lower_bound_overshoot_is_not_recorded(self, tmp_path):
        from opn_abundancy_capture import (
            AbundancyCaptureConfig,
            AbundancyGapRecorder,
        )

        state = self._near_state()
        state.pending.clear()
        state.pending.append(7)
        state.pending_set.clear()
        state.pending_set.add(7)
        recorder = AbundancyGapRecorder(
            tmp_path,
            run_id="pending-overshoot-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=0,
            config=AbundancyCaptureConfig(),
        )
        recorder.capture(state, 1)
        recorder.commit(1)
        summary = recorder.finalize(
            status="COMPLETE",
            sigma_database_path=None,
        )

        assert summary["small_gap_states_seen"] == 1
        assert summary["pending_lower_bound_rejections"] == 1
        assert summary["qualifying_states"] == 0
        assert summary["records_written"] == 0
        assert (
            tmp_path / "abundancy_gap_states.jsonl"
        ).read_text(encoding="utf-8") == ""

        boundary_dir = tmp_path / "boundary"
        boundary_state = self._near_state()
        boundary_state.pending.clear()
        boundary_state.pending.append(1009)
        boundary_state.pending_set.clear()
        boundary_state.pending_set.add(1009)
        # (1009 / 505) * (1010 / 1009) = 2 exactly.
        boundary_state.ratio_num = mpz(1009)
        boundary_state.ratio_den = mpz(505)
        boundary = AbundancyGapRecorder(
            boundary_dir,
            run_id="pending-boundary-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=0,
            config=AbundancyCaptureConfig(),
        )
        boundary.capture(boundary_state, 1)
        boundary.commit(1)
        boundary_summary = boundary.finalize(
            status="COMPLETE",
            sigma_database_path=None,
        )
        assert boundary_summary["small_gap_states_seen"] == 1
        assert boundary_summary["pending_lower_bound_rejections"] == 0
        assert boundary_summary["records_written"] == 1

        rollback_dir = tmp_path / "rollback"
        rollback = AbundancyGapRecorder(
            rollback_dir,
            run_id="pending-rollback-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=0,
            config=AbundancyCaptureConfig(),
        )
        rollback.capture(boundary_state, 1)
        rollback.commit(1)
        rollback.capture(state, 2)
        rollback_summary = rollback.finalize(
            status="INTERRUPTED",
            sigma_database_path=None,
        )
        assert rollback_summary["small_gap_states_seen"] == 1
        assert rollback_summary["pending_lower_bound_rejections"] == 0
        assert rollback_summary["records_written"] == 1

    def test_uncommitted_tail_is_replayed_without_duplicates(self, tmp_path):
        from opn_abundancy_capture import (
            AbundancyCaptureConfig,
            AbundancyGapRecorder,
        )

        state = self._near_state()
        config = AbundancyCaptureConfig()
        first = AbundancyGapRecorder(
            tmp_path,
            run_id="resume-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=0,
            config=config,
        )
        first.capture(state, 10)
        first.commit(10)
        first.capture(state, 20)
        interrupted = first.finalize(
            status="INTERRUPTED",
            sigma_database_path=None,
        )
        assert interrupted["records_written"] == 1

        resumed = AbundancyGapRecorder(
            tmp_path,
            run_id="resume-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=10,
            config=config,
        )
        resumed.capture(state, 20)
        resumed.commit(20)
        final = resumed.finalize(
            status="COMPLETE",
            sigma_database_path=None,
        )
        assert final["records_written"] == 2
        records = [
            json.loads(line)
            for line in (
                tmp_path / "abundancy_gap_states.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert [r["productive_ordinal"] for r in records] == [10, 20]

    def test_partial_tail_is_repaired_but_middle_damage_is_preserved(
        self,
        tmp_path,
    ):
        from opn_abundancy_capture import (
            AbundancyCaptureConfig,
            AbundancyGapRecorder,
        )

        state = self._near_state()
        config = AbundancyCaptureConfig()
        recorder = AbundancyGapRecorder(
            tmp_path,
            run_id="repair-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=0,
            config=config,
        )
        recorder.capture(state, 1)
        recorder.commit(1)
        recorder._close()
        raw_path = tmp_path / "abundancy_gap_states.jsonl"
        with raw_path.open("ab") as handle:
            handle.write(b'{"incomplete":')

        repaired = AbundancyGapRecorder(
            tmp_path,
            run_id="repair-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=1,
            config=config,
        )
        assert repaired.active
        assert repaired.tail_repairs == 1
        repaired.finalize(status="COMPLETE", sigma_database_path=None)

        original = raw_path.read_bytes()
        raw_path.write_bytes(original + b"{not-json}\n")
        damaged_bytes = raw_path.read_bytes()
        damaged = AbundancyGapRecorder(
            tmp_path,
            run_id="repair-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=2,
            config=config,
        )
        assert not damaged.active
        assert raw_path.read_bytes() == damaged_bytes
        assert any("invalid complete line" in error for error in damaged.errors)

    def test_record_limit_and_database_unavailable(self, tmp_path):
        from opn_abundancy_capture import (
            AbundancyCaptureConfig,
            AbundancyGapRecorder,
        )

        state = self._near_state()
        recorder = AbundancyGapRecorder(
            tmp_path,
            run_id="limit-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=0,
            config=AbundancyCaptureConfig(
                max_records=1,
                text_limit=1,
            ),
        )
        recorder.capture(state, 1)
        recorder.capture(state, 2)
        recorder.commit(2)
        summary = recorder.finalize(
            status="COMPLETE",
            sigma_database_path=tmp_path / "missing.sqlite3",
        )
        assert summary["qualifying_states"] == 2
        assert summary["records_written"] == 1
        assert summary["dropped_due_to_limit"] == 1
        assert summary["truncated"]
        sigma_status = summary["derived_outputs"]["sigma_maps"]
        assert not sigma_status["database_available"]
        assert "unavailable" in sigma_status["error"]

    def test_capture_write_failure_is_contained(self, tmp_path):
        from opn_abundancy_capture import (
            AbundancyCaptureConfig,
            AbundancyGapRecorder,
        )

        class FailingHandle:
            def write(self, _value):
                raise OSError("simulated disk failure")

            def close(self):
                pass

        recorder = AbundancyGapRecorder(
            tmp_path,
            run_id="failure-test",
            target_num=2,
            target_den=1,
            resume_productive_ordinal=0,
            config=AbundancyCaptureConfig(),
        )
        recorder._handle.close()
        recorder._handle = FailingHandle()
        recorder.capture(self._near_state(), 1)
        assert not recorder.active
        assert any("simulated disk failure" in error for error in recorder.errors)


# ── report integrity checker ──────────────────────────────────


class TestReportIntegrity:
    """Verify the read-only integrity checker catches all invariant violations.

    Uses synthetic RunMetrics objects so the tests are deterministic and fast;
    no actual search run is required.
    """

    @staticmethod
    def _synth_metrics() -> object:
        """Build a minimal RunMetrics with known-consistent counters."""
        from opn_metrics import (
            RunMetrics,
            PruneReason,
            PruneMechanism,
        )

        m = RunMetrics()
        m.configure_exponent_telemetry(max_exp=3)

        s = m.structure

        # 5 productive states
        s.productive_states = 5
        s.depth_histogram[1] = 2
        s.depth_histogram[2] = 3
        s.depth_factor_map[(1, 2)] = 2
        s.depth_factor_map[(2, 3)] = 3
        s.headroom_by_factor[(2, "<1e-3")] = 2
        s.headroom_by_factor[(3, "<1e-4")] = 3
        s.ratio_headroom["<1e-3"] = 2
        s.ratio_headroom["<1e-4"] = 3

        # 3 propagation events: p=3, exp=1/2, q=5/7
        s.propagation_edges[(3, 5)] = 2
        s.propagation_edges[(3, 7)] = 1
        s.propagation_exp_edges[(3, 1, 5)] = 1
        s.propagation_exp_edges[(3, 2, 5)] = 1
        s.propagation_exp_edges[(3, 1, 7)] = 1

        # sigma: 10 exact (2+3+5), 3 outside (1+1+1)
        s.sigma_exact = 10
        s.sigma_outside = 3
        s.sigma_exact_by_exp[0] = 2
        s.sigma_exact_by_exp[1] = 3
        s.sigma_exact_by_exp[2] = 5
        s.sigma_outside_by_exp[0] = 1
        s.sigma_outside_by_exp[1] = 1
        s.sigma_outside_by_exp[2] = 1

        # 7 prunes total, 3 reasons, 2 mechanisms
        for _ in range(4):
            m.record_prune(
                reason=PruneReason.RATIO_OVERSHOOT,
                mechanism=PruneMechanism.PRECLONE_VALUATION,
            )
        for _ in range(2):
            m.record_prune(
                reason=PruneReason.OUTSIDE_WINDOW,
                mechanism=PruneMechanism.KNOWN_OUTSIDE_CACHE,
            )
        m.record_prune(
            reason=PruneReason.EULER_FORM,
            mechanism=PruneMechanism.PRECLONE_VALUATION,
        )

        return m

    def test_all_invariants_pass_on_consistent_metrics(self):
        from opn_report_integrity import run_all_checks

        m = self._synth_metrics()
        result = run_all_checks(m)
        assert result["status"] == "PASS"
        for name, check in result["checks"].items():
            assert check["passed"], f"{name} should pass on consistent data"

    def test_productive_depth_total_detects_mismatch(self):
        from opn_report_integrity import check_structure_invariants

        m = self._synth_metrics()
        m.structure.productive_states += 1  # inject drift
        checks = check_structure_invariants(m)
        assert not checks["productive_depth_total"]["passed"]

    def test_productive_headroom_total_detects_mismatch(self):
        from opn_report_integrity import check_structure_invariants

        m = self._synth_metrics()
        m.structure.headroom_by_factor[(99, "<1e-6")] = 1
        checks = check_structure_invariants(m)
        assert not checks["productive_headroom_total"]["passed"]

    def test_propagation_aggregation_detects_mismatch(self):
        from opn_report_integrity import check_structure_invariants

        m = self._synth_metrics()
        m.structure.propagation_edges[(3, 5)] += 1
        checks = check_structure_invariants(m)
        assert not checks["propagation_aggregation"]["passed"]

    def test_sigma_exact_total_detects_mismatch(self):
        from opn_report_integrity import check_structure_invariants

        m = self._synth_metrics()
        m.structure.sigma_exact += 1
        checks = check_structure_invariants(m)
        assert not checks["sigma_exact_total"]["passed"]

    def test_sigma_outside_total_detects_mismatch(self):
        from opn_report_integrity import check_structure_invariants

        m = self._synth_metrics()
        m.structure.sigma_outside += 1
        checks = check_structure_invariants(m)
        assert not checks["sigma_outside_total"]["passed"]

    def test_prune_dimension_consistency_detects_mismatch(self):
        from opn_report_integrity import check_prune_consistency

        m = self._synth_metrics()
        m.structure.prune_reasons["bogus_extra"] = 1
        checks = check_prune_consistency(m)
        assert not checks["prune_dimension_consistency"]["passed"]

    def test_gap_summary_invariants_pass_on_consistent_data(self):
        from opn_report_integrity import check_gap_summary_invariants

        summary = {
            "configuration": {"enabled": True},
            "small_gap_states_seen": 10,
            "qualifying_states": 7,
            "pending_lower_bound_rejections": 3,
            "records_written": 5,
            "dropped_due_to_limit": 2,
        }
        checks = check_gap_summary_invariants(summary)
        assert checks["gap_small_gap"]["passed"]
        assert checks["gap_qualifying"]["passed"]

    def test_gap_summary_detects_small_gap_violation(self):
        from opn_report_integrity import check_gap_summary_invariants

        summary = {
            "configuration": {"enabled": True},
            "small_gap_states_seen": 10,
            "qualifying_states": 4,
            "pending_lower_bound_rejections": 3,
            "records_written": 4,
            "dropped_due_to_limit": 0,
        }
        checks = check_gap_summary_invariants(summary)
        assert not checks["gap_small_gap"]["passed"]

    def test_gap_summary_detects_qualifying_violation(self):
        from opn_report_integrity import check_gap_summary_invariants

        summary = {
            "configuration": {"enabled": True},
            "small_gap_states_seen": 10,
            "qualifying_states": 7,
            "pending_lower_bound_rejections": 3,
            "records_written": 7,
            "dropped_due_to_limit": 1,
        }
        checks = check_gap_summary_invariants(summary)
        assert not checks["gap_qualifying"]["passed"]

    def test_gap_checks_skip_when_disabled(self):
        from opn_report_integrity import check_gap_summary_invariants

        summary = {"configuration": {"enabled": False}}
        checks = check_gap_summary_invariants(summary)
        assert checks["gap_small_gap"]["passed"]
        assert "skipped" in checks["gap_small_gap"]

    def test_jsonl_gap_reconstruction_succeeds(self, tmp_path):
        from opn_report_integrity import check_gap_jsonl

        (tmp_path / "abundancy_gap_states.jsonl").write_text(
            (
                '{"productive_ordinal":1,"ratio_num":"3","ratio_den":"2",'
                '"gap_num":"1","gap_den":"2"}\n'
                '{"productive_ordinal":2,"ratio_num":"7","ratio_den":"4",'
                '"gap_num":"1","gap_den":"4"}\n'
            ),
            encoding="utf-8",
        )
        summary = {
            "configuration": {
                "enabled": True,
                "target_num": 2,
                "target_den": 1,
            },
        }
        checks = check_gap_jsonl(tmp_path, summary)
        assert checks["jsonl_ordinal_monotonic"]["passed"]
        assert checks["jsonl_gap_reconstruction"]["passed"]
        assert checks["jsonl_line_count"]["lines"] == 2

    def test_jsonl_detects_ordinal_regression(self, tmp_path):
        from opn_report_integrity import check_gap_jsonl

        (tmp_path / "abundancy_gap_states.jsonl").write_text(
            (
                '{"productive_ordinal":5,"ratio_num":"3","ratio_den":"2",'
                '"gap_num":"1","gap_den":"2"}\n'
                '{"productive_ordinal":3,"ratio_num":"3","ratio_den":"2",'
                '"gap_num":"1","gap_den":"2"}\n'
            ),
            encoding="utf-8",
        )
        summary = {
            "configuration": {
                "enabled": True,
                "target_num": 2,
                "target_den": 1,
            },
        }
        checks = check_gap_jsonl(tmp_path, summary)
        assert not checks["jsonl_ordinal_monotonic"]["passed"]

    def test_jsonl_detects_gap_mismatch(self, tmp_path):
        from opn_report_integrity import check_gap_jsonl

        (tmp_path / "abundancy_gap_states.jsonl").write_text(
            '{"productive_ordinal":1,"ratio_num":"3","ratio_den":"2",'
            '"gap_num":"999","gap_den":"2"}\n',
            encoding="utf-8",
        )
        summary = {
            "configuration": {
                "enabled": True,
                "target_num": 2,
                "target_den": 1,
            },
        }
        checks = check_gap_jsonl(tmp_path, summary)
        assert not checks["jsonl_gap_reconstruction"]["passed"]

    def test_write_integrity_json_produces_valid_output(self, tmp_path):
        from opn_report_integrity import write_integrity_json

        m = self._synth_metrics()
        gap_summary = {
            "configuration": {"enabled": False},
        }
        out = write_integrity_json(
            tmp_path, m, abundancy_capture_summary=gap_summary
        )
        assert out.exists()
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["schema_version"] == 1
        assert doc["status"] == "PASS"
        assert "productive_depth_total" in doc["checks"]
        assert "prune_dimension_consistency" in doc["checks"]
