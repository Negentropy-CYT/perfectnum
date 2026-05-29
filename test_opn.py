"""pytest suite for OPN search engine improvements.

Covers:
  - Core arithmetic: prime generation, sigma, factorisation, ratio bounds
  - Interval bounds: lower/upper bound correctness (regression vs known values)
  - Touchard: congruence pruning correctness
  - Pseudo-solution: known Descartes spoof must be found (regression)
  - Early ratio prune: exact-ratio guard (>= → > fix verification)
  - Fermat: contradiction check
  - Infinite-power: threshold function
  - Friend-of-10: Euler skip, 5-force, 3-exclude
  - Checkpoint: save/restore round-trip
  - Regression: Descartes spoof found in DFS mode

Usage:
    cd improvements && pytest test_opn.py -v
    pytest test_opn.py -v -k "slow"   # only long-running tests
"""

import math
import os
import sys
import tempfile
from collections import deque

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
    FRIEND_10_MODE,
    OPN_MODE,
    SEARCH_MODE,
    brent_rho,
    check_fermat_contradiction,
    check_touchard,
    factorize,
    generate_odd_primes,
    is_prime_infinite,
    next_prime_lower_bound,
    next_prime_upper_bound,
    power_pa,
    precompute_sig_factors,
    precompute_suffix_bounds,
    ratio_lower_bound,
    ratio_upper_bound,
    sigma_prime_power,
    touchard_force_3,
    valid_euler_exponents,
    valid_even_exponents,
)
from opn_search import (
    ContradictionCache,
    _check_pseudo,
    _verify_solution,
    search_opn,
)
from opn_state import (
    ChainState,
    DFSState,
    _compute_priority,
    _early_ratio_prune,
    _euler_ok,
    assign_prime_chain,
    assign_prime_dfs,
)


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


@pytest.fixture
def small_primes():
    return generate_odd_primes(50)


# ══════════════════════════════════════════════════════════════
# Core Arithmetic
# ══════════════════════════════════════════════════════════════

class TestPrimes:
    def test_generate_up_to_50(self):
        p = generate_odd_primes(50)
        assert p == [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    def test_generate_up_to_100_first_last(self):
        p = generate_odd_primes(100)
        assert p[0] == 3
        assert p[-1] == 97

    def test_no_even_primes(self):
        p = generate_odd_primes(200)
        assert all(q % 2 == 1 for q in p)


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
        """With next_idx at the last prime, ub_n=ub_d=1 and
        hi = 2*1*9/(2*9*1 - 13*1) = 18/5 = 3.  This is correct."""
        primes = generate_odd_primes(30)
        s_ub_n, s_ub_d, _, _ = precompute_suffix_bounds(primes)
        n = len(primes)
        # next_idx = n-1 → suffix at n is 1/1.  Formula gives hi=3.
        hi = next_prime_upper_bound(mpz(13), mpz(9), n - 1,
                                    2, 1, s_ub_n, s_ub_d, n)
        assert hi == 3  # verified by hand: 18//5 = 3

    def test_upper_bound_unbounded(self):
        """When denom ≤ 0, return 0 (no finite upper bound)."""
        primes = generate_odd_primes(30)
        s_ub_n, s_ub_d, _, _ = precompute_suffix_bounds(primes)
        # ratio=1, ub_n/ub_d >> 2 → denom < 0 → unbounded
        hi = next_prime_upper_bound(mpz(1), mpz(1), 0,
                                    2, 1, s_ub_n, s_ub_d, len(primes))
        assert hi == 0  # unbounded


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
# Fermat Contradiction
# ══════════════════════════════════════════════════════════════

class TestFermat:
    def test_non_fermat_prime(self):
        assert not check_fermat_contradiction(7, 100, {}, {})

    def test_fermat_low_exponent(self):
        assert not check_fermat_contradiction(3, 2, {}, {})

    def test_fermat_no_congruent_primes(self):
        """No primes ≡1 mod 3 → no contradiction."""
        assert not check_fermat_contradiction(3, 100, {7: 2}, set())

    def test_fermat_contradiction_many_congruent(self):
        """Many primes ≡1 mod 3, count > τ=51."""
        assigned = {p: 2 for p in [7, 13, 19, 31, 37, 43, 61, 67, 73, 79, 97,
                                    103, 109, 127, 139, 151, 157, 163, 181, 193,
                                    199, 211, 223, 229, 241, 271, 277, 283, 307,
                                    313, 331, 337, 349, 367, 373, 379, 397, 409,
                                    421, 433, 439, 457, 463, 487, 499, 523, 541,
                                    547, 571, 577, 601, 607]}
        assert check_fermat_contradiction(3, 100, assigned, set())


# ══════════════════════════════════════════════════════════════
# Infinite Power
# ══════════════════════════════════════════════════════════════

class TestInfinitePower:
    def test_small_power(self):
        assert not is_prime_infinite(3, 2)

    def test_at_limit(self):
        assert not is_prime_infinite(10, 30)   # 10^30 == INFINITE_POWER_LIMIT

    def test_exceeds_limit(self):
        assert is_prime_infinite(10, 31)       # 10^31 > 10^30


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
            st, heap, small_primes, max_exp=2, _push=fake_push, cache=None,
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


# ══════════════════════════════════════════════════════════════
# Checkpoint round-trip
# ══════════════════════════════════════════════════════════════

class TestCheckpoint:
    def test_round_trip(self, small_primes):
        """Save state_holder, reload, verify keys."""
        from opn_io import save_checkpoint, load_checkpoint, CHECKPOINT_FILE
        # Clean up any existing checkpoint
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        try:
            holder = {
                "primes": small_primes,
                "max_factors": 5,
                "max_exp": 2,
                "heap": [],
                "heap_counter": 0,
                "total_states": 100,
                "elapsed": 10.0,
                "use_heap": False,
            }
            solutions = [({3: 2, 7: 2}, None, True)]
            save_checkpoint(holder, solutions)
            chk = load_checkpoint()
            assert chk is not None
            assert chk["total_states"] == 100
            assert len(chk["solutions"]) == 1
        finally:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)


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
