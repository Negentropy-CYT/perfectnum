"""
opn_core — arithmetic engine for odd-perfect-number search.

Provides prime generation, Brent-style Pollard-Rho factorisation,
σ(p^a) computation, ratio upper/lower bounds, cached sigma / power /
factorisation results, σ-factor-set precomputation, and all
user-configurable constants.
"""

import math
import random
from typing import Dict, List, Tuple

import gmpy2
from gmpy2 import mpz

# ── configuration ─────────────────────────────────────────────
CHECKPOINT_FILE  = "checkpoint_merged.pkl"
SOLUTIONS_FILE   = "solutions_merged.txt"

MAX_PRIME         = 500
MAX_FACTORS       = 8
MAX_EXP           = 4          # 2 = original a_i=1; 6+ for variable exponents
PROPAGATE         = False     # False = pseudo-solution; True = true OPN
PROGRESS_INTERVAL = 100_000

# resonance heuristic weights
RESONANCE_REUSE_W   = 1.5
RESONANCE_NEWF_W    = 0.7
RESONANCE_GIANT_W   = 0.15
PRIORITY_RESONANCE_W = 0.3
PRIORITY_DEPTH_W     = 0.01
HEAP_MAX_SIZE        = 200_000

# ── caches ────────────────────────────────────────────────────
SIGMA_CACHE:   Dict[Tuple[int, int], mpz] = {}
POWER_CACHE:   Dict[Tuple[int, int], int] = {}
FACTOR_CACHE:  Dict[int, List[Tuple[int, int]]] = {}
_SIG_FACTORS:  Dict[Tuple[int, int], set[int]] = {}


# ── prime generation ──────────────────────────────────────────
def generate_odd_primes(limit: int) -> List[int]:
    """Return all odd primes ≤ *limit* (Eratosthenes sieve)."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i: limit + 1: i] = [False] * (((limit - i * i) // i) + 1)
    return [p for p in range(3, limit + 1, 2) if sieve[p]]


# ── Brent Pollard-Rho factorisation ───────────────────────────
def brent_rho(n: int) -> int:
    """Return a non-trivial factor of *n* (Brent's cycle detection)."""
    if n % 2 == 0:
        return 2
    if gmpy2.is_prime(n):
        return n
    while True:
        y = random.randrange(1, n - 1)
        c = random.randrange(1, n - 1)
        m = random.randrange(1, n - 1)
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g


def _factor_recursive(n: int, out: List[int]) -> None:
    """Recursively split *n* into prime factors, appending to *out*."""
    if n == 1:
        return
    if gmpy2.is_prime(n):
        out.append(int(n))
        return
    d = brent_rho(int(n))
    _factor_recursive(d, out)
    _factor_recursive(n // d, out)


def factorize(n: int) -> List[Tuple[int, int]]:
    """Return sorted list of ``(prime, exponent)`` for *n* (cached)."""
    n = int(n)
    if n in FACTOR_CACHE:
        return FACTOR_CACHE[n]
    flat: List[int] = []
    _factor_recursive(n, flat)
    mp: Dict[int, int] = {}
    for f in flat:
        mp[f] = mp.get(f, 0) + 1
    res = sorted(mp.items())
    FACTOR_CACHE[n] = res
    return res


# ── sigma & power (cached) ────────────────────────────────────
def sigma_prime_power(p: int, a: int) -> mpz:
    """σ(p^a) = (p^(a+1)-1) / (p-1)."""
    key = (p, a)
    if key in SIGMA_CACHE:
        return SIGMA_CACHE[key]
    val = (mpz(p) ** (a + 1) - 1) // (p - 1)
    SIGMA_CACHE[key] = val
    return val


def power_pa(p: int, a: int) -> int:
    """p^a (as Python int; fits the search range)."""
    key = (p, a)
    if key in POWER_CACHE:
        return POWER_CACHE[key]
    val = p ** a
    POWER_CACHE[key] = val
    return val


# ── exponent domains ──────────────────────────────────────────
def valid_even_exponents(lb: int, max_exp: int) -> List[int]:
    """Even exponents in [max(lb, 2) … max_exp]."""
    if lb % 2:
        lb += 1
    if lb < 2:
        lb = 2
    return list(range(lb, max_exp + 1, 2))


def valid_euler_exponents(lb: int, max_exp: int) -> List[int]:
    """Euler exponents ≡ 1 (mod 4) in [max(lb, 1) … max_exp]."""
    x = max(lb, 1)
    while x % 4 != 1:
        x += 1
    return list(range(x, max_exp + 1, 4))


# ── σ-factor sets (precomputation for resonance heuristic) ────
def precompute_sig_factors(primes: List[int], max_exp: int) -> None:
    """Populate ``_SIG_FACTORS[(p, a)] = {odd prime factors of σ(p^a)}``."""
    _SIG_FACTORS.clear()
    for p in primes:
        for a in range(2, max_exp + 1, 2):
            sig = int(sigma_prime_power(p, a))
            _SIG_FACTORS[(p, a)] = {q for q, _ in factorize(sig) if q != 2}
        for a in valid_euler_exponents(1, max_exp):
            sig = int(sigma_prime_power(p, a))
            _SIG_FACTORS[(p, a)] = {q for q, _ in factorize(sig) if q != 2}


# ── ratio bounds ──────────────────────────────────────────────
def ratio_upper_bound(
    ratio_num: mpz, ratio_den: mpz,
    assigned: Dict[int, int], excluded: set[int],
    primes: List[int],
) -> Tuple[mpz, mpz]:
    """Maximum possible σ(N)/N from current state.

    Assumes every available (non-assigned, non-excluded) prime
    contributes its asymptotic maximum ``p/(p-1)``.
    """
    num = mpz(ratio_num)
    den = mpz(ratio_den)
    for p in primes:
        if p in assigned or p in excluded:
            continue
        num *= p
        den *= (p - 1)
    return num, den


def ratio_lower_bound(
    ratio_num: mpz, ratio_den: mpz, pending,  # Deque[int]
) -> Tuple[mpz, mpz]:
    """Minimum possible σ(N)/N — pending primes contribute ``(p+1)/p``."""
    num = mpz(ratio_num)
    den = mpz(ratio_den)
    for p in pending:
        num *= (p + 1)
        den *= p
    return num, den
