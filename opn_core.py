"""
opn_core — arithmetic engine for odd-perfect-number search.

Provides prime generation, Brent-style Pollard-Rho factorisation,
σ(p^a) computation, ratio upper/lower bounds, cached sigma / power /
factorisation results, σ-factor-set precomputation, and all
user-configurable constants.
"""

import math
import random
from collections import Counter
from typing import Dict, List, Tuple

import gmpy2
from gmpy2 import mpz

# ── configuration ─────────────────────────────────────────────
CHECKPOINT_FILE  = "checkpoint_merged.pkl"
SOLUTIONS_FILE   = "solutions_merged.txt"
TELEMETRY_FILE   = "telemetry.txt"

MAX_PRIME         = 200       # largest odd prime considered
MAX_FACTORS       = 10        # max distinct prime factors in N
MAX_EXP           = 2         # max exponent (2 = a_i=1 restriction)
PROPAGATE         = False     # False = pseudo-solution DFS; True = true OPN chain
PROGRESS_INTERVAL = 1_000

# ── search mode (target + Euler + forced/excluded primes) ─────
# Single configuration point for OPN vs. friend-of-10 searches.
# All ratio comparisons read target from SEARCH_MODE.

from dataclasses import dataclass

@dataclass(frozen=True)
class SearchMode:
    """Immutable search-mode descriptor: target abundancy, Euler rule,
    forced primes (must be in N) and excluded primes (can never be in N)."""
    target_num: int = 2
    target_den: int = 1
    require_euler: bool = True       # True → OPN Euler form; False → all even
    forced_primes: dict = None       # {prime: min_exponent} dict
    excluded_primes: set = None      # frozenset of primes forbidden in N

    def __post_init__(self):
        object.__setattr__(self, 'forced_primes',
                           dict(self.forced_primes or {}))
        object.__setattr__(self, 'excluded_primes',
                           frozenset(self.excluded_primes or set()))


# Pre-defined modes
OPN_MODE = SearchMode(target_num=2, target_den=1, require_euler=True)
FRIEND_10_MODE = SearchMode(target_num=9, target_den=5, require_euler=False,
                            forced_primes={5: 2}, excluded_primes={3})

# Active mode — set by the friend preset, or override in code.
SEARCH_MODE = OPN_MODE

# ── friend-of-10 preset [INACTIVE] ──────────────────────────────
# Uncomment the block below to switch to friend-of-10 mode.
#
# MAX_PRIME    = 200
# MAX_FACTORS  = 9
# MAX_EXP      = 4
# PROPAGATE    = True
# SEARCH_MODE  = FRIEND_10_MODE

# resonance heuristic weights
RESONANCE_REUSE_W   = 1.5
RESONANCE_NEWF_W    = 0.7
RESONANCE_GIANT_W   = 0.15
PRIORITY_RESONANCE_W = 0.0  # disabled: resonance is structurally negative in chain mode
PRIORITY_DEPTH_W     = 0.01
HEAP_MAX_SIZE        = 200_000

# ── caches ────────────────────────────────────────────────────
SIGMA_CACHE:   Dict[Tuple[int, int], mpz] = {}
POWER_CACHE:   Dict[Tuple[int, int], int] = {}
FACTOR_CACHE:  Dict[int, List[Tuple[int, int]]] = {}
_SIG_FACTORS:  Dict[Tuple[int, int], set[int]] = {}
_SIG_VALUATIONS: Dict[Tuple[int, int], Dict[int, int]] = {}  # (p,a) → {q: v_q(σ(p^a))}

# ── prune telemetry ──────────────────────────────────────────
PRUNE_STATS:        "Counter[str]"           = Counter()
DEPTH_STATS:        "Counter[int]"           = Counter()
CLONE_STATS:        "Counter[str]"           = Counter()
CONTRADICTION_ATTR: "Counter[Tuple[int,str]]" = Counter()  # (prime, reason)

# ── structural telemetry ─────────────────────────────────────
PENDING_SIZE_HIST:  "Counter[int]"           = Counter()  # pending queue size
CASCADE_DEPTH_HIST: "Counter[int]"           = Counter()  # propagation chain length
PROPAGATION_EDGES:  "Counter[Tuple[int,int]]"= Counter()  # (source, introduced)
CLONE_PAYLOAD:      "Counter[int]"           = Counter()  # len(assigned) at clone
RATIO_HEADROOM:     "Counter[str]"           = Counter()  # 2-ratio bucketed
DEPTH_FACTOR_MAP:   "Counter[Tuple[int,int]]"= Counter()  # (depth, |f|) 2D histogram
HEADROOM_BY_FACTOR: "Counter[Tuple[int,str]]"= Counter()  # (|f|, headroom_bucket)
OBLIGATION_SIGS: "Counter[Tuple]"     = Counter()  # (frozen-pending, |f|, coarse-headroom)

# ── search-policy data (derived from telemetry) ───────────────
TOXIC_SKIP: set[int] = set()
EXCLUDE_EXP_4: set[int] = set()        # primes whose σ(p^4) factors all > MAX_PRIME
EXP4_FILTER_HITS: "Counter[int]" = Counter()  # per-prime filter verification


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
    """Populate ``_SIG_FACTORS`` and ``_SIG_VALUATIONS`` for all (p, a).

    _SIG_VALUATIONS stores the full {q: v_q(σ(p^a))} mapping, enabling
    pre-clone valuation contradiction checks that avoid wasted clones.
    """
    _SIG_FACTORS.clear()
    _SIG_VALUATIONS.clear()
    for p in primes:
        for a in range(2, max_exp + 1, 2):
            sig = int(sigma_prime_power(p, a))
            facs = factorize(sig)
            _SIG_FACTORS[(p, a)] = {q for q, _ in facs if q != 2}
            _SIG_VALUATIONS[(p, a)] = {q: e for q, e in facs if q != 2}
        for a in valid_euler_exponents(1, max_exp):
            sig = int(sigma_prime_power(p, a))
            facs = factorize(sig)
            _SIG_FACTORS[(p, a)] = {q for q, _ in facs if q != 2}
            _SIG_VALUATIONS[(p, a)] = {q: e for q, e in facs if q != 2}


# ── suffix-product precomputation (O(1) ratio bounds) ──────────

def precompute_suffix_bounds(primes: List[int]):
    """Build suffix arrays for O(1) ratio bounds.

    suffix_ub[i] = ∏_{j≥i} p_j/(p_j-1)   (upper bound: max ratio contribution)
    suffix_lb[i] = ∏_{j≥i} (p_j+1)/p_j   (lower bound: min ratio contribution)
    """
    n = len(primes)
    ub_num = [mpz(1)] * (n + 1)
    ub_den = [mpz(1)] * (n + 1)
    lb_num = [mpz(1)] * (n + 1)
    lb_den = [mpz(1)] * (n + 1)
    for i in range(n - 1, -1, -1):
        p = primes[i]
        ub_num[i] = ub_num[i + 1] * p
        ub_den[i] = ub_den[i + 1] * (p - 1)
        lb_num[i] = lb_num[i + 1] * (p + 1)
        lb_den[i] = lb_den[i + 1] * p
    return ub_num, ub_den, lb_num, lb_den


# ── ratio bounds (suffix-based, O(1) per query) ───────────────

def ratio_upper_bound(
    ratio_num: mpz, ratio_den: mpz,
    assigned: Dict[int, int], excluded: set[int],
    primes: List[int],
    next_idx: int = -1,
    suffix_ub_num: list = None,
    suffix_ub_den: list = None,
) -> Tuple[mpz, mpz]:
    """Maximum possible σ(N)/N — O(1) with suffix, O(|primes|) fallback."""
    if suffix_ub_num is not None and next_idx >= 0:
        n = len(primes)
        if next_idx >= n:
            return mpz(ratio_num), mpz(ratio_den)
        # full suffix product for all remaining primes
        num = mpz(ratio_num) * suffix_ub_num[next_idx]
        den = mpz(ratio_den) * suffix_ub_den[next_idx]
        # remove contribution of primes already decided (assigned / excluded)
        limit = primes[next_idx]
        for p in assigned:
            if p >= limit:
                num //= p           # factor p was in suffix_ub_num
        for p in excluded:
            if p >= limit:
                num //= p
        # den contains ∏(p-1); assigned/excluded primes' (p-1) factors
        # must also be removed — do it in a second pass over the same sets
        for p in assigned:
            if p >= limit:
                den //= (p - 1)
        for p in excluded:
            if p >= limit:
                den //= (p - 1)
        return num, den

    # fallback: O(|primes|)
    num = mpz(ratio_num)
    den = mpz(ratio_den)
    for p in primes:
        if p in assigned or p in excluded:
            continue
        num *= p
        den *= (p - 1)
    return num, den


def ratio_lower_bound(
    ratio_num: mpz, ratio_den: mpz, pending,
) -> Tuple[mpz, mpz]:
    """Minimum possible σ(N)/N — pending primes contribute ``(p+1)/p``."""
    num = mpz(ratio_num)
    den = mpz(ratio_den)
    for p in pending:
        num *= (p + 1)
        den *= p
    return num, den


# ── Touchard congruence pruning (O(1), no modulo) ──────────────

def check_touchard(euler_prime, assigned, excluded):
    """Check Touchard's theorem: any OPN satisfies N≡1(mod12) or N≡9(mod36).

    Returns True if the partial state is consistent with Touchard.
    Returns False if a contradiction is detected (prune this branch).

    Case A: 3 ∈ N → exponent must be even (3 ≡ 3 mod 4, can't be Euler)
                     → 3² | N → always satisfiable with remaining freedom.
    Case B: 3 ∉ N → N ≡ 1 (mod 12) → if Euler ≡ 2 (mod 3), then
                     the odd-exponent contribution ≡ 2 (mod 3), requiring
                     3 | N to reach N ≡ 0 or 1 (mod 3).
                     If 3 ∉ N and Euler ≡ 2 mod 3 → contradiction.
    """
    has_3 = 3 in assigned
    excluded_3 = 3 in excluded

    if has_3:
        exp3 = assigned[3]
        if exp3 % 2 == 1:
            return False      # 3 ≡ 3 mod 4, cannot be the Euler prime
        if exp3 < 2:
            return False
        return True

    if euler_prime is None:
        return True            # Euler not yet chosen — defer check

    if euler_prime % 3 == 2:
        if excluded_3:
            return False       # 3 explicitly excluded → impossible

    return True


def touchard_force_3(euler_prime, assigned, excluded):
    """Return True if prime 3 MUST be included in N based on Touchard.

    This happens when Euler ≡ 2 (mod 3) — the Euler prime's contribution
    to N mod 3 is 2, requiring 3 | N to reach N ≡ 0 or ≡ 1 (mod 3).
    """
    if 3 in assigned or 3 in excluded:
        return False
    if euler_prime is None:
        return False
    return euler_prime % 3 == 2


# ── toxic-skip seeding ───────────────────────────────────────

def compute_exclude_exp4(primes: List[int], max_exp: int,
                         max_prime: int) -> None:
    """Populate EXCLUDE_EXP_4: primes whose σ(p^4) factors ALL exceed
    MAX_PRIME.  These a=4 include branches deterministically produce
    cofactors that enter pending but can never be resolved within the
    current prime window.

    This is WINDOW-COMPLETE (not globally complete): a factor > MAX_PRIME
    today may be resolvable if MAX_PRIME is increased later (e.g. 197^4
    produces {661, 991, 2311} — all > 293 now, but 661 fits at MAX_PRIME≥661).
    The filter is parameter-sensitive; rerunning with a larger prime pool
    automatically reclassifies borderline primes.

    Conservative: only filters when ALL odd factors > max_prime.
    If at least one factor fits in the pool, the branch is kept.
    a=2 is NEVER filtered.
    """
    EXCLUDE_EXP_4.clear()
    if max_exp < 4:
        return
    for p in primes:
        facs = _SIG_FACTORS.get((p, 4))
        if facs is None:
            continue          # not precomputed — conservative, don't filter
        if facs and all(q > max_prime for q in facs):
            EXCLUDE_EXP_4.add(p)


def compute_toxic_skip_list() -> None:
    """Seed TOXIC_SKIP from contradiction attribution data."""
    from collections import Counter as _Counter
    excluded_counts: 'Counter[int]' = _Counter()
    for (q, reason), count in CONTRADICTION_ATTR.items():
        if reason == "excluded_pre":
            excluded_counts[q] += count
    TOXIC_SKIP.clear()
    TOXIC_SKIP.update(q for q, _ in excluded_counts.most_common(5))


# ═══════════════════════════════════════════════════════════════
# Stage 1: Precise next-prime interval bounds (Nielsen Prop. 3)
# ═══════════════════════════════════════════════════════════════

def next_prime_lower_bound(ratio_num, ratio_den, target_num, target_den):
    """Smallest prime p such that current_ratio × (p+1)/p ≤ target.

    Derived from: R × (p+1)/p ≤ T  ⇒  p ≥ R/(T−R).
    Uses integer ceiling division.
    """
    denom = target_num * ratio_den - target_den * ratio_num
    if denom <= 0:
        return 0   # already at or above target — no lower bound
    return (ratio_num * target_den + denom - 1) // denom


def next_prime_upper_bound(ratio_num, ratio_den, next_idx,
                           target_num, target_den,
                           suffix_ub_num, suffix_ub_den,
                           n_primes):
    """Largest prime p such that:
       current_ratio × p/(p−1) × suffix_ub_ratio ≥ target.

    Derived from: R × p/(p−1) × U ≥ T  ⇒  p ≤ 1/(1 − R×U/T).
    p is clamped to MAX_PRIME (window constraint).

    Returns 0 if no finite upper bound exists (interval is unbounded).
    """
    if next_idx >= n_primes:
        return 0
    # suffix ratio for primes AFTER the one we're assigning
    ub_n = suffix_ub_num[next_idx + 1] if next_idx + 1 < n_primes else mpz(1)
    ub_d = suffix_ub_den[next_idx + 1] if next_idx + 1 < n_primes else mpz(1)

    # R × p/(p−1) × ub_n/ub_d ≥ T
    # → p/(p−1) ≥ T×ub_d / (R×ub_n)
    # → p ≤ T×ub_d×ratio_den / (T×ub_d×ratio_den − target_den×ratio_num×ub_n)
    num = mpz(target_num) * ub_d * ratio_den
    den = (mpz(target_num) * ratio_den * ub_d
           - mpz(target_den) * ratio_num * ub_n)
    if den <= 0:
        return 0   # no finite upper bound
    return int(num // den)


# ═══════════════════════════════════════════════════════════════
# Stage 2: Fermat prime pruning (Nielsen Lemma 3.6-3.7)
# ═══════════════════════════════════════════════════════════════

FERMAT_PRIMES = {3, 5, 17, 257, 65537}


def check_fermat_contradiction(p, exp, assigned, excluded):
    """Nielsen Lemmas 3.6-3.7: Fermat prime high-exponent pruning.

    When a Fermat prime has exponent ≥ 80, the number of primes
    ≡ 1 (mod p) in N is bounded by a τ value derived from
    v_p(σ(p^a)).  If the count exceeds the bound, a prime > 10^11
    must exist — contradiction within a finite MAX_PRIME window.

    Returns True if contradiction detected (prune this branch).
    """
    if p not in FERMAT_PRIMES:
        return False
    if exp < 80:
        return False

    cnt_in = sum(1 for q in assigned if q % p == 1)
    cnt_ex = sum(1 for q in excluded if q % p == 1)
    total = cnt_in + cnt_ex

    # Conservative τ bound:  τ = floor(exp / 2) + 1
    tau = (exp // 2) + 1
    return total > tau


# ═══════════════════════════════════════════════════════════════
# Stage 3: Infinite-power approximation
# ═══════════════════════════════════════════════════════════════

INFINITE_POWER_LIMIT = 10**30


def is_prime_infinite(p, a):
    """Return True if p^a exceeds the factorisation threshold."""
    return pow(p, a) > INFINITE_POWER_LIMIT
