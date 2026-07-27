"""
opn_core — arithmetic engine for odd-perfect-number search.

Provides prime generation, Brent-style Pollard-Rho factorisation,
σ(p^a) computation, ratio upper/lower bounds, cached sigma / power /
factorisation results, σ-factor-set precomputation, and all
user-configurable constants.
"""

import math
import random
import time
from array import array
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import gmpy2
import numpy as np
from gmpy2 import mpz

# ── configuration ─────────────────────────────────────────────
CHECKPOINT_FILE  = "checkpoint_merged.pkl"
SOLUTIONS_FILE   = "solutions_merged.txt"
TELEMETRY_FILE   = "telemetry.txt"

MAX_PRIME         = 5000000000     # largest odd prime considered
MAX_FACTORS       = 60         # max distinct prime factors in N
MAX_EXP           = 18         # max exponent (2 = a_i=1 restriction)
PROPAGATE         = True     # False = pseudo-solution DFS; True = true OPN chain
PROGRESS_INTERVAL = 1_000
CHECKPOINT_INTERVAL_SECONDS = 300.0  # periodic save at a stable search boundary
ENABLE_FERMAT_DEBT = False
POOL_GCD_MODE = "hierarchical"          # "flat" or "hierarchical"
POOL_SUPERBLOCK_FANOUT = 16

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


# ── pool-analysis result types ─────────────────────────────────

@dataclass(slots=True)
class SigmaPoolAnalysis:
    """Result of analysing the odd part of σ(p^a) against a prime pool.

    exact=True: *valuations* is the complete odd-prime valuation map
    and *residual* is 1.

    exact=False: *residual* > 1 and has no prime factor in the pool.
    *valuations* is only the pool-internal partial map and must not
    be used for factor-chain propagation.
    """
    exact: bool
    valuations: Dict[int, int]
    residual: mpz
    outside_witness: Optional[int] = None


@dataclass(slots=True, frozen=True)
class PrimeBlock:
    """A range over a shared prime array with precomputed product for gcd."""
    start: int
    stop: int
    product: mpz


@dataclass(slots=True, frozen=True)
class PrimeSuperBlock:
    """A consecutive range of leaf PrimeBlock objects."""
    start: int          # inclusive leaf-block index
    stop: int           # exclusive leaf-block index
    product: mpz        # product of every child block product


@dataclass(slots=True, frozen=True)
class PrimeBlockPlan:
    """Compact prime data plus a two-level GCD screening structure."""
    primes: Sequence[int]
    blocks: Tuple[PrimeBlock, ...]
    superblocks: Tuple[PrimeSuperBlock, ...] = ()


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

# ── caches ────────────────────────────────────────────────────
SIGMA_CACHE:   Dict[Tuple[int, int], mpz] = {}
POWER_CACHE:   Dict[Tuple[int, int], int] = {}
FACTOR_CACHE:  Dict[int, List[Tuple[int, int]]] = {}
_SIG_FACTORS:  Dict[Tuple[int, int], set[int]] = {}
_SIG_VALUATIONS: Dict[Tuple[int, int], Dict[int, int]] = {}  # (p,a) → {q: v_q(σ(p^a))}
SIGMA_MAP_STATS: "Counter[str]" = Counter()  # hits, misses, factor_seconds
SIGMA_MISS_TIMES: "List[Tuple[int,int,int,float,int]]" = []  # (p, a, sigma_bits, seconds, max_odd_factor)
_TOTIENT_CACHE: Dict[int, int] = {}
_CAPACITY_CACHE: Dict[int, int] = {}

# ── prune telemetry ──────────────────────────────────────────
PRUNE_STATS:        "Counter[str]"           = Counter()
DEPTH_STATS:        "Counter[int]"           = Counter()
CLONE_STATS:        "Counter[str]"           = Counter()
CONTRADICTION_ATTR: "Counter[Tuple[int,str]]" = Counter()  # (prime, reason)

# ── structural telemetry ─────────────────────────────────────
PENDING_SIZE_HIST:  "Counter[int]"           = Counter()  # pending queue size
CASCADE_DEPTH_HIST: "Counter[int]"           = Counter()  # propagation chain length
PROPAGATION_EDGES:  "Counter[Tuple[int,int]]"= Counter()  # (source, introduced)
PROPAGATION_EXP_EDGES: "Counter[Tuple[int,int,int]]" = Counter()  # (source, exp, introduced)
CLONE_PAYLOAD:      "Counter[int]"           = Counter()  # len(assigned) at clone
RATIO_HEADROOM:     "Counter[str]"           = Counter()  # 2-ratio bucketed
DEPTH_FACTOR_MAP:   "Counter[Tuple[int,int]]"= Counter()  # (depth, |f|) 2D histogram
HEADROOM_BY_FACTOR: "Counter[Tuple[int,str]]"= Counter()  # (|f|, headroom_bucket)
OBLIGATION_SIGS: "Counter[Tuple]"     = Counter()  # (frozen-pending, |f|, coarse-headroom)
PENDING_PRIME_FREQ: "Counter[int]"    = Counter()  # global pending-prime histogram
OUTSIDE_WINDOW_SOURCE: "Counter[Tuple[int,int,int]]" = Counter()  # (p,exp,q) that forced q>window
SIGMA_POOL_STATS:   "Counter[str]"           = Counter()  # pool analysis telemetry
OUTSIDE_POOL_SOURCES: "Counter[Tuple[int,int,int]]" = Counter()  # (p,exp,residual_bits)
WINDOW_KNOWN_HITS: "Counter[Tuple[int,int]]" = Counter()  # (p,exp) → times reused via is_known_outside
PERF_STATS: "Counter[str]" = Counter()
ANALYZER_SLOWEST: "List[Tuple[float,int,int,int,bool]]" = []  # top-15 slowest pool analyses

# ── search-policy data (derived from telemetry) ───────────────
TOXIC_SKIP: set[int] = set()
EXCLUDE_EXP_4: set[int] = set()        # primes whose sigma(p^4) has a factor > window
EXP4_FILTER_HITS: "Counter[int]" = Counter()  # per-prime filter verification


# ── prime generation ──────────────────────────────────────────
# ── pool-aware sigma analyser ──────────────────────────────────

def _remove_all(value: mpz, q: int) -> Tuple[mpz, int]:
    """Return (value / q^e, e) where q^e || value."""
    e = 0
    while value % q == 0:
        value //= q
        e += 1
    return value, e


def build_prime_blocks(primes, block_size: int = 256):
    """Partition a compact prime sequence into indexed blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks = []
    for start in range(0, len(primes), block_size):
        stop = min(start + block_size, len(primes))
        product = mpz(1)
        for idx in range(start, stop):
            product *= int(primes[idx])
        blocks.append(PrimeBlock(start=start, stop=stop, product=product))
    return tuple(blocks)


def build_prime_superblocks(blocks: tuple, fanout: int):
    """Group consecutive leaf blocks into superblocks."""
    if fanout < 2:
        raise ValueError("superblock fanout must be at least 2")
    result = []
    for start in range(0, len(blocks), fanout):
        stop = min(start + fanout, len(blocks))
        product = mpz(1)
        for idx in range(start, stop):
            product *= blocks[idx].product
        result.append(PrimeSuperBlock(start=start, stop=stop, product=product))
    return tuple(result)


def build_prime_block_plan(primes, *, block_size: int, superblock_fanout: int,
                           eligible_primes=None, build_superblocks: bool = True):
    """Build a block plan over full or exponent-filtered primes."""
    pool = eligible_primes if eligible_primes is not None else primes
    blocks = build_prime_blocks(pool, block_size)
    superblocks = (
        build_prime_superblocks(blocks, superblock_fanout)
        if build_superblocks else ()
    )
    return PrimeBlockPlan(primes=pool, blocks=blocks, superblocks=superblocks)


def _strip_prime_block(residual: mpz, inside: dict, plan, block, stats) -> mpz:
    """Remove all factors represented by one positive leaf block."""
    for idx in range(block.start, block.stop):
        if residual == 1:
            break
        q = int(plan.primes[idx])
        if residual % q != 0:
            continue
        residual, exponent = _remove_all(residual, q)
        inside[q] = inside.get(q, 0) + exponent
        stats["pool_factors_removed"] += 1
    return residual


def _scan_blocks_flat(residual: mpz, inside: dict, plan, stats) -> mpz:
    """Flat correctness-oracle scanner."""
    for block in plan.blocks:
        if residual == 1:
            break
        stats["leaf_blocks_tested"] += 1
        if gmpy2.gcd(residual, block.product) == 1:
            continue
        stats["positive_blocks"] += 1
        residual = _strip_prime_block(residual, inside, plan, block, stats)
    return residual


def _scan_blocks_hierarchical(residual: mpz, inside: dict, plan, stats) -> mpz:
    """Two-level scanner: superblock gcd → leaf-block gcd."""
    blocks = plan.blocks
    for sb in plan.superblocks:
        if residual == 1:
            break
        stats["superblocks_tested"] += 1
        if gmpy2.gcd(residual, sb.product) == 1:
            stats["leaf_blocks_skipped"] += sb.stop - sb.start
            continue
        stats["positive_superblocks"] += 1
        for idx in range(sb.start, sb.stop):
            if residual == 1:
                break
            stats["leaf_blocks_tested"] += 1
            if gmpy2.gcd(residual, blocks[idx].product) == 1:
                continue
            stats["positive_blocks"] += 1
            residual = _strip_prime_block(residual, inside, plan, blocks[idx], stats)
    return residual


class SigmaPoolAnalyzer:
    """Analyse σ(p^a) against the configured odd-prime pool.

    Tries stripping all in-pool primes; if residual > 1 the branch
    is certified to contain an out-of-pool factor without ever
    running full Pollard–Rho factorisation.
    """

    def __init__(self, primes, *, block_size: int = 256,
                 superblock_fanout: int = 16,
                 gcd_mode: str = "flat",
                 stats=None) -> None:
        if not primes:
            raise ValueError("prime pool must not be empty")
        if int(primes[0]) != 3:
            raise ValueError("complete odd-prime pool must start at 3")
        previous = 1
        for raw_q in primes:
            q = int(raw_q)
            if q < 3 or q % 2 == 0:
                raise ValueError("prime pool must contain only odd integers >= 3")
            if q <= previous:
                raise ValueError("prime pool must be strictly increasing")
            previous = q
        if gcd_mode not in {"flat", "hierarchical"}:
            raise ValueError("gcd_mode must be 'flat' or 'hierarchical'")
        if superblock_fanout < 2:
            raise ValueError("superblock_fanout must be at least 2")

        self.primes = primes
        self.prime_limit = int(primes[-1])
        self.block_size = block_size
        self.superblock_fanout = superblock_fanout
        self.gcd_mode = gcd_mode
        self._scan = _scan_blocks_hierarchical if gcd_mode == "hierarchical" else _scan_blocks_flat
        self._use_superblocks = (gcd_mode == "hierarchical")

        # Single shared full-pool plan for even n (no filter benefit)
        self._full_plan: PrimeBlockPlan | None = None
        # Per-(odd n) filtered plans
        self._plans_by_n: Dict[int, PrimeBlockPlan] = {}

        self._cache: Dict[Tuple[int, int], SigmaPoolAnalysis] = {}
        self.stats: "Counter[str]" = stats if stats is not None else Counter()
        self.slowest: List[Tuple[float, int, int, int, bool]] = []

    def plan_for_exp(self, exp: int) -> PrimeBlockPlan:
        """Return the block plan for *exp*, filtered by necessary-order condition."""
        n = exp + 1
        if n % 2 == 0:
            if self._full_plan is None:
                self._full_plan = build_prime_block_plan(
                    self.primes,
                    block_size=self.block_size,
                    superblock_fanout=self.superblock_fanout,
                    eligible_primes=None,
                    build_superblocks=self._use_superblocks,
                )
            return self._full_plan

        cached = self._plans_by_n.get(n)
        if cached is not None:
            return cached
        # Use same storage type as the master pool (defaults to uint32).
        type_code = getattr(self.primes, "typecode", "I")
        eligible = array(type_code)
        append = eligible.append
        for raw_q in self.primes:
            q = int(raw_q)
            if n % q == 0 or math.gcd(q - 1, n) > 1:
                append(q)
        plan = build_prime_block_plan(
            self.primes,
            block_size=self.block_size,
            superblock_fanout=self.superblock_fanout,
            eligible_primes=eligible,
            build_superblocks=self._use_superblocks,
        )
        self._plans_by_n[n] = plan
        return plan

    def analyze(self, p: int, exp: int) -> SigmaPoolAnalysis:
        key = (p, exp)
        cached = self._cache.get(key)
        if cached is not None:
            self.stats["hits"] += 1
            return cached

        self.stats["misses"] += 1
        started = time.perf_counter()

        # Fast path: a globally exact factorisation already exists
        exact_cached = _SIG_VALUATIONS.get(key)
        if exact_cached is not None:
            outside = next((q for q in exact_cached if q > self.prime_limit), None)
            if outside is None:
                result = SigmaPoolAnalysis(exact=True, valuations=exact_cached, residual=mpz(1))
                self.stats["exact_from_global_cache"] += 1
            else:
                result = SigmaPoolAnalysis(
                    exact=False,
                    valuations={q: e for q, e in exact_cached.items() if q <= self.prime_limit},
                    residual=mpz(outside),
                    outside_witness=outside,
                )
                self.stats["outside_from_global_cache"] += 1
            self._cache[key] = result
            return result

        residual = mpz(sigma_prime_power(p, exp))
        residual, _v2 = _remove_all(residual, 2)
        inside: Dict[int, int] = {}

        plan = self.plan_for_exp(exp)
        self.stats["candidate_leaf_blocks"] += len(plan.blocks)
        residual = self._scan(residual, inside, plan, self.stats)
        # backward-compat alias
        self.stats["blocks_tested"] = self.stats["leaf_blocks_tested"]

        if residual == 1:
            result = SigmaPoolAnalysis(exact=True, valuations=inside, residual=mpz(1))
            _SIG_VALUATIONS[key] = inside
            _SIG_FACTORS[key] = set(inside)
            self.stats["exact"] += 1
        else:
            result = SigmaPoolAnalysis(exact=False, valuations=inside, residual=residual)
            self.stats["outside_certificates"] += 1

        elapsed = time.perf_counter() - started
        self.stats["analysis_ns"] += int(elapsed * 1_000_000_000)
        self.slowest.append((elapsed, p, exp, int(residual).bit_length(), result.exact))
        self.slowest.sort(reverse=True)
        del self.slowest[15:]
        ANALYZER_SLOWEST[:] = self.slowest

        self._cache[key] = result
        return result

    def is_known_outside(self, p: int, exp: int) -> bool:
        c = self._cache.get((p, exp))
        return c is not None and not c.exact


def generate_odd_primes(limit: int, *, segment_odds: int = 2_000_000) -> array:
    """Return all odd primes ≤ *limit* as a compact array.

    Uses ``array('I')`` (32-bit) when *limit* ≤ 2³²-1, otherwise
    ``array('Q')`` (64-bit).  The sieve is segmented: working memory
    is O(segment_odds) instead of O(limit).
    """
    if limit < 3:
        return array("I")
    if segment_odds <= 0:
        raise ValueError("segment_odds must be positive")

    use_32bit = limit <= 0xFFFFFFFF
    array_code = "I" if use_32bit else "Q"
    np_uint = np.uint32 if use_32bit else np.uint64

    root = math.isqrt(limit)
    base_sieve = np.ones(root + 1, dtype=np.bool_)
    base_sieve[:2] = False
    for p in range(2, math.isqrt(root) + 1):
        if base_sieve[p]:
            base_sieve[p * p: root + 1: p] = False
    base_primes = np.flatnonzero(base_sieve)
    odd_base = base_primes[base_primes >= 3]

    result = array(array_code)
    segment_span = 2 * segment_odds
    for low in range(3, limit + 1, segment_span):
        high = min(limit, low + segment_span - 2)
        if high % 2 == 0:
            high -= 1
        count = ((high - low) // 2) + 1
        segment = np.ones(count, dtype=np.bool_)
        for p_val in odd_base:
            p = int(p_val)
            p_sq = p * p
            if p_sq > high:
                break
            start = max(p_sq, ((low + p - 1) // p) * p)
            if start % 2 == 0:
                start += p
            first = (start - low) // 2
            segment[first::p] = False
        indices = np.flatnonzero(segment)
        values = (low + 2 * indices).astype(np_uint, copy=False)
        result.frombytes(values.tobytes())
    return result


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


def _valuation(n: int, q: int) -> int:
    """Return v_q(n) for positive *n* and prime *q*."""
    value = int(n)
    exponent = 0
    while value % q == 0:
        value //= q
        exponent += 1
    return exponent


def _power_minus_one_valuation(p: int, d: int, q: int) -> int:
    """Return v_q(p^d-1) using modular powers instead of constructing p^d."""
    if pow(p, d, q) != 1:
        return 0
    exponent = 1
    modulus = q * q
    while pow(p, d, modulus) == 1:
        exponent += 1
        modulus *= q
    return exponent


def residue_class_count(q: int, e: int, n: int) -> int:
    """Count units x mod q^e with q^e dividing 1+x+...+x^(n-1).

    This exact count describes one prospective source component.  A zero
    count for exponent ``n-1`` must not by itself prune a valuation debt:
    several future source components may split that debt.
    """
    if q < 3 or q % 2 == 0 or not gmpy2.is_prime(q):
        raise ValueError("q must be an odd prime")
    if e < 1 or n < 1:
        raise ValueError("e and n must be positive")
    t = _valuation(n, q)
    g = math.gcd(n, q - 1)
    nonsingular = (g - 1) * q ** min(t, e - 1)
    singular = q ** (e - 1) if t >= e else 0
    return nonsingular + singular


@lru_cache(maxsize=None)
def sigma_valuation_from_order(p: int, a: int, q: int) -> int:
    """Compute ``v_q(sigma(p^a))`` without factoring ``sigma(p^a)``.

    For distinct odd primes p and q, put n=a+1.  If d=ord_q(p), the
    valuation is zero when d does not divide n, is v_q(n) when d=1,
    and otherwise is v_q(p^d-1)+v_q(n/d).  Only divisors of n need to
    be tested, so the calculation stays small even when q is large.
    """
    if a < 0:
        raise ValueError("exponent must be non-negative")
    if q < 3 or q % 2 == 0 or not gmpy2.is_prime(q):
        raise ValueError("q must be an odd prime")
    if p == q or p % q == 0:
        return 0

    n = a + 1
    if p % q == 1:
        return _valuation(n, q)

    order = None
    for d in range(2, n + 1):
        if n % d == 0 and pow(p, d, q) == 1:
            order = d
            break
    if order is None:
        return 0
    return (
        _power_minus_one_valuation(p, order, q)
        + _valuation(n // order, q)
    )


# ── totient & maximum-prime capacity bound ────────────────────

def totient(n: int) -> int:
    """Euler's totient φ(n), cached.  Uses ``factorize()`` which is cached."""
    if n in _TOTIENT_CACHE:
        return _TOTIENT_CACHE[n]
    if n <= 1:
        _TOTIENT_CACHE[n] = n
        return n
    result = n
    for p, _ in factorize(n):
        result -= result // p
    _TOTIENT_CACHE[n] = result
    return result


def _divisors_gt_one(factors: List[Tuple[int, int]]) -> List[int]:
    """All divisors > 1 from a prime factorisation (result of ``factorize``)."""
    divisors = [1]
    for p, e in factors:
        pe_powers = [1]
        for _ in range(e):
            pe_powers.append(pe_powers[-1] * p)
        new_divs = []
        for d in divisors:
            for pe in pe_powers[1:]:
                new_divs.append(d * pe)
        divisors.extend(new_divs)
    divisors.remove(1)
    divisors.sort()
    return divisors


def max_prime_capacity(p: int) -> int:
    r"""B(u) = ½ Σ_{d|u, d>1} φ(d)²  where  u = oddpart(p-1).

    Upper bound on the exponent of *p* when *p* is the largest prime
    factor of an odd perfect number (proved in Lean as
    ``all_odd_order_layers_cyclotomic_exponent_sum_le_budget``).
    """
    if p in _CAPACITY_CACHE:
        return _CAPACITY_CACHE[p]
    if p < 3:
        _CAPACITY_CACHE[p] = 0
        return 0
    u = p - 1
    while u % 2 == 0:
        u //= 2
    if u == 1:
        _CAPACITY_CACHE[p] = 0
        return 0
    factors = factorize(u)
    total = 0
    for d in _divisors_gt_one(factors):
        phi = totient(d)
        total += phi * phi
    result = total // 2
    _CAPACITY_CACHE[p] = result
    return result


def euler_max_exp_capacity(capacity: int) -> int:
    """Largest integer ≡ 1 (mod 4) not exceeding *capacity*.

    0 when *capacity* < 1.  Matches ``euler_rounding`` in Lean."""
    if capacity < 1:
        return 0
    return 1 + 4 * ((capacity - 1) // 4)


def even_max_exp_capacity(capacity: int) -> int:
    """Largest even integer not exceeding *capacity*.

    0 when *capacity* < 2.  Matches ``nonEuler_rounding`` in Lean."""
    if capacity < 2:
        return 0
    return 2 * (capacity // 2)


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
def sigma_valuation_map(p: int, a: int) -> Dict[int, int]:
    """Return the odd-prime valuation map of sigma(p^a), cached exactly."""
    key = (p, a)
    cached = _SIG_VALUATIONS.get(key)
    if cached is not None:
        SIGMA_MAP_STATS["hits"] += 1
        return cached

    SIGMA_MAP_STATS["misses"] += 1
    t0 = time.perf_counter()
    sig = int(sigma_prime_power(p, a))
    valuations = {q: e for q, e in factorize(sig) if q != 2}
    elapsed = time.perf_counter() - t0
    SIGMA_MAP_STATS["factor_seconds"] += elapsed
    max_q = max(valuations) if valuations else 0
    SIGMA_MISS_TIMES.append((p, a, sig.bit_length(), elapsed, max_q))
    _SIG_VALUATIONS[key] = valuations
    _SIG_FACTORS[key] = set(valuations)
    return valuations


def precompute_sig_factors(primes: List[int], max_exp: int) -> None:
    """Populate ``_SIG_FACTORS`` and ``_SIG_VALUATIONS`` for all (p, a).

    _SIG_VALUATIONS stores the full {q: v_q(σ(p^a))} mapping, enabling
    pre-clone valuation contradiction checks that avoid wasted clones.
    """
    _SIG_FACTORS.clear()
    _SIG_VALUATIONS.clear()
    for p in primes:
        for a in range(2, max_exp + 1, 2):
            sigma_valuation_map(p, a)
        if p % 4 == 1:
            for a in valid_euler_exponents(1, max_exp):
                sigma_valuation_map(p, a)


# ── factor-slot-aware ratio bounds ─────────────────────────────

def _top_component_ratio(
    primes: List[int],
    start_idx: int,
    slots: int,
    assigned,
    excluded,
    reserved,
) -> Tuple[mpz, mpz]:
    """Return the largest relaxed ratio from ``slots`` optional components.

    Every finite prime-power component satisfies

        sigma(p^a) / p^a < p / (p - 1).

    The right-hand side is strictly decreasing in p, so the maximum over at
    most ``slots`` distinct available primes is obtained from the smallest
    available primes. Only those primes are multiplied; no full suffix
    products are materialized.
    """
    num = mpz(1)
    den = mpz(1)
    if slots <= 0:
        return num, den

    selected = 0
    for idx in range(max(start_idx, 0), len(primes)):
        p = primes[idx]
        if p in assigned or p in excluded or p in reserved:
            continue
        num *= p
        den *= p - 1
        selected += 1
        if selected == slots:
            break
    return num, den

def ratio_upper_bound(
    ratio_num: mpz, ratio_den: mpz,
    assigned: Dict[int, int], excluded: set[int],
    primes: List[int],
    *,
    next_idx: int,
    remaining_slots: int,
    pending=(),
) -> Tuple[mpz, mpz]:
    """Return a rigorous completion upper bound for sigma(N)/N.

    ``remaining_slots`` is the maximum number of new distinct prime factors.
    Live pending primes are mandatory, may lie before ``next_idx``, and consume
    slots before the smallest optional primes are selected.

    The caller must reject a state first if a live pending prime is excluded,
    outside the finite prime window, or if there are more pending primes than
    remaining slots.
    """
    if remaining_slots < 0:
        raise ValueError("remaining_slots must be non-negative")

    mandatory = {
        int(p) for p in pending
        if p not in assigned
    }
    if mandatory & excluded:
        raise ValueError("a pending prime is excluded")
    if len(mandatory) > remaining_slots:
        raise ValueError("pending primes exceed remaining factor slots")

    num = mpz(ratio_num)
    den = mpz(ratio_den)

    for p in mandatory:
        num *= p
        den *= (p - 1)

    optional_num, optional_den = _top_component_ratio(
        primes,
        next_idx,
        remaining_slots - len(mandatory),
        assigned,
        excluded,
        mandatory,
    )
    num *= optional_num
    den *= optional_den
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
    """Populate EXCLUDE_EXP_4 when sigma(p^4) has a factor above the window.

    Every odd sigma factor is mandatory, so one out-of-window factor is
    sufficient to make the branch impossible in the finite search box.

    This is WINDOW-COMPLETE (not globally complete): a factor > MAX_PRIME
    today may be resolvable if MAX_PRIME is increased later (e.g. 197^4
    produces {661, 991, 2311} — all > 293 now, but 661 fits at MAX_PRIME≥661).
    The filter is parameter-sensitive; rerunning with a larger prime pool
    automatically reclassifies borderline primes.

    a=2 is NEVER filtered.
    """
    EXCLUDE_EXP_4.clear()
    if max_exp < 4:
        return
    for p in primes:
        facs = set(sigma_valuation_map(p, 4))
        if any(q > max_prime for q in facs):
            EXCLUDE_EXP_4.add(p)


def exponent_forces_outside_window(p: int, exp: int, prime_limit: int) -> bool:
    """True if σ(p^exp) has any odd prime factor > *prime_limit*.

    In OPN mode every such factor is a mandatory obligation that cannot
    be resolved within the finite prime window.  This is window-complete:
    raising *prime_limit* may reclassify borderline primes.
    """
    vals = sigma_valuation_map(p, exp)
    return any(q > prime_limit for q in vals)


def exp4_forced_outside_window(p: int, max_prime: int) -> bool:
    """Return whether an odd factor of sigma(p^4) exceeds the window."""
    factors = set(sigma_valuation_map(p, 4))
    outside = any(q > max_prime for q in factors)
    if outside:
        EXCLUDE_EXP_4.add(p)
    return outside


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


def next_prime_upper_bound(
    ratio_num,
    ratio_den,
    candidate_idx,
    remaining_slots,
    target_num,
    target_den,
    primes,
    assigned,
    excluded,
):
    """Return the largest candidate allowed by the best remaining tail.

    The tail contains at most ``remaining_slots - 1`` components and uses
    the smallest available primes after the candidate. The calculation is
    exact and returns zero when no finite upper bound exists.
    """
    if candidate_idx >= len(primes) or remaining_slots <= 0:
        return 0
    ub_n, ub_d = _top_component_ratio(
        primes,
        candidate_idx + 1,
        remaining_slots - 1,
        assigned,
        excluded,
        (),
    )

    # R × p/(p−1) × ub_n/ub_d ≥ T
    # → p/(p−1) ≥ T×ub_d / (R×ub_n)
    # → p ≤ T×ub_d×ratio_den / (T×ub_d×ratio_den − target_den×ratio_num×ub_n)
    num = mpz(target_num) * ub_d * ratio_den
    den = (mpz(target_num) * ratio_den * ub_d
           - mpz(target_den) * ratio_num * ub_n)
    if den <= 0:
        return 0   # no finite upper bound
    return int(num // den)


# Used by the rigorous reverse-valuation debt bound in opn_state.  The
# previous "assigned + excluded congruent primes" check was not implied by
# Nielsen's Lemmas 3.6-3.7 and has deliberately been removed.
FERMAT_PRIMES = {3, 5, 17, 257, 65537}


def is_prime_infinite(p, a):
    """Match the four factorisation cutoffs in Nielsen's Mathematica code."""
    pa = pow(p, a)
    if p < 30:
        return pa > 10**200
    if p < 104:
        return pa > 10**150
    if p < 10000:
        return pa > 10**50
    return pa > 10**30
