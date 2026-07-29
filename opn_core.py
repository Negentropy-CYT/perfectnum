"""
opn_core — arithmetic engine for odd-perfect-number search.

Provides prime generation, Brent-style Pollard-Rho factorisation,
σ(p^a) computation, ratio upper/lower bounds, cached sigma / power /
factorisation results, σ-factor-set precomputation, and all
user-configurable constants.
"""

import math
import random
import hashlib
import sqlite3
import struct
import time
from array import array
import gmpy2
import numpy as np
from gmpy2 import mpz

from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from opn_metrics import PoolPerformance
from opn_plan_cache import (
    PersistentPlanCache,
    PlanCacheBuild,
    PlanCacheBusyError,
    PlanCacheError,
    PlanCacheKey,
    PlanCacheValidationError,
)
from opn_sigma_db import PersistedSigmaRecord, SigmaAnalysisDatabase
if TYPE_CHECKING:
    from opn_metrics import PerformanceMetrics, StructureMetrics

# ── configuration ─────────────────────────────────────────────
CHECKPOINT_FILE  = "checkpoint_merged.pkl"
SOLUTIONS_FILE   = "solutions_merged.txt"
TELEMETRY_FILE   = "telemetry.txt"

MAX_PRIME         = 1000000000     # largest odd prime considered
MAX_FACTORS       = 60         # max distinct prime factors in N
MAX_EXP           = 35         # max exponent (2 = a_i=1 restriction)
PROPAGATE         = True     # False = Descartes-spoof DFS; True = true OPN chain
PROGRESS_INTERVAL = 1_000
CHECKPOINT_INTERVAL_SECONDS = 300.0  # periodic save at a stable search boundary
ENABLE_FERMAT_DEBT = False
POOL_GCD_MODE = "hierarchical"          # "flat" or "hierarchical"
POOL_SUPERBLOCK_FANOUT = 16

# Persistent sigma analysis and plan-build policy.
SIGMA_DATABASE_ENABLED = True
SIGMA_DATABASE_FILE = "sigma_pool.sqlite3"
POOL_PLAN_BUILD_POLICY = "adaptive"  # "eager", "after_db_miss", "adaptive"
POOL_PLAN_DISK_CACHE_ENABLED = True
POOL_PLAN_DISK_CACHE_DIR = "plan_cache"
POOL_PLAN_DISK_MIN_FREE_BYTES = 2 * 1024**3
POOL_ADAPTIVE_BUILD_THRESHOLD = 3
# Incremental plans are worthwhile only when the persisted prefix represents
# a substantial part of the current pool.  Otherwise one reusable full plan
# avoids building almost-full interval plans before new database misses.
POOL_INCREMENTAL_MIN_PREFIX_FRACTION = 0.5

# Number of master-pool primes processed by one NumPy chunk.
# At 2,000,000:
#   uint32 temporary working memory is roughly 30–40 MiB;
#   uint64 temporary working memory is roughly 45–60 MiB.
POOL_PLAN_CHUNK_PRIMES = 4_000_000

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

    exact=False: *residual* is the complete odd cofactor left after every
    pool prime (with multiplicity) has been removed, so it is > 1 and has
    no prime factor in the pool.  *valuations* is only the pool-internal
    partial map and must not be used for factor-chain propagation.

    *outside_witness* is the smallest outside-pool prime when an exact global
    factorisation supplied that information; a cold scan may leave it unset.
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
    """A logical range of leaf blocks with one persistent product."""
    start_leaf: int     # inclusive logical leaf-block index
    stop_leaf: int      # exclusive logical leaf-block index
    product: mpz        # product of every prime in the logical range


@dataclass(slots=True, frozen=True)
class PrimeBlockPlan:
    """Prime pool with persistent superblocks and optional flat blocks."""
    primes: Sequence[int]
    block_size: int
    superblock_fanout: int
    leaf_block_count: int
    # Flat-mode correctness oracle only.  Hierarchical plans keep this empty.
    blocks: Tuple[PrimeBlock, ...] = ()
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
_TOTIENT_CACHE: Dict[int, int] = {}
_CAPACITY_CACHE: Dict[int, int] = {}

# ── search-policy data (derived from telemetry) ───────────────
TOXIC_SKIP: set[int] = set()
EXCLUDE_EXP_4: set[int] = set()        # primes whose sigma(p^4) has a factor > window
EXP4_FILTER_HITS: "Counter[int]" = Counter()  # per-prime filter verification

# ── optional performance-metrics bridge (set by search engine) ─
_sigma_map_perf: "PerformanceMetrics | None" = None
"""Set by the search engine before sigma_valuation_map is called in hot paths."""


# ── prime generation ──────────────────────────────────────────
# ── pool-aware sigma analyser ──────────────────────────────────

def _remove_all(value: mpz, q: int) -> Tuple[mpz, int]:
    """Return (value / q^e, e) where q^e || value."""
    e = 0
    while value % q == 0:
        value //= q
        e += 1
    return value, e


@lru_cache(maxsize=None)
def distinct_prime_factors(value: int) -> Tuple[int, ...]:
    """Return the distinct prime factors of a positive integer."""
    if value < 1:
        raise ValueError("value must be positive")

    factors: list[int] = []
    remainder = value
    divisor = 2

    while divisor * divisor <= remainder:
        if remainder % divisor == 0:
            factors.append(divisor)

            while remainder % divisor == 0:
                remainder //= divisor

        divisor = 3 if divisor == 2 else divisor + 2

    if remainder > 1:
        factors.append(remainder)

    return tuple(factors)


@lru_cache(maxsize=None)
def squarefree_kernel(value: int) -> int:
    """Return rad(value), the product of its distinct prime factors."""
    result = 1

    for factor in distinct_prime_factors(value):
        result *= factor

    return result


def _numpy_prime_view(
    primes: Sequence[int],
) -> Tuple[np.ndarray, np.dtype]:
    """Return a zero-copy NumPy view when the pool is array('I'/'Q').

    A normal Python sequence is converted to one compact NumPy array.
    """
    if isinstance(primes, array):
        if primes.typecode == "I" and primes.itemsize == 4:
            dtype = np.dtype("=u4")
        elif primes.typecode == "Q" and primes.itemsize == 8:
            dtype = np.dtype("=u8")
        else:
            raise TypeError(
                "compact prime pool must use array('I') or array('Q')"
            )

        return np.frombuffer(primes, dtype=dtype), dtype

    if len(primes) == 0:
        raise ValueError("prime pool must not be empty")

    dtype = np.dtype(
        "=u4"
        if int(primes[-1]) <= 0xFFFFFFFF
        else "=u8"
    )

    return np.asarray(primes, dtype=dtype), dtype


def _typed_searchsorted_right(
    sorted_values: np.ndarray,
    value: int,
) -> int:
    """Return a right insertion point without whole-array dtype promotion.

    On very large unsigned arrays, passing a Python ``int`` directly to
    NumPy's ``searchsorted`` can make dtype resolution traverse the complete
    input.  Converting the scalar to the array's exact dtype preserves the
    intended O(log n) binary search.
    """
    if sorted_values.ndim != 1:
        raise ValueError("searchsorted input must be one-dimensional")
    if not np.issubdtype(sorted_values.dtype, np.integer):
        raise TypeError("searchsorted input must have an integer dtype")

    raw_value = int(value)
    bounds = np.iinfo(sorted_values.dtype)
    if raw_value < int(bounds.min):
        return 0
    if raw_value > int(bounds.max):
        return len(sorted_values)

    typed_value = sorted_values.dtype.type(raw_value)
    return int(
        np.searchsorted(
            sorted_values,
            typed_value,
            side="right",
        )
    )


def _validate_prime_pool_scalar(
    primes: Sequence[int],
) -> None:
    """Validate an external Python sequence without unsigned coercion."""
    if len(primes) == 0:
        raise ValueError("prime pool must not be empty")
    if int(primes[0]) != 3:
        raise ValueError("complete odd-prime pool must start at 3")

    previous = 1
    for raw_q in primes:
        q = int(raw_q)
        if q < 3 or q % 2 == 0:
            raise ValueError(
                "prime pool must contain only odd integers >= 3"
            )
        if q <= previous:
            raise ValueError("prime pool must be strictly increasing")
        previous = q


def validate_prime_pool_vectorized(
    primes: Sequence[int],
    *,
    chunk_size: int = 4_000_000,
) -> None:
    """Validate odd-pool structure, vectorizing compact integer storage.

    ``array('I'/'Q')`` and one-dimensional NumPy integer arrays are checked
    in bounded chunks.  Other Python sequences use the scalar path so invalid
    negative or oversized values cannot be hidden by unsigned conversion.
    This validates the same structural contract as the former constructor
    loop; it does not independently prove primality.
    """
    if chunk_size <= 0:
        raise ValueError("prime validation chunk_size must be positive")
    if len(primes) == 0:
        raise ValueError("prime pool must not be empty")

    if isinstance(primes, array):
        compact_unsigned = (
            primes.typecode == "I"
            and primes.itemsize == 4
        ) or (
            primes.typecode == "Q"
            and primes.itemsize == 8
        )
        if not compact_unsigned:
            _validate_prime_pool_scalar(primes)
            return
        view, _dtype = _numpy_prime_view(primes)
    elif isinstance(primes, np.ndarray):
        if primes.ndim != 1:
            raise ValueError("prime pool NumPy array must be one-dimensional")
        if not np.issubdtype(primes.dtype, np.integer):
            raise TypeError("prime pool NumPy array must have an integer dtype")
        view = primes
    else:
        _validate_prime_pool_scalar(primes)
        return

    if int(view[0]) != 3:
        raise ValueError("complete odd-prime pool must start at 3")

    previous_last: Optional[int] = None
    for start in range(0, len(view), chunk_size):
        stop = min(start + chunk_size, len(view))
        chunk = view[start:stop]

        if np.any(chunk < 3) or (
            np.count_nonzero(np.bitwise_and(chunk, 1))
            != len(chunk)
        ):
            raise ValueError(
                "prime pool must contain only odd integers >= 3"
            )
        if (
            previous_last is not None
            and int(chunk[0]) <= previous_last
        ):
            raise ValueError("prime pool must be strictly increasing")
        if (
            len(chunk) > 1
            and np.any(chunk[1:] <= chunk[:-1])
        ):
            raise ValueError("prime pool must be strictly increasing")

        previous_last = int(chunk[-1])


def prime_pool_prefix_digest(
    primes: Sequence[int],
    stop: int | None = None,
    *,
    chunk_size: int = 1_000_000,
) -> bytes:
    """Return a storage-independent SHA-256 digest of a prime-pool prefix.

    Values are hashed as canonical little-endian uint64 integers, so an
    ``array('I')`` prefix and the same values in ``array('Q')`` have the same
    identity.  The bounded conversion chunks avoid a second full-size copy.
    """
    if chunk_size <= 0:
        raise ValueError("prime digest chunk_size must be positive")

    view, _dtype = _numpy_prime_view(primes)
    prefix_stop = len(view) if stop is None else int(stop)
    if prefix_stop <= 0 or prefix_stop > len(view):
        raise ValueError("prime digest prefix is out of range")

    digest = hashlib.sha256()
    digest.update(b"opn-prime-pool-v1\0")
    digest.update(struct.pack(">Q", prefix_stop))

    for start in range(0, prefix_stop, chunk_size):
        chunk_stop = min(start + chunk_size, prefix_stop)
        canonical = np.asarray(
            view[start:chunk_stop],
            dtype=np.dtype("<u8"),
        )
        digest.update(memoryview(canonical).cast("B"))

    return digest.digest()


def _factor_masks_for_chunk(
    prime_chunk: np.ndarray,
    factors: Tuple[int, ...],
) -> Dict[int, np.ndarray]:
    """Build the necessary-order mask for each small factor.

    For one factor ell, the mask represents:

        q == ell  or  q ≡ 1 (mod ell)

    The first case covers q | n; the second covers ell | gcd(q-1, n).
    """
    q_minus_one = np.empty_like(prime_chunk)
    np.subtract(prime_chunk, 1, out=q_minus_one)

    remainders = np.empty_like(prime_chunk)
    temporary = np.empty(prime_chunk.shape, dtype=np.bool_)

    masks: Dict[int, np.ndarray] = {}

    for factor in factors:
        mask = np.empty(prime_chunk.shape, dtype=np.bool_)

        # q divides n. Since q is prime, q must equal one of rad(n)'s factors.
        np.equal(prime_chunk, factor, out=mask)

        # factor divides q - 1.
        np.remainder(
            q_minus_one,
            factor,
            out=remainders,
        )
        np.equal(
            remainders,
            0,
            out=temporary,
        )
        np.logical_or(
            mask,
            temporary,
            out=mask,
        )

        masks[factor] = mask

    return masks


def _mask_for_factors(
    factor_masks: Dict[int, np.ndarray],
    factors: Tuple[int, ...],
) -> np.ndarray:
    """Combine factor masks for one radical's prime factors."""
    if len(factors) == 1:
        return factor_masks[factors[0]]

    result = factor_masks[factors[0]].copy()

    for factor in factors[1:]:
        np.logical_or(
            result,
            factor_masks[factor],
            out=result,
        )

    return result


def build_filtered_prime_pools_vectorized(
    primes: Sequence[int],
    radicals: Sequence[int],
    *,
    chunk_primes: int = POOL_PLAN_CHUNK_PRIMES,
    pool_perf: "PoolPerformance | None" = None,
) -> Dict[int, np.ndarray]:
    """Build every requested radical-filtered pool in two chunked passes.

    Pass 1 counts exact output lengths.
    Pass 2 fills preallocated arrays.

    No repeated np.concatenate() is used, so output construction is linear
    and temporary memory depends on chunk_primes rather than MAX_PRIME.
    """
    if chunk_primes <= 0:
        raise ValueError("chunk_primes must be positive")

    radical_keys = tuple(
        sorted(set(int(value) for value in radicals))
    )

    if not radical_keys:
        return {}

    factors_by_radical = {
        radical: distinct_prime_factors(radical)
        for radical in radical_keys
    }

    all_factors = tuple(
        sorted({
            factor
            for factors in factors_by_radical.values()
            for factor in factors
        })
    )

    prime_view, dtype = _numpy_prime_view(primes)
    source_count = len(prime_view)

    counts = {
        radical: 0
        for radical in radical_keys
    }

    # ── pass 1: count exact output lengths ────────────────────
    count_started = time.perf_counter_ns()

    for start in range(0, source_count, chunk_primes):
        stop = min(start + chunk_primes, source_count)
        chunk = prime_view[start:stop]

        factor_masks = _factor_masks_for_chunk(
            chunk,
            all_factors,
        )

        for radical in radical_keys:
            factors = factors_by_radical[radical]
            mask = _mask_for_factors(
                factor_masks,
                factors,
            )

            counts[radical] += int(
                np.count_nonzero(mask)
            )

    count_ns = time.perf_counter_ns() - count_started

    # Allocate final arrays exactly once.
    outputs = {
        radical: np.empty(
            counts[radical],
            dtype=dtype,
        )
        for radical in radical_keys
    }

    positions = {
        radical: 0
        for radical in radical_keys
    }

    # ── pass 2: fill exact-size outputs ───────────────────────
    fill_started = time.perf_counter_ns()

    for start in range(0, source_count, chunk_primes):
        stop = min(start + chunk_primes, source_count)
        chunk = prime_view[start:stop]

        factor_masks = _factor_masks_for_chunk(
            chunk,
            all_factors,
        )

        for radical in radical_keys:
            factors = factors_by_radical[radical]
            mask = _mask_for_factors(
                factor_masks,
                factors,
            )

            selected = chunk[mask]
            selected_count = len(selected)

            destination_start = positions[radical]
            destination_stop = (
                destination_start + selected_count
            )

            outputs[radical][
                destination_start:destination_stop
            ] = selected

            positions[radical] = destination_stop

    fill_ns = time.perf_counter_ns() - fill_started

    for radical in radical_keys:
        if positions[radical] != counts[radical]:
            raise RuntimeError(
                f"filtered-plan fill mismatch for rad={radical}: "
                f"{positions[radical]} != {counts[radical]}"
            )

    if pool_perf is not None:
        pool_perf.plan_filter_count_ns += count_ns
        pool_perf.plan_filter_fill_ns += fill_ns
        pool_perf.plan_filter_ns += count_ns + fill_ns

        pool_perf.plan_filter_source_values += (
            2 * source_count
        )
        pool_perf.filtered_prime_values += sum(
            counts.values()
        )
        pool_perf.filtered_prime_bytes += sum(
            output.nbytes
            for output in outputs.values()
        )

    return outputs


def build_filtered_prime_pools_vectorized_memmap(
    primes: Sequence[int],
    builds: Dict[int, PlanCacheBuild],
    *,
    chunk_primes: int = POOL_PLAN_CHUNK_PRIMES,
    pool_perf: "PoolPerformance | None" = None,
) -> Dict[int, int]:
    """Filter several radical pools directly into cache-owned mmap files.

    This is the same two-pass necessary-order filter as
    :func:`build_filtered_prime_pools_vectorized`, but the exact-sized outputs
    are disk-backed from their creation.  All radicals still share both source
    passes, avoiding one full traversal per exponent class.
    """
    if chunk_primes <= 0:
        raise ValueError("chunk_primes must be positive")
    radical_keys = tuple(sorted(builds))
    if not radical_keys:
        return {}

    factors_by_radical = {
        radical: distinct_prime_factors(radical)
        for radical in radical_keys
    }
    all_factors = tuple(
        sorted({
            factor
            for factors in factors_by_radical.values()
            for factor in factors
        })
    )
    prime_view, _dtype = _numpy_prime_view(primes)
    source_count = len(prime_view)
    counts = {radical: 0 for radical in radical_keys}

    count_started = time.perf_counter_ns()
    for start in range(0, source_count, chunk_primes):
        stop = min(start + chunk_primes, source_count)
        chunk = prime_view[start:stop]
        factor_masks = _factor_masks_for_chunk(
            chunk,
            all_factors,
        )
        for radical in radical_keys:
            mask = _mask_for_factors(
                factor_masks,
                factors_by_radical[radical],
            )
            counts[radical] += int(np.count_nonzero(mask))
    count_ns = time.perf_counter_ns() - count_started

    outputs: Dict[int, np.ndarray] = {}
    try:
        outputs = {
            radical: builds[radical].allocate_primes(
                counts[radical]
            )
            for radical in radical_keys
        }
        positions = {radical: 0 for radical in radical_keys}

        fill_started = time.perf_counter_ns()
        for start in range(0, source_count, chunk_primes):
            stop = min(start + chunk_primes, source_count)
            chunk = prime_view[start:stop]
            factor_masks = _factor_masks_for_chunk(
                chunk,
                all_factors,
            )
            for radical in radical_keys:
                mask = _mask_for_factors(
                    factor_masks,
                    factors_by_radical[radical],
                )
                selected = chunk[mask]
                destination_start = positions[radical]
                destination_stop = (
                    destination_start + len(selected)
                )
                outputs[radical][
                    destination_start:destination_stop
                ] = selected
                positions[radical] = destination_stop
        fill_ns = time.perf_counter_ns() - fill_started

        for radical in radical_keys:
            if positions[radical] != counts[radical]:
                raise RuntimeError(
                    f"disk-plan fill mismatch for rad={radical}: "
                    f"{positions[radical]} != {counts[radical]}"
                )
            mapping = outputs[radical]
            if isinstance(mapping, np.memmap):
                mapping.flush()
                mapping._mmap.close()
    except BaseException:
        for mapping in outputs.values():
            if isinstance(mapping, np.memmap):
                try:
                    mapping._mmap.close()
                except (BufferError, OSError):
                    pass
        raise

    if pool_perf is not None:
        pool_perf.plan_filter_count_ns += count_ns
        pool_perf.plan_filter_fill_ns += fill_ns
        pool_perf.plan_filter_ns += count_ns + fill_ns
        pool_perf.plan_filter_source_values += 2 * source_count
        pool_perf.filtered_prime_values += sum(counts.values())

    return counts


def _product_prime_range(
    primes: Sequence[int],
    start: int,
    stop: int,
) -> mpz:
    """Return the exact product of ``primes[start:stop]``."""
    product = mpz(1)
    for idx in range(start, stop):
        product *= int(primes[idx])
    return product


def build_prime_blocks(primes, block_size: int = 256):
    """Partition a compact prime sequence into indexed blocks."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    blocks = []
    for start in range(0, len(primes), block_size):
        stop = min(start + block_size, len(primes))
        product = _product_prime_range(primes, start, stop)
        blocks.append(PrimeBlock(start=start, stop=stop, product=product))
    return tuple(blocks)


def build_prime_superblocks(blocks: tuple, fanout: int):
    """Group resident flat-mode leaf blocks into superblocks.

    This helper remains available for tests and comparison tooling. Production
    hierarchical plans use :func:`build_compact_superblocks` so the complete
    leaf-product layer is never resident.
    """
    if fanout < 2:
        raise ValueError("superblock fanout must be at least 2")
    result = []
    for start in range(0, len(blocks), fanout):
        stop = min(start + fanout, len(blocks))
        product = mpz(1)
        for idx in range(start, stop):
            product *= blocks[idx].product
        result.append(
            PrimeSuperBlock(
                start_leaf=start,
                stop_leaf=stop,
                product=product,
            )
        )
    return tuple(result)


def build_compact_superblocks(
    primes: Sequence[int],
    *,
    block_size: int,
    superblock_fanout: int,
) -> tuple[Tuple[PrimeSuperBlock, ...], int, int, int]:
    """Build superblocks while retaining at most one fanout of leaf products.

    Returns ``(superblocks, leaf_count, leaf_product_ns, super_product_ns)``.
    Leaf products are exact temporary construction values and become
    unreachable after their superblock has been assembled.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if superblock_fanout < 2:
        raise ValueError("superblock fanout must be at least 2")

    prime_count = len(primes)
    leaf_count = (prime_count + block_size - 1) // block_size
    result = []
    leaf_product_ns = 0
    super_product_ns = 0

    for start_leaf in range(0, leaf_count, superblock_fanout):
        stop_leaf = min(
            start_leaf + superblock_fanout,
            leaf_count,
        )

        leaf_started = time.perf_counter_ns()
        temporary_leaf_products = []
        for leaf_idx in range(start_leaf, stop_leaf):
            prime_start = leaf_idx * block_size
            prime_stop = min(prime_start + block_size, prime_count)
            temporary_leaf_products.append(
                _product_prime_range(
                    primes,
                    prime_start,
                    prime_stop,
                )
            )
        leaf_product_ns += time.perf_counter_ns() - leaf_started

        super_started = time.perf_counter_ns()
        super_product = mpz(1)
        for leaf_product in temporary_leaf_products:
            super_product *= leaf_product
        super_product_ns += time.perf_counter_ns() - super_started

        result.append(
            PrimeSuperBlock(
                start_leaf=start_leaf,
                stop_leaf=stop_leaf,
                product=super_product,
            )
        )
        del leaf_product
        del temporary_leaf_products

    return (
        tuple(result),
        leaf_count,
        leaf_product_ns,
        super_product_ns,
    )


def build_prime_block_plan(
    primes,
    *,
    block_size: int,
    superblock_fanout: int,
    eligible_primes=None,
    build_superblocks: bool = True,
    pool_perf: "PoolPerformance | None" = None,
):
    """Build a block plan over full or exponent-filtered primes."""
    pool = (
        eligible_primes
        if eligible_primes is not None
        else primes
    )

    if build_superblocks:
        (
            superblocks,
            leaf_count,
            leaf_ns,
            superblock_ns,
        ) = build_compact_superblocks(
            pool,
            block_size=block_size,
            superblock_fanout=superblock_fanout,
        )
        blocks = ()
    else:
        leaf_started = time.perf_counter_ns()
        blocks = build_prime_blocks(
            pool,
            block_size,
        )
        leaf_ns = time.perf_counter_ns() - leaf_started
        leaf_count = len(blocks)
        superblocks = ()
        superblock_ns = 0

    if pool_perf is not None:
        pool_perf.leaf_product_ns += leaf_ns
        pool_perf.superblock_product_ns += superblock_ns
        pool_perf.plans_built += 1
        pool_perf.plan_leaf_blocks += leaf_count
        pool_perf.plan_superblocks += len(superblocks)
        pool_perf.logical_leaf_blocks += leaf_count
        pool_perf.resident_leaf_blocks += len(blocks)

    return PrimeBlockPlan(
        primes=pool,
        block_size=block_size,
        superblock_fanout=superblock_fanout,
        leaf_block_count=leaf_count,
        blocks=blocks,
        superblocks=superblocks,
    )


def _strip_prime_range(
    residual: mpz,
    inside: dict,
    plan: PrimeBlockPlan,
    start: int,
    stop: int,
    perf: "PoolPerformance",
) -> mpz:
    """Remove every pool factor represented by one positive prime range."""
    for idx in range(start, stop):
        if residual == 1:
            break
        q = int(plan.primes[idx])
        if residual % q != 0:
            continue
        residual, exponent = _remove_all(residual, q)
        inside[q] = inside.get(q, 0) + exponent
        perf.factors_removed += 1
    return residual


def _strip_prime_block(
    residual: mpz,
    inside: dict,
    plan: PrimeBlockPlan,
    block: PrimeBlock,
    perf: "PoolPerformance",
) -> mpz:
    """Remove all factors represented by one resident flat-mode block."""
    return _strip_prime_range(
        residual,
        inside,
        plan,
        block.start,
        block.stop,
        perf,
    )


def _build_dynamic_leaf_product(
    plan: PrimeBlockPlan,
    leaf_idx: int,
) -> tuple[int, int, mpz]:
    """Rebuild one logical leaf product from the immutable prime sequence."""
    if leaf_idx < 0 or leaf_idx >= plan.leaf_block_count:
        raise IndexError("logical leaf index out of range")
    start = leaf_idx * plan.block_size
    stop = min(start + plan.block_size, len(plan.primes))
    return (
        start,
        stop,
        _product_prime_range(plan.primes, start, stop),
    )


def _scan_blocks_flat(
    residual: mpz,
    inside: dict,
    plan,
    perf: "PoolPerformance",
    *,
    start_leaf: int = 0,
) -> mpz:
    """Flat correctness-oracle scanner."""
    if start_leaf < 0 or start_leaf > plan.leaf_block_count:
        raise ValueError("flat scan start leaf is out of range")
    for block_idx in range(start_leaf, len(plan.blocks)):
        if residual == 1:
            break
        block = plan.blocks[block_idx]
        perf.leaf_blocks_tested += 1
        if gmpy2.gcd(residual, block.product) == 1:
            continue
        perf.positive_blocks += 1
        residual = _strip_prime_block(residual, inside, plan, block, perf)
    return residual


def _scan_blocks_hierarchical(
    residual: mpz,
    inside: dict,
    plan,
    perf: "PoolPerformance",
    *,
    start_leaf: int = 0,
) -> mpz:
    """Two-level scanner: superblock gcd → leaf-block gcd.

    ``start_leaf`` may skip a prefix already certified coprime to ``residual``.
    The boundary is rounded down to its containing superblock, so no possible
    factor after the boundary can be missed.
    """
    if start_leaf < 0 or start_leaf > plan.leaf_block_count:
        raise ValueError("hierarchical scan start leaf is out of range")
    if start_leaf == plan.leaf_block_count:
        return residual

    start_superblock = start_leaf // plan.superblock_fanout
    for superblock_idx in range(start_superblock, len(plan.superblocks)):
        if residual == 1:
            break
        sb = plan.superblocks[superblock_idx]
        perf.superblocks_tested += 1
        super_hit = gmpy2.gcd(residual, sb.product)
        if super_hit == 1:
            perf.leaf_blocks_skipped += (
                sb.stop_leaf - sb.start_leaf
            )
            continue
        perf.positive_superblocks += 1
        found_positive_leaf = False
        for leaf_idx in range(sb.start_leaf, sb.stop_leaf):
            if residual == 1:
                break
            if super_hit == 1:
                break

            rebuild_started = time.perf_counter_ns()
            start, stop, leaf_product = _build_dynamic_leaf_product(
                plan,
                leaf_idx,
            )
            perf.dynamic_leaf_product_ns += (
                time.perf_counter_ns() - rebuild_started
            )
            perf.dynamic_leaf_products_built += 1
            perf.dynamic_leaf_prime_values += stop - start

            perf.leaf_blocks_tested += 1
            leaf_hit = gmpy2.gcd(super_hit, leaf_product)
            if leaf_hit == 1:
                continue
            found_positive_leaf = True
            perf.positive_blocks += 1
            residual = _strip_prime_range(
                residual,
                inside,
                plan,
                start,
                stop,
                perf,
            )
            # Leaf prime sets are disjoint and the superblock product is
            # squarefree, so these hit primes cannot occur in a later leaf.
            super_hit //= leaf_hit
        if __debug__ and not found_positive_leaf:
            raise AssertionError(
                "positive superblock produced no positive leaf"
            )
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
                 plan_chunk_primes: int = POOL_PLAN_CHUNK_PRIMES,
                 pool_perf: "PoolPerformance | None" = None,
                 structure: "StructureMetrics | None" = None,
                 database_path: str | None = None,
                 plan_cache_dir: str | None = None,
                 plan_cache_minimum_free_bytes: int =
                 POOL_PLAN_DISK_MIN_FREE_BYTES,
                 plan_build_policy: str = "eager",
                 adaptive_build_threshold: int =
                 POOL_ADAPTIVE_BUILD_THRESHOLD) -> None:
        validate_prime_pool_vectorized(primes)
        if gcd_mode not in {"flat", "hierarchical"}:
            raise ValueError("gcd_mode must be 'flat' or 'hierarchical'")
        if superblock_fanout < 2:
            raise ValueError("superblock_fanout must be at least 2")
        if plan_build_policy not in {
            "eager",
            "after_db_miss",
            "adaptive",
        }:
            raise ValueError("invalid pool plan build policy")
        if adaptive_build_threshold < 1:
            raise ValueError("adaptive build threshold must be positive")
        if plan_cache_dir is not None and gcd_mode != "hierarchical":
            raise ValueError(
                "persistent plan cache requires hierarchical GCD mode"
            )

        self.primes = primes
        self._prime_view, _dtype = _numpy_prime_view(primes)
        self.prime_limit = int(primes[-1])
        self.block_size = block_size
        self.superblock_fanout = superblock_fanout
        self.gcd_mode = gcd_mode
        self.plan_chunk_primes = plan_chunk_primes
        self.plan_build_policy = plan_build_policy
        self.adaptive_build_threshold = adaptive_build_threshold
        self._scan = _scan_blocks_hierarchical if gcd_mode == "hierarchical" else _scan_blocks_flat
        self._use_superblocks = (gcd_mode == "hierarchical")

        # Single shared full-pool plan for even n (no filter benefit)
        self._full_plan: PrimeBlockPlan | None = None
        # Plans are keyed by rad(exp + 1), not exp + 1 itself.
        self._plans_by_radical: Dict[int, PrimeBlockPlan] = {}
        # Incremental plans are keyed by (prefix stop, 0-or-radical).
        self._interval_plans: Dict[Tuple[int, int], PrimeBlockPlan] = {}
        self._required_exponents: Tuple[int, ...] = ()
        self._normal_plan_miss_keys: set[int] = set()

        self._cache: Dict[Tuple[int, int], SigmaPoolAnalysis] = {}
        self._perf: "PoolPerformance" = pool_perf if pool_perf is not None else PoolPerformance()
        self._structure: "StructureMetrics | None" = structure
        self._pool_digest_cache: Dict[int, bytes] = {}
        self._prime_prefix_stop_cache: Dict[int, int] = {}
        self._normal_plan_scan_leaf_cache: Dict[Tuple[int, int], int] = {}
        self._database: SigmaAnalysisDatabase | None = None
        self.database_error: str | None = None
        if database_path is not None:
            try:
                self._database = SigmaAnalysisDatabase(database_path)
            except (OSError, sqlite3.Error, ValueError) as exc:
                self.database_error = str(exc)
        self._plan_cache: PersistentPlanCache | None = None
        self.plan_cache_error: str | None = None
        if plan_cache_dir is not None:
            try:
                self._plan_cache = PersistentPlanCache(
                    plan_cache_dir,
                    minimum_free_bytes=(
                        plan_cache_minimum_free_bytes
                    ),
                )
            except (OSError, ValueError) as exc:
                self.plan_cache_error = str(exc)

    def configure_plan_build(
        self,
        exponents: Sequence[int],
    ) -> None:
        """Record the exponent domain and eagerly build only by policy."""
        self._required_exponents = tuple(
            sorted(set(int(exp) for exp in exponents))
        )
        if self.plan_build_policy == "eager":
            self.prebuild_plans(self._required_exponents)

    @staticmethod
    def _normal_plan_key(exp: int) -> int:
        n = exp + 1
        return 0 if n % 2 == 0 else squarefree_kernel(n)

    def _cached_normal_plan(
        self,
        exp: int,
    ) -> PrimeBlockPlan | None:
        key = self._normal_plan_key(exp)
        if key == 0:
            return self._full_plan
        return self._plans_by_radical.get(key)

    def _evict_interval_plans_for_key(self, key: int) -> None:
        stale = [
            interval_key
            for interval_key in self._interval_plans
            if interval_key[1] == key
        ]
        for interval_key in stale:
            del self._interval_plans[interval_key]

    def _maybe_adaptive_prebuild(self, exp: int) -> None:
        if (
            self.plan_build_policy != "adaptive"
            or not self._required_exponents
        ):
            return
        self._normal_plan_miss_keys.add(
            self._normal_plan_key(exp)
        )
        if (
            len(self._normal_plan_miss_keys)
            >= self.adaptive_build_threshold
        ):
            # Full-window plans supersede incremental plans.  Release the
            # latter before bulk construction so both layers never define
            # the steady-state memory footprint.
            self._interval_plans.clear()
            self.prebuild_plans(self._required_exponents)

    def _pool_digest_for_limit(
        self,
        limit: int,
    ) -> bytes | None:
        """Return the certified current-pool prefix ending at *limit*."""
        limit = int(limit)
        cached = self._pool_digest_cache.get(limit)
        if cached is not None:
            return cached
        if limit > self.prime_limit:
            return None

        stop = self._prime_prefix_stop(limit)
        if (
            stop == 0
            or int(self._prime_view[stop - 1]) != limit
        ):
            return None

        digest = prime_pool_prefix_digest(
            self._prime_view,
            stop,
        )
        self._pool_digest_cache[limit] = digest
        return digest

    def _disk_plan_key(
        self,
        radical: int,
        *,
        source_start: int,
    ) -> PlanCacheKey:
        if source_start < 0 or source_start >= len(self._prime_view):
            raise ValueError("disk plan source start is out of range")
        digest = self._pool_digest_for_limit(self.prime_limit)
        if digest is None:
            raise ValueError(
                "current prime pool has no certifiable upper bound"
            )
        return PlanCacheKey(
            pool_digest=digest.hex(),
            prime_limit=self.prime_limit,
            source_start=int(source_start),
            source_count=len(self._prime_view) - source_start,
            radical=int(radical),
            dtype=self._prime_view.dtype.str,
            block_size=self.block_size,
            superblock_fanout=self.superblock_fanout,
        )

    def _source_primes(self, source_start: int):
        if source_start == 0:
            return self.primes
        return self._prime_view[source_start:]

    def _plan_from_disk_payload(
        self,
        key: PlanCacheKey,
        loaded,
    ) -> PrimeBlockPlan:
        source = self._source_primes(key.source_start)
        if key.radical == 0:
            if loaded.prime_count != len(source):
                raise PlanCacheValidationError(
                    "unfiltered disk plan prime count mismatch"
                )
            pool = source
        else:
            if loaded.eligible_primes is None:
                raise PlanCacheValidationError(
                    "filtered disk plan lacks its mmap array"
                )
            pool = loaded.eligible_primes

        superblocks = tuple(
            PrimeSuperBlock(
                start_leaf=index * self.superblock_fanout,
                stop_leaf=min(
                    (index + 1) * self.superblock_fanout,
                    loaded.leaf_count,
                ),
                product=product,
            )
            for index, product in enumerate(
                loaded.superblock_products
            )
        )
        self._perf.plan_leaf_blocks += loaded.leaf_count
        self._perf.plan_superblocks += len(superblocks)
        self._perf.logical_leaf_blocks += loaded.leaf_count
        return PrimeBlockPlan(
            primes=pool,
            block_size=self.block_size,
            superblock_fanout=self.superblock_fanout,
            leaf_block_count=loaded.leaf_count,
            blocks=(),
            superblocks=superblocks,
        )

    def _load_disk_plan(
        self,
        radical: int,
        *,
        source_start: int,
    ) -> tuple[PrimeBlockPlan | None, PlanCacheKey | None]:
        cache = self._plan_cache
        if cache is None:
            return None, None
        key = self._disk_plan_key(
            radical,
            source_start=source_start,
        )
        try:
            loaded = cache.load(key)
            if loaded is None:
                self._perf.disk_plan_misses += 1
                return None, key
            plan = self._plan_from_disk_payload(key, loaded)
        except PlanCacheValidationError:
            self._perf.disk_plan_invalid += 1
            return None, key
        except (OSError, PlanCacheError, ValueError) as exc:
            self.plan_cache_error = str(exc)
            self._plan_cache = None
            return None, None
        self._perf.disk_plan_hits += 1
        return plan, key

    def _commit_disk_plan(
        self,
        build: PlanCacheBuild,
        plan: PrimeBlockPlan,
        *,
        prime_count: int | None = None,
        first_prime: int | None = None,
        last_prime: int | None = None,
    ):
        if prime_count is None:
            prime_count = len(plan.primes)
        if first_prime is None or last_prime is None:
            if prime_count:
                first_prime = int(plan.primes[0])
                last_prime = int(plan.primes[-1])
            else:
                first_prime = 0
                last_prime = 0
        mapped = build.commit(
            prime_count=prime_count,
            leaf_count=plan.leaf_block_count,
            superblock_products=tuple(
                block.product
                for block in plan.superblocks
            ),
            first_prime=first_prime,
            last_prime=last_prime,
        )
        self._perf.disk_plan_writes += 1
        return mapped

    def _disk_unfiltered_plan(
        self,
        *,
        source_start: int,
    ) -> PrimeBlockPlan:
        loaded, key = self._load_disk_plan(
            0,
            source_start=source_start,
        )
        if loaded is not None:
            return loaded
        source = self._source_primes(source_start)
        if key is None or self._plan_cache is None:
            return build_prime_block_plan(
                source,
                block_size=self.block_size,
                superblock_fanout=self.superblock_fanout,
                eligible_primes=None,
                build_superblocks=True,
                pool_perf=self._perf,
            )

        build = None
        plan = None
        try:
            build = self._plan_cache.begin(key)
            plan = build_prime_block_plan(
                source,
                block_size=self.block_size,
                superblock_fanout=self.superblock_fanout,
                eligible_primes=None,
                build_superblocks=True,
                pool_perf=self._perf,
            )
            self._commit_disk_plan(build, plan)
            return plan
        except PlanCacheBusyError:
            if build is not None:
                build.abort()
        except (OSError, PlanCacheError, ValueError) as exc:
            if build is not None:
                build.abort()
            self.plan_cache_error = str(exc)
            self._plan_cache = None
            if plan is not None:
                return plan
        except BaseException:
            if build is not None:
                build.abort()
            raise
        return build_prime_block_plan(
            source,
            block_size=self.block_size,
            superblock_fanout=self.superblock_fanout,
            eligible_primes=None,
            build_superblocks=True,
            pool_perf=self._perf,
        )

    def _disk_filtered_plans(
        self,
        radicals: Sequence[int],
        *,
        source_start: int,
    ) -> Dict[int, PrimeBlockPlan]:
        """Load or build filtered plans with mmap-backed prime arrays."""
        radical_keys = tuple(sorted(set(int(r) for r in radicals)))
        results: Dict[int, PrimeBlockPlan] = {}
        keys: Dict[int, PlanCacheKey] = {}
        missing = []

        for radical in radical_keys:
            loaded, key = self._load_disk_plan(
                radical,
                source_start=source_start,
            )
            if loaded is not None:
                results[radical] = loaded
            else:
                missing.append(radical)
                if key is not None:
                    keys[radical] = key

        if not missing:
            return results
        cache = self._plan_cache
        if cache is None:
            return self._memory_filtered_plans(
                missing,
                source_start=source_start,
                destination=results,
            )

        builds: Dict[int, PlanCacheBuild] = {}
        fallback = []
        for radical in missing:
            key = keys.get(radical)
            if key is None:
                fallback.append(radical)
                continue
            try:
                builds[radical] = cache.begin(key)
            except PlanCacheBusyError:
                fallback.append(radical)
            except (OSError, PlanCacheError, ValueError) as exc:
                self.plan_cache_error = str(exc)
                fallback.append(radical)

        source = self._source_primes(source_start)
        if builds:
            try:
                counts = (
                    build_filtered_prime_pools_vectorized_memmap(
                        source,
                        builds,
                        chunk_primes=self.plan_chunk_primes,
                        pool_perf=self._perf,
                    )
                )
            except BaseException as exc:
                for build in builds.values():
                    build.abort()
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                self.plan_cache_error = str(exc)
                fallback.extend(builds)
                builds = {}
                counts = {}

            for radical, build in tuple(builds.items()):
                mapping = None
                plan = None
                try:
                    mapping = build.open_staging_primes(
                        counts[radical]
                    )
                    plan = build_prime_block_plan(
                        source,
                        block_size=self.block_size,
                        superblock_fanout=self.superblock_fanout,
                        eligible_primes=mapping,
                        build_superblocks=True,
                        pool_perf=self._perf,
                    )
                    prime_count = len(mapping)
                    if prime_count:
                        first_prime = int(mapping[0])
                        last_prime = int(mapping[-1])
                    else:
                        first_prime = 0
                        last_prime = 0
                    if isinstance(mapping, np.memmap):
                        mapping._mmap.close()
                    mapping = None
                    committed_mapping = self._commit_disk_plan(
                        build,
                        plan,
                        prime_count=prime_count,
                        first_prime=first_prime,
                        last_prime=last_prime,
                    )
                    assert committed_mapping is not None
                    results[radical] = PrimeBlockPlan(
                        primes=committed_mapping,
                        block_size=plan.block_size,
                        superblock_fanout=(
                            plan.superblock_fanout
                        ),
                        leaf_block_count=plan.leaf_block_count,
                        blocks=(),
                        superblocks=plan.superblocks,
                    )
                except BaseException as exc:
                    if isinstance(mapping, np.memmap):
                        try:
                            mapping._mmap.close()
                        except (BufferError, OSError):
                            pass
                    build.abort()
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        for other_radical, other in builds.items():
                            if other_radical != radical:
                                other.abort()
                        raise
                    self.plan_cache_error = str(exc)
                    fallback.append(radical)

        if fallback:
            self._memory_filtered_plans(
                fallback,
                source_start=source_start,
                destination=results,
            )
        return results

    def _memory_filtered_plans(
        self,
        radicals: Sequence[int],
        *,
        source_start: int,
        destination: Dict[int, PrimeBlockPlan] | None = None,
    ) -> Dict[int, PrimeBlockPlan]:
        results = destination if destination is not None else {}
        radical_keys = tuple(sorted(set(int(r) for r in radicals)))
        if not radical_keys:
            return results
        source = self._source_primes(source_start)
        pools = build_filtered_prime_pools_vectorized(
            source,
            radical_keys,
            chunk_primes=self.plan_chunk_primes,
            pool_perf=self._perf,
        )
        for radical in radical_keys:
            results[radical] = build_prime_block_plan(
                source,
                block_size=self.block_size,
                superblock_fanout=self.superblock_fanout,
                eligible_primes=pools[radical],
                build_superblocks=self._use_superblocks,
                pool_perf=self._perf,
            )
        return results

    def _prime_prefix_stop(self, limit: int) -> int:
        """Return the number of pool primes not exceeding ``limit``."""
        limit = int(limit)
        cached = self._prime_prefix_stop_cache.get(limit)
        if cached is not None:
            return cached
        stop = _typed_searchsorted_right(self._prime_view, limit)
        self._prime_prefix_stop_cache[limit] = stop
        return stop

    def _scan_start_leaf(
        self,
        plan: PrimeBlockPlan,
        exp: int,
        lower_limit: int | None,
    ) -> int:
        """Locate a certified prefix boundary in a shared normal plan.

        An interval plan already begins above ``lower_limit`` and scans from
        leaf zero.  A normal plan can be reused without rescanning the certified
        prefix by starting at the leaf containing its first eligible prime
        above the persisted limit.
        """
        if lower_limit is None:
            return 0

        normal = self._cached_normal_plan(exp)
        if plan is not normal:
            return 0

        cache_key = (
            self._normal_plan_key(exp),
            int(lower_limit),
        )
        cached = self._normal_plan_scan_leaf_cache.get(cache_key)
        if cached is not None:
            return cached

        plan_view, _dtype = _numpy_prime_view(plan.primes)
        first_new_index = _typed_searchsorted_right(
            plan_view,
            int(lower_limit),
        )
        if first_new_index >= len(plan_view):
            start_leaf = plan.leaf_block_count
        else:
            start_leaf = first_new_index // plan.block_size
        self._normal_plan_scan_leaf_cache[cache_key] = start_leaf
        return start_leaf

    def _candidate_leaf_count(
        self,
        plan: PrimeBlockPlan,
        start_leaf: int,
    ) -> int:
        """Return the logical leaf count reachable by this scan."""
        if start_leaf < 0 or start_leaf > plan.leaf_block_count:
            raise ValueError("candidate start leaf is out of range")
        if start_leaf == plan.leaf_block_count:
            return 0
        if not self._use_superblocks:
            return plan.leaf_block_count - start_leaf

        start_superblock = start_leaf // plan.superblock_fanout
        effective_start = plan.superblocks[start_superblock].start_leaf
        return plan.leaf_block_count - effective_start

    def _record_structure(
        self,
        p: int,
        exp: int,
        result: SigmaPoolAnalysis,
    ) -> None:
        if self._structure is None:
            return
        key = (int(p), int(exp))
        if key in self._structure.sigma_classified_keys:
            return
        self._structure.sigma_classified_keys.add(key)
        if result.exact:
            self._structure.sigma_exact += 1
        else:
            self._structure.sigma_outside += 1
            residual_bits = int(result.residual).bit_length()
            self._structure.outside_pool_sources[
                (key[0], key[1], residual_bits)
            ] += 1

    def flush(self) -> None:
        """Durably commit pending persistent-cache records."""
        if self._database is not None:
            self._database.flush()

    def close(self) -> None:
        seen_mappings: set[int] = set()
        plans = list(self._plans_by_radical.values())
        plans.extend(self._interval_plans.values())
        if self._full_plan is not None:
            plans.append(self._full_plan)
        for plan in plans:
            mapping = plan.primes
            if (
                isinstance(mapping, np.memmap)
                and id(mapping) not in seen_mappings
            ):
                seen_mappings.add(id(mapping))
                mmap_object = getattr(mapping, "_mmap", None)
                if mmap_object is not None:
                    try:
                        mmap_object.close()
                    except (BufferError, OSError):
                        pass
        if self._database is not None:
            self._database.close()
            self._database = None

    def prebuild_plans(
        self,
        exponents: Sequence[int],
    ) -> None:
        """Prebuild every plan required by the configured exponent set."""
        started = time.perf_counter_ns()

        unique_exponents = tuple(
            sorted(set(int(exp) for exp in exponents))
        )

        need_full_plan = any(
            (exp + 1) % 2 == 0
            for exp in unique_exponents
        )

        radical_keys = tuple(
            sorted({
                squarefree_kernel(exp + 1)
                for exp in unique_exponents
                if (exp + 1) % 2 == 1
            })
        )

        missing_radicals = tuple(
            radical
            for radical in radical_keys
            if radical not in self._plans_by_radical
        )

        if missing_radicals:
            for radical in missing_radicals:
                self._evict_interval_plans_for_key(radical)
            if self._plan_cache is not None:
                built_plans = self._disk_filtered_plans(
                    missing_radicals,
                    source_start=0,
                )
                self._plans_by_radical.update(built_plans)
            else:
                filtered_pools = (
                    build_filtered_prime_pools_vectorized(
                        self.primes,
                        missing_radicals,
                        chunk_primes=self.plan_chunk_primes,
                        pool_perf=self._perf,
                    )
                )

                for radical in missing_radicals:
                    self._plans_by_radical[radical] = (
                        build_prime_block_plan(
                            self.primes,
                            block_size=self.block_size,
                            superblock_fanout=self.superblock_fanout,
                            eligible_primes=filtered_pools[radical],
                            build_superblocks=self._use_superblocks,
                            pool_perf=self._perf,
                        )
                    )

        if need_full_plan and self._full_plan is None:
            self._evict_interval_plans_for_key(0)
            if self._plan_cache is not None:
                self._full_plan = self._disk_unfiltered_plan(
                    source_start=0,
                )
            else:
                self._full_plan = build_prime_block_plan(
                    self.primes,
                    block_size=self.block_size,
                    superblock_fanout=self.superblock_fanout,
                    eligible_primes=None,
                    build_superblocks=self._use_superblocks,
                    pool_perf=self._perf,
                )

        self._perf.filtered_plan_count = len(self._plans_by_radical)
        self._perf.full_plan_built = self._full_plan is not None
        self._perf.plan_prebuild_ns += (time.perf_counter_ns() - started)

    def plan_for_exp(
        self,
        exp: int,
        *,
        lower_limit: int | None = None,
    ) -> PrimeBlockPlan:
        """Return a full-window or incremental necessary-order plan."""
        if lower_limit is not None and lower_limit >= 3:
            normal = self._cached_normal_plan(exp)
            if normal is not None:
                return normal
            prefix_stop = self._prime_prefix_stop(lower_limit)
            if (
                prefix_stop / len(self._prime_view)
                >= POOL_INCREMENTAL_MIN_PREFIX_FRACTION
            ):
                return self._interval_plan_for_exp(
                    exp,
                    lower_limit=int(lower_limit),
                )

        n = exp + 1

        if n % 2 == 0:
            if self._full_plan is None:
                self._maybe_adaptive_prebuild(exp)
            if self._full_plan is None:
                self._evict_interval_plans_for_key(0)
                if self._plan_cache is not None:
                    self._full_plan = self._disk_unfiltered_plan(
                        source_start=0,
                    )
                else:
                    self._full_plan = build_prime_block_plan(
                        self.primes,
                        block_size=self.block_size,
                        superblock_fanout=self.superblock_fanout,
                        eligible_primes=None,
                        build_superblocks=self._use_superblocks,
                        pool_perf=self._perf,
                    )
            return self._full_plan

        radical = squarefree_kernel(n)
        cached = self._plans_by_radical.get(radical)
        if cached is not None:
            return cached

        self._maybe_adaptive_prebuild(exp)
        cached = self._plans_by_radical.get(radical)
        if cached is not None:
            return cached

        self._evict_interval_plans_for_key(radical)
        if self._plan_cache is not None:
            plan = self._disk_filtered_plans(
                [radical],
                source_start=0,
            )[radical]
        else:
            filtered = build_filtered_prime_pools_vectorized(
                self.primes,
                [radical],
                chunk_primes=self.plan_chunk_primes,
                pool_perf=self._perf,
            )[radical]

            plan = build_prime_block_plan(
                self.primes,
                block_size=self.block_size,
                superblock_fanout=self.superblock_fanout,
                eligible_primes=filtered,
                build_superblocks=self._use_superblocks,
                pool_perf=self._perf,
            )

        self._plans_by_radical[radical] = plan
        self._perf.filtered_plan_count = len(self._plans_by_radical)
        return plan

    def _interval_plan_for_exp(
        self,
        exp: int,
        *,
        lower_limit: int,
    ) -> PrimeBlockPlan:
        """Build a shared plan over primes in ``(lower_limit, P]``."""
        start = self._prime_prefix_stop(lower_limit)
        if start >= len(self._prime_view):
            raise ValueError("incremental prime interval is empty")

        n = exp + 1
        radical = 0 if n % 2 == 0 else squarefree_kernel(n)
        key = (start, radical)
        cached = self._interval_plans.get(key)
        if cached is not None:
            return cached

        interval = self._prime_view[start:]
        if radical == 0:
            if self._plan_cache is not None:
                plan = self._disk_unfiltered_plan(
                    source_start=start,
                )
                self._interval_plans[key] = plan
                return plan
            eligible = interval
        else:
            if self._plan_cache is not None:
                plan = self._disk_filtered_plans(
                    [radical],
                    source_start=start,
                )[radical]
                self._interval_plans[key] = plan
                return plan
            eligible = build_filtered_prime_pools_vectorized(
                    interval,
                    [radical],
                    chunk_primes=self.plan_chunk_primes,
                    pool_perf=self._perf,
                )[radical]

        plan = build_prime_block_plan(
            interval,
            block_size=self.block_size,
            superblock_fanout=self.superblock_fanout,
            eligible_primes=eligible,
            build_superblocks=self._use_superblocks,
            pool_perf=self._perf,
        )
        self._interval_plans[key] = plan
        return plan

    def _analysis_from_exact(
        self,
        valuations: Dict[int, int],
    ) -> SigmaPoolAnalysis:
        inside: Dict[int, int] = {}
        outside_residual = mpz(1)
        outside_witness: Optional[int] = None

        for q, exponent in valuations.items():
            if q <= self.prime_limit:
                inside[q] = exponent
                continue
            outside_residual *= mpz(q) ** exponent
            if outside_witness is None or q < outside_witness:
                outside_witness = q

        if outside_witness is None:
            return SigmaPoolAnalysis(
                exact=True,
                valuations=valuations,
                residual=mpz(1),
            )
        return SigmaPoolAnalysis(
            exact=False,
            valuations=inside,
            residual=outside_residual,
            outside_witness=outside_witness,
        )

    def _disable_database(self, exc: Exception) -> None:
        self.database_error = str(exc)
        database = self._database
        self._database = None
        if database is not None:
            try:
                database.close()
            except (OSError, sqlite3.Error):
                pass

    def _load_persisted(
        self,
        p: int,
        exp: int,
        *,
        sigma_odd: mpz,
    ) -> PersistedSigmaRecord | None:
        if self._database is None:
            return None
        try:
            candidates, invalid = self._database.load_candidates(
                p,
                exp,
                sigma_odd=sigma_odd,
            )
        except sqlite3.Error as exc:
            self._perf.persistent_invalid += 1
            self._disable_database(exc)
            return None

        self._perf.persistent_invalid += invalid
        for record in candidates:
            if record.exact:
                self._perf.persistent_hits += 1
                return record
            if record.scanned_limit > self.prime_limit:
                continue
            expected_digest = self._pool_digest_for_limit(
                record.scanned_limit
            )
            if (
                expected_digest is not None
                and expected_digest == record.pool_digest
            ):
                self._perf.persistent_hits += 1
                return record

        self._perf.persistent_misses += 1
        return None

    def _store_persisted(
        self,
        p: int,
        exp: int,
        *,
        sigma_odd: mpz,
        result: SigmaPoolAnalysis,
        exact_valuations: Dict[int, int] | None = None,
    ) -> None:
        if self._database is None:
            return
        try:
            if exact_valuations is not None:
                self._database.store(
                    p=p,
                    exp=exp,
                    exact=True,
                    scanned_limit=0,
                    pool_digest=b"",
                    valuations=exact_valuations,
                    residual=mpz(1),
                    sigma_odd=sigma_odd,
                )
                return

            pool_digest = self._pool_digest_for_limit(
                self.prime_limit
            )
            if pool_digest is None:
                raise ValueError(
                    "current prime pool has no certifiable upper bound"
                )
            self._database.store(
                p=p,
                exp=exp,
                exact=result.exact,
                scanned_limit=self.prime_limit,
                pool_digest=pool_digest,
                valuations=result.valuations,
                residual=result.residual,
                sigma_odd=sigma_odd,
            )
        except (OSError, sqlite3.Error, ValueError) as exc:
            self._perf.persistent_invalid += 1
            self._disable_database(exc)

    def analyze(self, p: int, exp: int) -> SigmaPoolAnalysis:
        key = (p, exp)
        cached = self._cache.get(key)
        if cached is not None:
            self._perf.hits += 1
            return cached

        self._perf.misses += 1
        started = time.perf_counter()

        # Fast path: a globally exact factorisation already exists
        exact_cached = _SIG_VALUATIONS.get(key)
        if exact_cached is not None:
            result = self._analysis_from_exact(exact_cached)
            if result.exact:
                self._perf.exact_from_global_cache += 1
            else:
                self._perf.outside_from_global_cache += 1
            if self._database is not None:
                sigma_odd = mpz(sigma_prime_power(p, exp))
                sigma_odd, _v2 = _remove_all(sigma_odd, 2)
                self._store_persisted(
                    p,
                    exp,
                    sigma_odd=sigma_odd,
                    result=result,
                    exact_valuations=exact_cached,
                )
            self._record_structure(p, exp, result)
            self._cache[key] = result
            return result

        sigma_odd = mpz(sigma_prime_power(p, exp))
        sigma_odd, _v2 = _remove_all(sigma_odd, 2)

        persisted = self._load_persisted(
            p,
            exp,
            sigma_odd=sigma_odd,
        )
        if persisted is not None and persisted.exact:
            _SIG_VALUATIONS[key] = persisted.valuations
            _SIG_FACTORS[key] = set(persisted.valuations)
            result = self._analysis_from_exact(
                persisted.valuations
            )
            self._record_structure(p, exp, result)
            self._cache[key] = result
            return result

        if persisted is None:
            residual = mpz(sigma_odd)
            inside: Dict[int, int] = {}
            lower_limit = None
        else:
            residual = mpz(persisted.residual)
            inside = dict(persisted.valuations)
            lower_limit = persisted.scanned_limit

            if lower_limit == self.prime_limit:
                result = SigmaPoolAnalysis(
                    exact=False,
                    valuations=inside,
                    residual=residual,
                )
                self._record_structure(p, exp, result)
                self._cache[key] = result
                return result

        plan = self.plan_for_exp(
            exp,
            lower_limit=lower_limit,
        )
        start_leaf = self._scan_start_leaf(
            plan,
            exp,
            lower_limit,
        )
        self._perf.candidate_leaf_blocks += (
            self._candidate_leaf_count(plan, start_leaf)
        )

        scan_started = time.perf_counter_ns()
        residual = self._scan(
            residual,
            inside,
            plan,
            self._perf,
            start_leaf=start_leaf,
        )
        self._perf.cold_scan_ns += (time.perf_counter_ns() - scan_started)

        if residual == 1:
            result = SigmaPoolAnalysis(exact=True, valuations=inside, residual=mpz(1))
            _SIG_VALUATIONS[key] = inside
            _SIG_FACTORS[key] = set(inside)
        else:
            result = SigmaPoolAnalysis(exact=False, valuations=inside, residual=residual)

        self._store_persisted(
            p,
            exp,
            sigma_odd=sigma_odd,
            result=result,
            exact_valuations=inside if result.exact else None,
        )
        self._record_structure(p, exp, result)

        elapsed = time.perf_counter() - started
        self._perf.analysis_ns += int(elapsed * 1_000_000_000)
        self._perf.record_slow_analysis(
            elapsed, p, exp, int(residual).bit_length(), result.exact,
        )

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
        if _sigma_map_perf is not None:
            _sigma_map_perf.sigma_map_hits += 1
        return cached

    if _sigma_map_perf is not None:
        _sigma_map_perf.sigma_map_misses += 1
    t0 = time.perf_counter()
    sig = int(sigma_prime_power(p, a))
    valuations = {q: e for q, e in factorize(sig) if q != 2}
    elapsed = time.perf_counter() - t0
    if _sigma_map_perf is not None:
        _sigma_map_perf.sigma_map_factor_seconds += elapsed
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


def compute_toxic_skip_list(structure: "StructureMetrics") -> None:
    """Seed TOXIC_SKIP from contradiction attribution data."""
    from collections import Counter as _Counter
    excluded_counts: 'Counter[int]' = _Counter()
    for (q, reason), count in structure.contradiction_attribution.items():
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
