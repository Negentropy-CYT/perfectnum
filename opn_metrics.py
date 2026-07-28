"""
opn_metrics — typed observability data models.

Replaces the module-level Counter globals previously spread across
opn_core / opn_state / opn_search / opn_io with a single explicit
``RunMetrics`` object threaded through the call chain.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


# ═══════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════


class PruneReason(str, Enum):
    """Mathematical reason a branch was pruned (search-structure dimension)."""
    OUTSIDE_WINDOW = "outside_window"
    VALUATION_CONTRADICTION = "valuation_contradiction"
    RATIO_OVERSHOOT = "ratio_overshoot"
    RATIO_UNREACHABLE = "ratio_unreachable"
    FACTOR_SLOTS = "factor_slots"
    TOUCHARD = "touchard"
    EULER_FORM = "euler_form"
    EXCLUDED_PRIME = "excluded_prime"
    TERMINAL_RATIO = "terminal_ratio"
    TERMINAL_NO_EULER = "terminal_no_euler"
    CAPACITY_BOUND = "capacity_bound"
    DUPLICATE_STATE = "duplicate_state"


class PruneMechanism(str, Enum):
    """Execution path that detected the prune (performance dimension)."""
    KNOWN_OUTSIDE_CACHE = "known_outside_cache"
    COLD_POOL_CERTIFICATE = "cold_pool_certificate"
    EXACT_FACTOR_OUTSIDE = "exact_factor_outside"
    PRECLONE_VALUATION = "preclone_valuation"
    POSTCLONE_VALUATION = "postclone_valuation"
    PROSPECTIVE_RATIO = "prospective_ratio"
    TAIL_RATIO_BOUND = "tail_ratio_bound"
    INTERVAL_BOUND = "interval_bound"
    DIRECT_DOMAIN_CHECK = "direct_domain_check"
    TERMINAL_CHECK = "terminal_check"


class CloneEffect(str, Enum):
    """Whether a prune happened before or after clone()."""
    NONE = "none"
    AVOIDED = "avoided"
    WASTED = "wasted"


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

_HEADROOM_BUCKETS = (
    "<1e-6",
    "1e-6-1e-5",
    "1e-5-1e-4",
    "1e-4-1e-3",
    "1e-3-1e-2",
    ">1e-2",
)


def headroom_bucket(value: float) -> str:
    """Classify a ratio headroom value into a bucket string."""
    if value <= 1e-6:
        return "<1e-6"
    if value <= 1e-5:
        return "1e-6-1e-5"
    if value <= 1e-4:
        return "1e-5-1e-4"
    if value <= 1e-3:
        return "1e-4-1e-3"
    if value <= 1e-2:
        return "1e-3-1e-2"
    return ">1e-2"


# ═══════════════════════════════════════════════════════════════
# Structure metrics (search-tree shape, mathematical results)
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class StructureMetrics:
    """Mathematical search-tree counters — deterministic given the search box."""

    prune_reasons: Counter[str] = field(default_factory=Counter)

    productive_states: int = 0
    depth_histogram: Counter[int] = field(default_factory=Counter)
    depth_factor_map: Counter[tuple[int, int]] = field(
        default_factory=Counter
    )

    ratio_headroom: Counter[str] = field(default_factory=Counter)
    headroom_by_factor: Counter[tuple[int, str]] = field(
        default_factory=Counter
    )

    obligation_signatures: Counter[tuple] = field(default_factory=Counter)
    pending_prime_frequency: Counter[int] = field(default_factory=Counter)

    contradiction_attribution: Counter[tuple[int, str]] = field(
        default_factory=Counter
    )
    propagation_edges: Counter[tuple[int, int]] = field(
        default_factory=Counter
    )
    propagation_exp_edges: Counter[tuple[int, int, int]] = field(
        default_factory=Counter
    )

    outside_pool_sources: Counter[tuple[int, int, int]] = field(
        default_factory=Counter
    )
    outside_window_sources: Counter[tuple[int, int, int]] = field(
        default_factory=Counter
    )

    sigma_exact: int = 0
    sigma_outside: int = 0

    def record_productive(
        self,
        *,
        depth: int,
        assigned_count: int,
        pending: Iterable[int],
        ratio_num: int,
        ratio_den: int,
        target_num: int,
        target_den: int,
    ) -> None:
        """Record one productive clone's structural statistics."""
        ratio = ratio_num / ratio_den
        headroom = target_num / target_den - ratio
        bucket = headroom_bucket(headroom)

        self.productive_states += 1
        self.depth_histogram[depth] += 1
        self.depth_factor_map[(depth, assigned_count)] += 1
        self.ratio_headroom[bucket] += 1
        self.headroom_by_factor[(assigned_count, bucket)] += 1

        for q in pending:
            self.pending_prime_frequency[q] += 1

        coarse = (
            int(math.floor(-math.log10(headroom)))
            if headroom > 0
            else 99
        )

        self.obligation_signatures[
            (frozenset(pending), assigned_count, coarse)
        ] += 1


# ═══════════════════════════════════════════════════════════════
# Pool performance
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PoolPerformance:
    """Timing and count data for the sigma-pool analyser."""

    hits: int = 0
    misses: int = 0

    exact_from_global_cache: int = 0
    outside_from_global_cache: int = 0

    candidate_leaf_blocks: int = 0
    superblocks_tested: int = 0
    positive_superblocks: int = 0
    leaf_blocks_tested: int = 0
    leaf_blocks_skipped: int = 0
    positive_blocks: int = 0
    factors_removed: int = 0

    analysis_ns: int = 0
    cold_scan_ns: int = 0
    plan_prebuild_ns: int = 0
    plan_filter_ns: int = 0
    plan_filter_count_ns: int = 0
    plan_filter_fill_ns: int = 0
    plan_filter_source_values: int = 0
    leaf_product_ns: int = 0
    superblock_product_ns: int = 0

    filtered_prime_values: int = 0
    filtered_prime_bytes: int = 0
    filtered_plan_count: int = 0
    full_plan_built: bool = False

    plans_built: int = 0
    plan_leaf_blocks: int = 0
    plan_superblocks: int = 0

    slowest: list[tuple[float, int, int, int, bool]] = field(
        default_factory=list,
        repr=False,
    )

    def record_slow_analysis(
        self,
        seconds: float,
        p: int,
        exp: int,
        residual_bits: int,
        exact: bool,
    ) -> None:
        """Keep the top-15 slowest pool analyses."""
        item = (seconds, p, exp, residual_bits, exact)

        if len(self.slowest) < 15:
            heapq.heappush(self.slowest, item)
        elif seconds > self.slowest[0][0]:
            heapq.heapreplace(self.slowest, item)

    def sorted_slowest(self) -> list[tuple[float, int, int, int, bool]]:
        """Return slowest analyses sorted descending."""
        return sorted(self.slowest, reverse=True)


# ═══════════════════════════════════════════════════════════════
# Performance metrics (execution)
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class PerformanceMetrics:
    """Execution-performance counters — vary with hardware and implementation."""

    prune_mechanisms: Counter[str] = field(default_factory=Counter)

    branches_considered: int = 0
    clones_actual: int = 0
    clones_avoided: int = 0
    clones_wasted: int = 0
    clone_payload: Counter[int] = field(default_factory=Counter)

    pool: PoolPerformance = field(default_factory=PoolPerformance)

    ratio_upper_calls: int = 0
    ratio_upper_ns: int = 0
    fermat_debt_calls: int = 0
    fermat_debt_ns: int = 0

    sigma_map_hits: int = 0
    sigma_map_misses: int = 0
    sigma_map_factor_seconds: float = 0.0

    cache_sizes: dict[str, int] = field(default_factory=dict)
    memory_phases: dict[str, dict[str, int]] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# RunMetrics — top-level container
# ═══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class RunMetrics:
    """Single source of truth for all search observability data."""

    schema_version: int = 1
    structure: StructureMetrics = field(default_factory=StructureMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    def record_prune(
        self,
        *,
        reason: PruneReason,
        mechanism: PruneMechanism,
        clone_effect: CloneEffect = CloneEffect.NONE,
    ) -> None:
        """Record one prune event in both reason and mechanism dimensions."""
        self.structure.prune_reasons[reason.value] += 1
        self.performance.prune_mechanisms[mechanism.value] += 1
        self.performance.branches_considered += 1

        if clone_effect is CloneEffect.AVOIDED:
            self.performance.clones_avoided += 1
        elif clone_effect is CloneEffect.WASTED:
            self.performance.clones_wasted += 1

    def record_clone(self, assigned_count: int) -> None:
        """Record one actual clone (state copy)."""
        self.performance.branches_considered += 1
        self.performance.clones_actual += 1
        self.performance.clone_payload[assigned_count] += 1

    def checkpoint_payload(self) -> dict:
        """Return a pickle-compatible dict for checkpoint serialization."""
        return {
            "schema_version": self.schema_version,
            "structure": self.structure,
            "performance": self.performance,
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict) -> "RunMetrics":
        """Restore RunMetrics from a checkpoint dict."""
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported metrics schema version")

        return cls(
            schema_version=1,
            structure=payload["structure"],
            performance=payload["performance"],
        )
