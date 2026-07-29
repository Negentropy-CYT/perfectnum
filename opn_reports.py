"""
opn_reports — structure / performance / summary report writers.

Replaces the monolithic ``write_telemetry_report()`` with separate
semantically-clean output files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from opn_metrics import (
    RunMetrics,
    _HEADROOM_BUCKETS,
)


def prepare_run_directory(run_id: str) -> Path:
    """Create and return the per-run output directory."""
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    git_commit: str,
    git_dirty: bool | None = None,
    started_at: str,
    config: dict,
    pruning_policy: str = "",
    telemetry_schema_version: int = 0,
) -> None:
    """Write manifest.json — machine-readable run identity and configuration."""
    doc = {
        "schema_version": 1,
        "telemetry_schema_version": telemetry_schema_version,
        "pruning_policy": pruning_policy,
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "started_at": started_at,
        "configuration": {
            "max_prime": config["max_prime"],
            "max_factors": config["max_factors"],
            "max_exp": config["max_exp"],
            "target_num": config["target_num"],
            "target_den": config["target_den"],
            "require_euler": config["require_euler"],
            "propagate": config["propagate"],
            "pool_gcd_mode": config["pool_gcd_mode"],
            "pool_fanout": config["pool_fanout"],
            "q3_prepool_mode": config.get("q3_prepool_mode"),
            "domain_ratio_mode": config.get("domain_ratio_mode"),
            "pending_selection": config.get("pending_selection"),
            "sigma_database_enabled": config.get(
                "sigma_database_enabled"
            ),
            "pool_plan_build_policy": config.get(
                "pool_plan_build_policy"
            ),
            "pool_plan_disk_cache_enabled": config.get(
                "pool_plan_disk_cache_enabled"
            ),
        },
    }
    with (run_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def write_summary(
    run_dir: Path,
    *,
    run_id: str,
    git_commit: str,
    status: str,
    started_at: str = "",
    max_prime: int,
    max_factors: int,
    max_exp: int,
    target_num: int,
    target_den: int,
    pool_gcd_mode: str,
    pool_fanout: int,
    solutions_found: int,
    elapsed_seconds: float,
    metrics: RunMetrics,
) -> None:
    """Write run_summary.txt — index of the run and its reports."""
    lines: List[str] = []
    w = lines.append

    w("Run Summary")
    w("=" * 60)
    w(f"  run_id:      {run_id}")
    w(f"  git_commit:  {git_commit}")
    w(f"  status:      {status}")
    w(f"  started:     {started_at}")
    w("")
    w("Configuration")
    w("-" * 40)
    w(f"  MAX_PRIME              {max_prime}")
    w(f"  MAX_FACTORS            {max_factors}")
    w(f"  MAX_EXP                {max_exp}")
    w(f"  target                 {target_num}/{target_den}")
    w(f"  POOL_GCD_MODE          {pool_gcd_mode}")
    w(f"  POOL_SUPERBLOCK_FANOUT  {pool_fanout}")
    w("")
    w("Result")
    w("-" * 40)
    w(f"  solutions              {solutions_found}")
    w(f"  elapsed_seconds        {elapsed_seconds:.1f}")
    w(f"  attempted_branches     {metrics.performance.branches_considered:,}")
    w(f"  productive_states      {metrics.structure.productive_states:,}")
    w(f"  clones_actual          {metrics.performance.clones_actual:,}")
    perf = metrics.performance
    if perf.clones_actual + perf.clones_avoided > 0:
        avoid_rate = 100.0 * perf.clones_avoided / (
            perf.clones_actual + perf.clones_avoided
        )
        w(f"  avoidance_rate         {avoid_rate:.1f}%")
    w("")
    w("Reports")
    w("-" * 40)
    w("  structure:    structure.txt / structure.json")
    w("  performance:  performance.txt / performance.json")
    w("  samples:      performance_samples.csv")
    w("")

    with (run_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── structure report ──────────────────────────────────────────


def _structure_lines(
    metrics: RunMetrics,
    *,
    max_prime: int,
    max_factors: int,
    max_exp: int,
    target_num: int,
    target_den: int,
    require_euler: bool,
    propagate: bool,
    elapsed: float,
    solutions_found: int,
    _sig_factors: dict | None = None,
) -> List[str]:
    s = metrics.structure
    lines: List[str] = []
    w = lines.append

    mode = f"target={target_num}/{target_den}"
    mode += " (chain)" if propagate else " (DFS)"
    if not require_euler:
        mode += " no-euler"

    from opn_core import valid_euler_exponents, valid_even_exponents

    euler_exp = valid_euler_exponents(1, max_exp)
    even_exp = valid_even_exponents(2, max_exp)
    w(f"# P={max_prime}  f≤{max_factors}  exp={max_exp}  "
      f"euler={euler_exp}  even={even_exp}")
    w(f"# {elapsed:.1f}s  |  {solutions_found} solutions  |  {mode}\n")

    # ── prune reasons ──
    pr_total = sum(s.prune_reasons.values())
    if pr_total:
        w("\n## Prune statistics")
        w(f"{'reason':>14}  {'count':>10}  {'%prune':>7}")
        for k, v in s.prune_reasons.most_common():
            pct = 100.0 * v / pr_total
            w(f"{k:>14}  {v:>10,}  {pct:>6.1f}%")

    # ── clone effectiveness ──
    p = metrics.performance
    attempted = p.clones_actual + p.clones_avoided
    if attempted:
        w("\n## Clone effectiveness")
        w(f"  attempted branches  {attempted:>10,}")
        w(f"  actual clones       {p.clones_actual:>10,}")
        w(f"  avoided (pre-clone) {p.clones_avoided:>10,}")
        avoid_rate = 100.0 * p.clones_avoided / attempted
        w(f"  avoidance rate      {avoid_rate:>10.1f}%")
        productive = s.productive_states
        wasted = p.clones_wasted
        overhead = p.clones_actual - productive - wasted
        w(f"    productive        {productive:>10,}  "
          f"({100.0*productive/p.clones_actual:5.1f}% of actual)")
        w(f"    wasted (post-cln) {wasted:>10,}  "
          f"({100.0*wasted/p.clones_actual:5.1f}% of actual)")
        w(f"    overhead (other)  {overhead:>10,}  "
          f"({100.0*overhead/p.clones_actual:5.1f}% of actual)")

    # ── depth histogram ──
    if s.depth_histogram:
        dt = sum(s.depth_histogram.values())
        w("\n## Depth histogram")
        for d in sorted(s.depth_histogram):
            pct = 100.0 * s.depth_histogram[d] / dt
            w(f"  depth {d:>2}: {s.depth_histogram[d]:>12,}  ({pct:4.1f}%)")

    # ── clone payload ──
    if p.clone_payload:
        ct = sum(p.clone_payload.values())
        w("\n## Clone payload (|assigned|)")
        for k in sorted(p.clone_payload):
            w(f"  |f|={k:>2}:   {p.clone_payload[k]:>12,}  "
              f"({100.0*p.clone_payload[k]/ct:5.1f}%)")

    # ── ratio headroom ──
    if s.ratio_headroom:
        rt = sum(s.ratio_headroom.values())
        w("\n## Ratio headroom")
        for b in _HEADROOM_BUCKETS:
            v = s.ratio_headroom.get(b, 0)
            if v:
                w(f"  {b:>12}  {v:>12,}  ({100.0*v/rt:5.1f}%)")

    # ── headroom by |f| ──
    if s.headroom_by_factor:
        w("\n## Headroom by |f|")
        header = f"  {'|f|':>4}"
        for b in _HEADROOM_BUCKETS:
            header += f"  {b:>10}"
        w(header)
        f_levels = sorted(set(f for f, _ in s.headroom_by_factor))
        for nf in f_levels:
            row = f"  {nf:>4}"
            for b in _HEADROOM_BUCKETS:
                v = s.headroom_by_factor.get((nf, b), 0)
                row += f"  {v:>10,}"
            w(row)

    # ── contradiction attribution ──
    if s.contradiction_attribution:
        w("\n## Contradiction attribution (top-15)")
        for (q, reason), count in s.contradiction_attribution.most_common(15):
            w(f"  ({q:>4}, {reason:<14}) {count:>10,}")

    # ── propagation edges ──
    if s.propagation_edges:
        w("\n## Propagation edges (top-10)")
        for (p_edge, q), count in s.propagation_edges.most_common(10):
            w(f"  {p_edge:>4} → {q:<8}  {count:>10,}")

    if s.propagation_exp_edges:
        w("\n## Propagation edges by exponent (top-10)")
        for (p_edge, e, q), count in s.propagation_exp_edges.most_common(10):
            w(f"  {p_edge:>4}^{e} → {q:<8}  {count:>10,}")

    # ── depth × |f| ──
    if s.depth_factor_map:
        w("\n## Depth × |f| (top-15)")
        for (d, nf), count in s.depth_factor_map.most_common(15):
            w(f"  depth={d:>3}  |f|={nf}  {count:>12,}")

    # ── pending-prime frequency ──
    if s.pending_prime_frequency:
        w("\n## Pending-prime frequency (top-15)")
        for q, count in s.pending_prime_frequency.most_common(15):
            w(f"  {q:>12}  {count:>10,}")

    # ── sigma classifications ──
    w("\n## Sigma classifications")
    w(f"  exact:             {s.sigma_exact:>12,}")
    w(f"  outside-window:    {s.sigma_outside:>12,}")

    if s.sigma_exact_by_exp or s.sigma_outside_by_exp:
        w("\n## Sigma classifications by exponent")
        w(f"  {'exp':>4}  {'exact':>10}  {'outside':>10}  {'total':>10}  {'exact%':>7}")
        for exp in range(len(s.sigma_exact_by_exp)):
            e = s.sigma_exact_by_exp[exp]
            o = s.sigma_outside_by_exp[exp]
            t = e + o
            if t:
                w(f"  {exp:>4}  {e:>10,}  {o:>10,}  {t:>10,}  {100.0*e/t:>6.1f}%")

    if s.valuation_contradictions_by_exp:
        w("\n## Valuation contradictions by source exponent")
        w(f"  {'exp':>4}  {'excluded':>10}  {'overrun':>10}  {'budget':>10}  {'q=3 total':>10}")
        for exp, counts in enumerate(s.valuation_contradictions_by_exp):
            q3 = s.valuation_q3_by_exp[exp]
            if any(counts) or q3:
                w(f"  {exp:>4}  {counts[0]:>10,}  {counts[1]:>10,}  {counts[2]:>10,}  {q3:>10,}")

    # ── outside-pool residual sources ──
    if s.outside_pool_sources:
        w("\n## Outside-pool residual sources (top-10)")
        for (p_val, e, bits), count in s.outside_pool_sources.most_common(10):
            w(f"  {p_val:>6}^{e} → residual ({bits} bits)  x{count:>6}")

    # ── outside-window sources ──
    if s.outside_window_sources:
        w("\n## Outside-window sources (top-10)")
        for (p_val, e, q), count in s.outside_window_sources.most_common(10):
            w(f"  {p_val:>4}^{e} → {q:<12}  {count:>10,}")

    # ── frequent pending-prime resolvability ──
    if s.pending_prime_frequency and _sig_factors is not None:
        lines.extend(
            _pending_source_lines(s, _sig_factors, max_prime)
        )

    w("")
    return lines


def write_structure_text(
    run_dir: Path,
    metrics: RunMetrics,
    *,
    max_prime: int,
    max_factors: int,
    max_exp: int,
    target_num: int,
    target_den: int,
    require_euler: bool,
    propagate: bool,
    elapsed: float,
    solutions_found: int,
    _sig_factors: dict | None = None,
) -> None:
    """Write structure.txt — mathematical search-tree shape."""
    lines = _structure_lines(
        metrics,
        max_prime=max_prime,
        max_factors=max_factors,
        max_exp=max_exp,
        target_num=target_num,
        target_den=target_den,
        require_euler=require_euler,
        propagate=propagate,
        elapsed=elapsed,
        solutions_found=solutions_found,
        _sig_factors=_sig_factors,
    )
    with (run_dir / "structure.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _counter_rows(counter, *, key_names=("first", "second")):
    """Convert a tuple-key Counter to a list of dicts for JSON serialization."""
    return [
        {key_names[0]: k[0], key_names[1]: k[1], "count": v}
        for k, v in counter.items()
    ]


def _counter_rows_3(counter, *, key_names=("first", "second", "third")):
    return [
        {key_names[0]: k[0], key_names[1]: k[1], key_names[2]: k[2], "count": v}
        for k, v in counter.items()
    ]


def write_structure_json(run_dir: Path, metrics: RunMetrics) -> None:
    """Write structure.json — machine-readable structural metrics."""
    s = metrics.structure

    doc = {
        "schema_version": metrics.schema_version,
        "productive_states": s.productive_states,
        "prune_reasons": dict(s.prune_reasons),
        "depth_histogram": dict(s.depth_histogram),
        "depth_factor_map": _counter_rows(s.depth_factor_map),
        "ratio_headroom": dict(s.ratio_headroom),
        "headroom_by_factor": [
            {"assigned_count": k[0], "bucket": k[1], "count": v}
            for k, v in s.headroom_by_factor.items()
        ],
        "contradiction_attribution": [
            {"prime": k[0], "reason": k[1], "count": v}
            for k, v in s.contradiction_attribution.items()
        ],
        "propagation_edges": _counter_rows(s.propagation_edges),
        "propagation_exp_edges": _counter_rows_3(
            s.propagation_exp_edges,
            key_names=("source", "exp", "target"),
        ),
        "outside_pool_sources": _counter_rows_3(
            s.outside_pool_sources,
            key_names=("prime", "exp", "residual_bits"),
        ),
        "outside_window_sources": _counter_rows_3(
            s.outside_window_sources,
            key_names=("prime", "exp", "outside_prime"),
        ),
        "sigma_exact": s.sigma_exact,
        "sigma_outside": s.sigma_outside,
        "sigma_by_exponent": [
            {
                "exp": exp,
                "exact": exact,
                "outside": s.sigma_outside_by_exp[exp],
            }
            for exp, exact in enumerate(s.sigma_exact_by_exp)
            if exact or s.sigma_outside_by_exp[exp]
        ],
        "valuation_contradictions_by_exponent": [
            {
                "exp": exp,
                "excluded": counts[0],
                "overrun": counts[1],
                "budget": counts[2],
                "q3_total": s.valuation_q3_by_exp[exp],
            }
            for exp, counts in enumerate(
                s.valuation_contradictions_by_exp
            )
            if any(counts) or s.valuation_q3_by_exp[exp]
        ],
        "pending_prime_frequency": dict(s.pending_prime_frequency),
    }

    with (run_dir / "structure.json").open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


# ── performance report ────────────────────────────────────────


def _performance_lines(
    metrics: RunMetrics,
    *,
    elapsed: float,
    sampled_peak_rss: int,
) -> List[str]:
    p = metrics.performance
    pool = p.pool
    lines: List[str] = []
    w = lines.append

    w(f"# elapsed={elapsed:.1f}s  peak_rss_bytes={sampled_peak_rss}\n")

    # ── phase timings ──
    w("## Phase timings")
    w(f"  {'phase':<28} {'seconds':>12}")
    w(f"  {'total':<28} {elapsed:>12.3f}")

    if pool.plan_prebuild_ns:
        w(f"  {'plan prebuild':<28} {pool.plan_prebuild_ns*1e-9:>12.3f}")
    if pool.cold_scan_ns:
        w(f"  {'cold scan':<28} {pool.cold_scan_ns*1e-9:>12.3f}")

    # ── plan build breakdown ──
    plan_items = [
        ("plan filter total", pool.plan_filter_ns),
        ("  filter count pass", pool.plan_filter_count_ns),
        ("  filter fill pass", pool.plan_filter_fill_ns),
        ("leaf product", pool.leaf_product_ns),
        ("superblock product", pool.superblock_product_ns),
    ]
    if any(ns for _, ns in plan_items):
        w("\n## Plan build timing")
        for label, ns_val in plan_items:
            if ns_val:
                w(f"  {label:<28} {ns_val*1e-9:>12.3f}")

    if pool.dynamic_leaf_product_ns:
        w("\n## Dynamic leaf rebuilding")
        w(
            f"  {'rebuild time':<28} "
            f"{pool.dynamic_leaf_product_ns*1e-9:>12.3f}"
        )
        w(
            f"  {'products rebuilt':<28} "
            f"{pool.dynamic_leaf_products_built:>12,}"
        )
        w(
            f"  {'prime values multiplied':<28} "
            f"{pool.dynamic_leaf_prime_values:>12,}"
        )

    if pool.pool_misses_by_exp:
        w("\n## Sigma-pool workload by exponent")
        w(f"  {'exp':>4}  {'mem-miss':>10}  {'cold-scan':>10}  {'scan-seconds':>13}  {'ms/scan':>8}")
        for exp in range(len(pool.pool_misses_by_exp)):
            miss = pool.pool_misses_by_exp[exp]
            scans = pool.cold_scans_by_exp[exp]
            ns_val = pool.cold_scan_ns_by_exp[exp]
            if miss or scans or ns_val:
                ms = ns_val / (scans * 1e6) if scans else 0
                w(f"  {exp:>4}  {miss:>10,}  {scans:>10,}  {ns_val*1e-9:>13.3f}  {ms:>7.1f}")

    # ── sigma pool cache ──
    w("\n## Sigma pool cache")
    w(f"  hits:               {pool.hits:>12,}")
    w(f"  misses:             {pool.misses:>12,}")
    total_pool_calls = pool.hits + pool.misses
    if total_pool_calls:
        w(f"  hit rate:           {100.0*pool.hits/total_pool_calls:>11.1f}%")
    w(f"  exact_from_global:  {pool.exact_from_global_cache:>12,}")
    w(f"  outside_from_global:{pool.outside_from_global_cache:>12,}")
    w(f"  persistent hits:    {pool.persistent_hits:>12,}")
    w(f"  persistent misses:  {pool.persistent_misses:>12,}")
    w(f"  persistent invalid: {pool.persistent_invalid:>12,}")
    w(f"  disk-plan hits:     {pool.disk_plan_hits:>12,}")
    w(f"  disk-plan misses:   {pool.disk_plan_misses:>12,}")
    w(f"  disk-plan invalid:  {pool.disk_plan_invalid:>12,}")
    w(f"  disk-plan writes:   {pool.disk_plan_writes:>12,}")

    # ── GCD block workload ──
    w("\n## GCD block workload")
    w(f"  candidate_leaf_blocks  {pool.candidate_leaf_blocks:>12,}")
    w(f"  superblocks_tested     {pool.superblocks_tested:>12,}")
    w(f"  positive_superblocks   {pool.positive_superblocks:>12,}")
    w(f"  leaf_blocks_tested     {pool.leaf_blocks_tested:>12,}")
    w(f"  leaf_blocks_skipped    {pool.leaf_blocks_skipped:>12,}")
    w(f"  positive_blocks        {pool.positive_blocks:>12,}")
    w(f"  factors_removed        {pool.factors_removed:>12,}")
    w(f"  logical_leaf_blocks    {pool.logical_leaf_blocks:>12,}")
    w(f"  resident_leaf_blocks   {pool.resident_leaf_blocks:>12,}")
    w(f"  dynamic_leaf_products  {pool.dynamic_leaf_products_built:>12,}")
    w(f"  dynamic_prime_values   {pool.dynamic_leaf_prime_values:>12,}")
    if pool.candidate_leaf_blocks:
        skip = 100.0 * (1 - pool.leaf_blocks_tested / pool.candidate_leaf_blocks)
        w(f"  screening skip rate    {skip:>12.3f}%")

    # ── prune mechanisms ──
    if p.prune_mechanisms:
        w("\n## Prune mechanisms (execution paths)")
        mech_total = sum(p.prune_mechanisms.values())
        for k, v in p.prune_mechanisms.most_common():
            w(f"  {k:<28} {v:>12,}  ({100.0*v/mech_total:5.1f}%)")

    # ── core timings ──
    if total_pool_calls or p.ratio_upper_calls or p.fermat_debt_calls:
        w("\n## Core timings")
        w(f"  {'phase':<20} {'calls':>10} {'seconds':>12} "
          f"{'us/call':>12} {'%runtime':>11}")

        def _timing(name, calls, ns_val):
            s_val = ns_val * 1e-9
            us = ns_val / calls / 1000.0 if calls else 0.0
            pct = 100.0 * s_val / elapsed if elapsed > 0 else 0.0
            w(f"  {name:<20} {calls:>10,} {s_val:>12.3f} "
              f"{us:>12.1f} {pct:>10.1f}%")

        _timing("pool analysis", total_pool_calls, pool.analysis_ns)
        _timing("ratio upper", p.ratio_upper_calls, p.ratio_upper_ns)
        _timing("fermat debt", p.fermat_debt_calls, p.fermat_debt_ns)

    # ── sigma-map cache ──
    if p.sigma_map_hits or p.sigma_map_misses:
        sm_total = p.sigma_map_hits + p.sigma_map_misses
        w("\n## σ-map cache")
        w(f"  hits:     {p.sigma_map_hits:>12,}")
        w(f"  misses:   {p.sigma_map_misses:>12,}")
        if sm_total:
            w(f"  hit rate: {100.0*p.sigma_map_hits/sm_total:>11.1f}%")
        w(f"  factor s: {p.sigma_map_factor_seconds:>12.1f}")

    # ── slowest pool analyses ──
    slow = pool.sorted_slowest()
    if slow:
        w("\n## Slowest pool analyses (top-15)")
        w(f"  {'p':>6}  {'exp':>3}  {'residual bits':>13}  "
          f"{'exact':>5}  {'seconds':>8}")
        for secs, p_val, a, bits, exact in slow:
            w(f"  {p_val:>6}  {a:>3}  {bits:>13}  "
              f"{str(exact):>5}  {secs:>8.3f}")

    # ── memory ──
    w("\n## Memory")
    w(f"  filtered pool storage MiB  {pool.filtered_prime_bytes/1024**2:>12.1f}")
    w(f"  filtered plans             {pool.filtered_plan_count:>12,}")
    w(f"  full plan built            {str(pool.full_plan_built):>12}")
    w(f"  sampled peak RSS MiB       {sampled_peak_rss/1024**2:>12.1f}")

    if p.memory_phases:
        w("  Phase snapshots:")
        for phase, mem in p.memory_phases.items():
            rss_mib = mem.get("rss_bytes", 0) / 1024**2
            w(f"    {phase:<28} RSS={rss_mib:>8.1f} MiB")

    w("")
    return lines


def write_performance_text(
    run_dir: Path,
    metrics: RunMetrics,
    *,
    elapsed: float,
    sampled_peak_rss: int,
) -> None:
    """Write performance.txt — execution-performance summary."""
    lines = _performance_lines(
        metrics,
        elapsed=elapsed,
        sampled_peak_rss=sampled_peak_rss,
    )
    with (run_dir / "performance.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_performance_json(
    run_dir: Path,
    metrics: RunMetrics,
    *,
    elapsed: float,
    sampled_peak_rss: int,
) -> None:
    """Write performance.json — machine-readable performance metrics."""
    perf = metrics.performance
    pool = perf.pool

    doc: Dict[str, Any] = {
        "schema_version": metrics.schema_version,
        "elapsed_seconds": elapsed,
        "sampled_peak_rss_bytes": sampled_peak_rss,
        "prune_mechanisms": dict(perf.prune_mechanisms),
        "branches_considered": perf.branches_considered,
        "clones_actual": perf.clones_actual,
        "clones_avoided": perf.clones_avoided,
        "clones_wasted": perf.clones_wasted,
        "clone_payload": dict(perf.clone_payload),
        "pool": {
            "hits": pool.hits,
            "misses": pool.misses,
            "exact_from_global_cache": pool.exact_from_global_cache,
            "outside_from_global_cache": pool.outside_from_global_cache,
            "persistent_hits": pool.persistent_hits,
            "persistent_misses": pool.persistent_misses,
            "persistent_invalid": pool.persistent_invalid,
            "disk_plan_hits": pool.disk_plan_hits,
            "disk_plan_misses": pool.disk_plan_misses,
            "disk_plan_invalid": pool.disk_plan_invalid,
            "disk_plan_writes": pool.disk_plan_writes,
            "cold_scans": pool.cold_scans,
            "pool_by_exponent": [
                {
                    "exp": exp,
                    "memory_misses": pool.pool_misses_by_exp[exp],
                    "cold_scans": pool.cold_scans_by_exp[exp],
                    "cold_scan_ns": pool.cold_scan_ns_by_exp[exp],
                }
                for exp in range(len(pool.pool_misses_by_exp))
                if (
                    pool.pool_misses_by_exp[exp]
                    or pool.cold_scans_by_exp[exp]
                    or pool.cold_scan_ns_by_exp[exp]
                )
            ],
            "candidate_leaf_blocks": pool.candidate_leaf_blocks,
            "superblocks_tested": pool.superblocks_tested,
            "positive_superblocks": pool.positive_superblocks,
            "leaf_blocks_tested": pool.leaf_blocks_tested,
            "leaf_blocks_skipped": pool.leaf_blocks_skipped,
            "positive_blocks": pool.positive_blocks,
            "factors_removed": pool.factors_removed,
            "analysis_ns": pool.analysis_ns,
            "cold_scan_ns": pool.cold_scan_ns,
            "plan_prebuild_ns": pool.plan_prebuild_ns,
            "plan_filter_ns": pool.plan_filter_ns,
            "plan_filter_count_ns": pool.plan_filter_count_ns,
            "plan_filter_fill_ns": pool.plan_filter_fill_ns,
            "plan_filter_source_values": pool.plan_filter_source_values,
            "leaf_product_ns": pool.leaf_product_ns,
            "superblock_product_ns": pool.superblock_product_ns,
            "filtered_prime_values": pool.filtered_prime_values,
            "filtered_prime_bytes": pool.filtered_prime_bytes,
            "filtered_plan_count": pool.filtered_plan_count,
            "full_plan_built": pool.full_plan_built,
            "plans_built": pool.plans_built,
            "plan_leaf_blocks": pool.plan_leaf_blocks,
            "plan_superblocks": pool.plan_superblocks,
            "logical_leaf_blocks": pool.logical_leaf_blocks,
            "resident_leaf_blocks": pool.resident_leaf_blocks,
            "dynamic_leaf_products_built": pool.dynamic_leaf_products_built,
            "dynamic_leaf_prime_values": pool.dynamic_leaf_prime_values,
            "dynamic_leaf_product_ns": pool.dynamic_leaf_product_ns,
            "slowest": [
                {
                    "seconds": s,
                    "prime": p_val,
                    "exp": e,
                    "residual_bits": bits,
                    "exact": exact,
                }
                for s, p_val, e, bits, exact in pool.sorted_slowest()
            ],
        },
        "ratio_upper_calls": perf.ratio_upper_calls,
        "ratio_upper_ns": perf.ratio_upper_ns,
        "fermat_debt_calls": perf.fermat_debt_calls,
        "fermat_debt_ns": perf.fermat_debt_ns,
        "q3_prepool_shadow_hits": perf.q3_prepool_shadow_hits,
        "q3_prepool_shadow_mismatches": perf.q3_prepool_shadow_mismatches,
        "q3_prepool_shadow_exact": perf.q3_prepool_shadow_exact,
        "q3_prepool_shadow_outside": perf.q3_prepool_shadow_outside,
        "q3_prepool_prunes": perf.q3_prepool_prunes,
        "q3_prepool_prunes_by_exp": list(perf.q3_prepool_prunes_by_exp),
        "domain_ratio_would_prune": perf.domain_ratio_would_prune,
        "memory_phases": perf.memory_phases,
    }

    with (run_dir / "performance.json").open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


# ── attractor helper ──────────────────────────────────────────


def _pending_source_lines(s, _sig_factors, max_prime) -> List[str]:
    """Build source/closability details for frequent pending primes."""
    lines: List[str] = []
    w = lines.append

    pending_primes = dict(s.pending_prime_frequency.most_common(10))

    if not pending_primes:
        return lines

    sources: Dict[int, list] = {}
    for q in pending_primes:
        sources[q] = []
        for (r, a), factors in _sig_factors.items():
            if q in factors:
                sources[q].append((r, a))

    w("\n## Frequent pending-prime source & closability")
    w(f"  {'pending prime':>14}  {'freq':>10}  {'in pool':>8}  "
      f"{'srcs':>5}  {'generated by':>30}")
    for q, count in pending_primes.items():
        in_pool = "YES" if q <= max_prime else "NO"
        src = sources.get(q, [])
        n_src = len(src)
        if not src:
            examples = "(unknown)"
        else:
            examples = ", ".join(f"{r}^{a}" for r, a in src[:3])
            if n_src > 3:
                examples += ", ..."
        w(f"  {q:>14}  {count:>10,}  {in_pool:>8}  "
          f"{n_src:>5}  ← {examples}")

    return lines


# ── master entry point ────────────────────────────────────────


def write_all_reports(
    *,
    run_dir: Path,
    run_id: str,
    git_commit: str,
    git_dirty: bool | None = None,
    status: str,
    started_at: str = "",
    config: dict,
    metrics: RunMetrics,
    elapsed_seconds: float,
    solutions_found: int,
    sampled_peak_rss: int,
    pruning_policy: str = "",
    telemetry_schema_version: int = 0,
    _sig_factors: dict | None = None,
) -> None:
    """Write all output files for a completed run."""
    write_manifest(
        run_dir,
        run_id=run_id,
        git_commit=git_commit,
        git_dirty=git_dirty,
        started_at=started_at,
        config=config,
        pruning_policy=pruning_policy,
        telemetry_schema_version=telemetry_schema_version,
    )
    write_summary(
        run_dir,
        run_id=run_id,
        git_commit=git_commit,
        status=status,
        started_at=started_at,
        max_prime=config["max_prime"],
        max_factors=config["max_factors"],
        max_exp=config["max_exp"],
        target_num=config["target_num"],
        target_den=config["target_den"],
        pool_gcd_mode=config["pool_gcd_mode"],
        pool_fanout=config["pool_fanout"],
        solutions_found=solutions_found,
        elapsed_seconds=elapsed_seconds,
        metrics=metrics,
    )
    write_structure_text(
        run_dir,
        metrics,
        max_prime=config["max_prime"],
        max_factors=config["max_factors"],
        max_exp=config["max_exp"],
        target_num=config["target_num"],
        target_den=config["target_den"],
        require_euler=config["require_euler"],
        propagate=config["propagate"],
        elapsed=elapsed_seconds,
        solutions_found=solutions_found,
        _sig_factors=_sig_factors,
    )
    write_structure_json(run_dir, metrics)
    write_performance_text(
        run_dir,
        metrics,
        elapsed=elapsed_seconds,
        sampled_peak_rss=sampled_peak_rss,
    )
    write_performance_json(
        run_dir,
        metrics,
        elapsed=elapsed_seconds,
        sampled_peak_rss=sampled_peak_rss,
    )
