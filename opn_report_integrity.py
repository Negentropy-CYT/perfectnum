"""Read-only integrity checks for search observability data.

All checks operate on already-collected metrics and output files.
They never modify counters, prune decisions, or search state.

ponytail: read-only checker, no hot-path instrumentation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from opn_metrics import RunMetrics


def check_structure_invariants(
    metrics: RunMetrics,
) -> dict[str, dict[str, Any]]:
    """Verify mathematical invariants within StructureMetrics."""
    s = metrics.structure
    results: dict[str, dict[str, Any]] = {}

    # I1: productive_states == sum(depth_factor_map)
    depth_sum = sum(s.depth_factor_map.values())
    results["productive_depth_total"] = {
        "passed": depth_sum == s.productive_states,
        "expected": s.productive_states,
        "observed": depth_sum,
    }

    # I2: productive_states == sum(headroom_by_factor)
    headroom_sum = sum(s.headroom_by_factor.values())
    results["productive_headroom_total"] = {
        "passed": headroom_sum == s.productive_states,
        "expected": s.productive_states,
        "observed": headroom_sum,
    }

    # I3: propagation_edges(p,q) == sum_exp propagation_exp_edges(p,exp,q)
    derived: Counter[tuple[int, int]] = Counter()
    for (p, exp, q), count in s.propagation_exp_edges.items():
        derived[(p, q)] += count

    mismatch = 0
    all_keys = set(s.propagation_edges.keys()) | set(derived.keys())
    for key in all_keys:
        if s.propagation_edges.get(key, 0) != derived.get(key, 0):
            mismatch += 1

    results["propagation_aggregation"] = {
        "passed": mismatch == 0,
        "mismatch_count": mismatch,
    }

    # I4: sigma_exact == sum(sigma_exact_by_exp)
    exact_sum = sum(s.sigma_exact_by_exp)
    results["sigma_exact_total"] = {
        "passed": exact_sum == s.sigma_exact,
        "expected": s.sigma_exact,
        "observed": exact_sum,
    }

    # I5: sigma_outside == sum(sigma_outside_by_exp)
    outside_sum = sum(s.sigma_outside_by_exp)
    results["sigma_outside_total"] = {
        "passed": outside_sum == s.sigma_outside,
        "expected": s.sigma_outside,
        "observed": outside_sum,
    }

    return results


def check_prune_consistency(metrics: RunMetrics) -> dict[str, dict[str, Any]]:
    """I6: sum(prune_reasons) == sum(prune_mechanisms)."""
    s = metrics.structure
    p = metrics.performance
    reason_total = sum(s.prune_reasons.values())
    mechanism_total = sum(p.prune_mechanisms.values())

    return {
        "prune_dimension_consistency": {
            "passed": reason_total == mechanism_total,
            "reason_total": reason_total,
            "mechanism_total": mechanism_total,
        },
    }


def check_gap_summary_invariants(
    abundancy_capture_summary: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """I7, I8: gap state counting invariants from the summary dict."""
    if abundancy_capture_summary is None:
        return {
            "gap_small_gap": {"passed": True, "skipped": True},
            "gap_qualifying": {"passed": True, "skipped": True},
        }

    cfg = abundancy_capture_summary.get("configuration", {})
    if not cfg.get("enabled", False):
        return {
            "gap_small_gap": {"passed": True, "skipped": "capture disabled"},
            "gap_qualifying": {"passed": True, "skipped": "capture disabled"},
        }

    small_gap = int(abundancy_capture_summary.get("small_gap_states_seen", 0))
    qualifying = int(abundancy_capture_summary.get("qualifying_states", 0))
    pending_rej = int(
        abundancy_capture_summary.get("pending_lower_bound_rejections", 0)
    )
    written = int(abundancy_capture_summary.get("records_written", 0))
    dropped = int(abundancy_capture_summary.get("dropped_due_to_limit", 0))

    results: dict[str, dict[str, Any]] = {}

    results["gap_small_gap"] = {
        "passed": small_gap == pending_rej + qualifying,
        "small_gap_states_seen": small_gap,
        "pending_rejected": pending_rej,
        "qualifying": qualifying,
        "computed": pending_rej + qualifying,
    }

    results["gap_qualifying"] = {
        "passed": qualifying == written + dropped,
        "qualifying_states": qualifying,
        "records_written": written,
        "dropped": dropped,
        "computed": written + dropped,
    }

    return results


def check_gap_jsonl(
    run_dir: Path,
    abundancy_capture_summary: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """I9, I10: verify JSONL file content consistency."""
    jsonl_path = run_dir / "abundancy_gap_states.jsonl"
    results: dict[str, dict[str, Any]] = {}

    if not jsonl_path.exists():
        results["jsonl_exists"] = {"passed": True, "skipped": "no JSONL file"}
        return results

    results["jsonl_exists"] = {"passed": True}
    target_num = 2
    target_den = 1
    if abundancy_capture_summary:
        cfg = abundancy_capture_summary.get("configuration", {})
        target_num = cfg.get("target_num", target_num)
        target_den = cfg.get("target_den", target_den)

    line_count = 0
    prev_ordinal = 0
    gap_errors = 0
    ordinal_errors = 0

    try:
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                line_count += 1

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    gap_errors += 1
                    continue

                ordinal = rec.get("productive_ordinal", 0)
                if ordinal <= prev_ordinal:
                    ordinal_errors += 1
                prev_ordinal = ordinal

                # I10: gap == target - ratio
                try:
                    ratio_num = int(rec["ratio_num"])
                    ratio_den = int(rec["ratio_den"])
                    gap_num = int(rec["gap_num"])
                    gap_den = int(rec["gap_den"])
                except (KeyError, ValueError):
                    gap_errors += 1
                    continue

                expected_gap_num = (
                    target_num * ratio_den - target_den * ratio_num
                )
                expected_gap_den = target_den * ratio_den
                if (
                    gap_num != expected_gap_num
                    or gap_den != expected_gap_den
                ):
                    gap_errors += 1
    except OSError:
        results["jsonl_readable"] = {"passed": False, "error": "OS error"}
        return results

    results["jsonl_line_count"] = {
        "passed": True,
        "lines": line_count,
    }
    results["jsonl_ordinal_monotonic"] = {
        "passed": ordinal_errors == 0,
        "errors": ordinal_errors,
    }
    results["jsonl_gap_reconstruction"] = {
        "passed": gap_errors == 0,
        "errors": gap_errors,
    }

    return results


def run_all_checks(
    metrics: RunMetrics,
    *,
    run_dir: Path | None = None,
    abundancy_capture_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all integrity checks. Returns {schema_version, status, checks}."""
    checks: dict[str, dict[str, Any]] = {}

    checks.update(check_structure_invariants(metrics))
    checks.update(check_prune_consistency(metrics))
    checks.update(check_gap_summary_invariants(abundancy_capture_summary))

    if run_dir is not None:
        checks.update(check_gap_jsonl(run_dir, abundancy_capture_summary))

    all_pass = all(c.get("passed", True) for c in checks.values())

    return {
        "schema_version": 1,
        "status": "PASS" if all_pass else "FAIL",
        "checks": checks,
    }


def write_integrity_json(
    run_dir: Path,
    metrics: RunMetrics,
    *,
    abundancy_capture_summary: dict[str, Any] | None = None,
) -> Path:
    """Run checks and write report_integrity.json to run_dir."""
    result = run_all_checks(
        metrics,
        run_dir=run_dir,
        abundancy_capture_summary=abundancy_capture_summary,
    )

    out_path = run_dir / "report_integrity.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return out_path
