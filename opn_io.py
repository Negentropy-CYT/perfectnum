"""
opn_io — display, checkpoint persistence, and solution-file output.

Provides human-readable candidate display (including factor-chain
trace for true OPN candidates), atomic pickle-based checkpoint
save/load, and plain-text solution summaries.
"""

import json
import math
import os
import pickle
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from gmpy2 import is_prime, mpz

from opn_core import (
    CASCADE_DEPTH_HIST,
    CHECKPOINT_FILE,
    CLONE_PAYLOAD,
    CLONE_STATS,
    CONTRADICTION_ATTR,
    DEPTH_FACTOR_MAP,
    DEPTH_STATS,
    EXCLUDE_EXP_4,
    EXP4_FILTER_HITS,
    HEADROOM_BY_FACTOR,
    MAX_PRIME,
    OBLIGATION_SIGS,
    OPN_MODE,
    PENDING_SIZE_HIST,
    PROPAGATE,
    PROPAGATION_EDGES,
    PROPAGATION_EXP_EDGES,
    OUTSIDE_WINDOW_SOURCE,
    PENDING_PRIME_FREQ,
    SIGMA_MAP_STATS,
    SIGMA_MISS_TIMES,
    SIGMA_POOL_STATS,
    ANALYZER_SLOWEST,
    OUTSIDE_POOL_SOURCES,
    WINDOW_KNOWN_HITS,
    PRUNE_STATS,
    RATIO_HEADROOM,
    SEARCH_MODE,
    TELEMETRY_FILE,
    _SIG_FACTORS,
    SOLUTIONS_FILE,
    MAX_FACTORS,
    MAX_EXP,
    valid_euler_exponents,
    valid_even_exponents,
    factorize,
    power_pa,
    sigma_prime_power,
)
from opn_state import ChainState, DFSState, validate_chain_state


CHECKPOINT_FORMAT_VERSION = 3  # bumped: pseudo-completeness fix + capacity bound


def _search_mode_fingerprint() -> dict:
    return {
        "target_num": SEARCH_MODE.target_num,
        "target_den": SEARCH_MODE.target_den,
        "require_euler": SEARCH_MODE.require_euler,
        "forced_primes": sorted(SEARCH_MODE.forced_primes.items()),
        "excluded_primes": sorted(SEARCH_MODE.excluded_primes),
    }


# ── display ───────────────────────────────────────────────────
def display_solution(st, sol_num: int, elapsed: float) -> None:
    """Print a single candidate (true OPN or pseudo) to stdout."""
    if st.pseudo:
        _display_pseudo(st, sol_num, elapsed)
    else:
        _display_true_opn(st, sol_num, elapsed)

    # factor list
    print("\n  Factors:")
    req_v = getattr(st, 'required_v', {})
    cur_v = getattr(st, 'current_v', {})
    for p, a in sorted(st.assigned.items()):
        tag = " (Euler)" if p == st.euler_prime else ""
        req = req_v.get(p, "")
        cur = cur_v.get(p, "")
        bal = f"  [req={req}, cur={cur}]" if req != "" or cur != "" else ""
        print(f"    {p}^{a}{tag}{bal}")

    # factor-chain trace (true OPN only)
    if not st.pseudo and st.euler_prime:
        _print_factor_chain(st)


def _display_pseudo(st, sol_num: int, elapsed: float) -> None:
    """Print a pseudo-OPN candidate with its composite r-factor."""
    denom = 2 * st.ratio_den - st.ratio_num
    r = st.ratio_num // denom
    n_val = mpz(r)
    for p, a in st.assigned.items():
        n_val *= mpz(power_pa(p, a))
    r_facs = factorize(int(r))
    r_str = " × ".join(f"{q}^{e}" for q, e in r_facs)

    print(f"\n{'=' * 60}")
    print(f"*** Pseudo-OPN Candidate  #{sol_num} ***")
    print(f"  N              = {n_val}")
    print(f"  log10(N)       = {math.log10(int(n_val)):.1f}")
    print(f"  digits         = {len(str(n_val))}")
    print(f"  |factors|      = {len(st.assigned)} + r")
    print(f"  r (composite)  = {r}  =  {r_str}")
    print(f"  r ≡ 1 mod 4    = {r % 4 == 1}")
    res = getattr(st, 'resonance', 0.0)
    print(f"  resonance      = {res:+.2f}")
    print(f"  elapsed        = {elapsed:.1f}s")


def _display_true_opn(st, sol_num: int, elapsed: float) -> None:
    """Print a true OPN candidate with Euler-prime verification."""
    n_val = mpz(1)
    for p, a in st.assigned.items():
        n_val *= mpz(power_pa(p, a))
    print(f"\n{'=' * 60}")
    print(f"*** OPN Candidate  #{sol_num} ***")
    print(f"  N          = {n_val}")
    print(f"  log10(N)   = {math.log10(int(n_val)):.1f}")
    print(f"  digits     = {len(str(n_val))}")
    print(f"  |factors|  = {len(st.assigned)}")
    print(f"  Euler      = {st.euler_prime}")
    print(f"  σ(N)/N     = {float(st.ratio_num / st.ratio_den):.12f}")
    print(f"  verified   = {_verify(st)}")
    res = getattr(st, 'resonance', 0.0)
    print(f"  resonance  = {res:+.2f}")
    print(f"  elapsed    = {elapsed:.1f}s")


def _verify(st) -> bool:
    """Recompute σ(N) from scratch to confirm σ(N) == 2N."""
    lhs = mpz(1)
    rhs = mpz(1)
    for p, a in st.assigned.items():
        lhs *= sigma_prime_power(p, a)
        rhs *= mpz(power_pa(p, a))
    return lhs * SEARCH_MODE.target_den == SEARCH_MODE.target_num * rhs


def _print_factor_chain(st) -> None:
    """Trace σ-propagation from the Euler prime outward (BFS)."""
    print(f"\n  Factor chain (from Euler prime {st.euler_prime}):")
    seen: set[int] = set()
    todo: Deque[Tuple[int, int]] = deque(
        [(st.euler_prime, st.assigned[st.euler_prime])]
    )
    while todo:
        p, a = todo.popleft()
        if p in seen:
            continue
        seen.add(p)
        sig = int(sigma_prime_power(p, a))
        facs = factorize(sig)
        fac_str = " × ".join(f"{q}^{e}" for q, e in facs if q != 2)
        print(f"    σ({p}^{a}) = {sig} = {fac_str}")
        for q, e in facs:
            if q == 2:
                continue
            if q not in seen and q in st.assigned:
                todo.append((q, st.assigned[q]))


# ── checkpoint persistence ────────────────────────────────────
def save_checkpoint(state_holder: dict, solutions: list) -> None:
    """Atomically persist search state + solutions + telemetry to disk."""
    chk = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "search_mode":   _search_mode_fingerprint(),
        "primes":       state_holder.get("primes", []),
        "max_factors":  state_holder.get("max_factors", MAX_FACTORS),
        "max_exp":      state_holder.get("max_exp", MAX_EXP),
        "heap":         state_holder.get("heap", []),
        "heap_counter": state_holder.get("heap_counter", 0),
        "total_states": state_holder.get("total_states", 0),
        "elapsed":      state_holder.get("elapsed", 0.0),
        "use_heap":     state_holder.get("use_heap", True),
        "snapshot_id":  state_holder.get("snapshot_id", 0),
        "solutions":    solutions,
        "prune_stats":  dict(PRUNE_STATS),
        "depth_stats":  dict(DEPTH_STATS),
        "clone_stats":  dict(CLONE_STATS),
        "pending_size": dict(PENDING_SIZE_HIST),
        "cascade_depth":dict(CASCADE_DEPTH_HIST),
        "prop_edges":   [(k, v) for k, v in PROPAGATION_EDGES.items()],
        "clone_payload":dict(CLONE_PAYLOAD),
        "ratio_headroom":dict(RATIO_HEADROOM),
        "depth_factor":  [(k, v) for k, v in DEPTH_FACTOR_MAP.items()],
        "headroom_by_f": [(k, v) for k, v in HEADROOM_BY_FACTOR.items()],
        "obligation_sigs":[(k, v) for k, v in OBLIGATION_SIGS.items()],
        "exp4_filter":   dict(EXP4_FILTER_HITS),
        "contradiction_attr": dict(CONTRADICTION_ATTR),
    }
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(chk, f, pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CHECKPOINT_FILE)


def load_checkpoint() -> Optional[dict]:
    """Return saved state dict, or ``None`` if no checkpoint exists.

    Validates internal consistency after deserialisation and reports any
    issues found (silent corruption guard for long-running searches).
    """
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            chk = pickle.load(f)
    except Exception as e:
        print(f"警告: 检查点损坏 ({e})")
        return None

    issues = validate_checkpoint(chk)
    if issues:
        print("警告: 检查点一致性检查发现问题:")
        for issue in issues:
            print(f"  - {issue}")
        print("为保证搜索完备性，本次不会恢复该检查点。文件保持不变。")
        return None

    # restore telemetry counters so stats accumulate across sessions
    if chk.get("prune_stats"):
        PRUNE_STATS.update(chk["prune_stats"])
    if chk.get("depth_stats"):
        DEPTH_STATS.update({int(k): v for k, v in chk["depth_stats"].items()})
    if chk.get("clone_stats"):
        CLONE_STATS.update(chk["clone_stats"])
    if chk.get("contradiction_attr"):
        CONTRADICTION_ATTR.update(
            {(int(k[0]), k[1]): v for k, v in chk["contradiction_attr"].items()}
        )
    if chk.get("pending_size"):
        PENDING_SIZE_HIST.update({int(k): v for k, v in chk["pending_size"].items()})
    if chk.get("cascade_depth"):
        CASCADE_DEPTH_HIST.update({int(k): v for k, v in chk["cascade_depth"].items()})
    if chk.get("prop_edges"):
        PROPAGATION_EDGES.update({(int(k[0]), int(k[1])): v for k, v in chk["prop_edges"]})
    if chk.get("clone_payload"):
        CLONE_PAYLOAD.update({int(k): v for k, v in chk["clone_payload"].items()})
    if chk.get("ratio_headroom"):
        RATIO_HEADROOM.update(chk["ratio_headroom"])
    if chk.get("depth_factor"):
        DEPTH_FACTOR_MAP.update({(int(k[0]), int(k[1])): v for k, v in chk["depth_factor"]})
    if chk.get("headroom_by_f"):
        HEADROOM_BY_FACTOR.update({(int(k[0]), k[1]): v for k, v in chk["headroom_by_f"]})
    if chk.get("obligation_sigs"):
        OBLIGATION_SIGS.update(
            {(frozenset(k[0]), int(k[1]), int(k[2])): v for k, v in chk["obligation_sigs"]})
    if chk.get("exp4_filter"):
        EXP4_FILTER_HITS.update({int(k): v for k, v in chk["exp4_filter"].items()})

    return chk


# ── telemetry report (file output) ────────────────────────────

def write_telemetry_report(elapsed: float, solutions_found: int) -> None:
    """Write comprehensive telemetry to TELEMETRY_FILE."""
    lines: List[str] = []
    w = lines.append

    euler_exp = valid_euler_exponents(1, MAX_EXP)
    even_exp = valid_even_exponents(2, MAX_EXP)
    mode = f"target={SEARCH_MODE.target_num}/{SEARCH_MODE.target_den}"
    mode += " (chain)" if PROPAGATE else " (DFS)"
    if not SEARCH_MODE.require_euler:
        mode += " no-euler"
    w(f"# P={MAX_PRIME}  f≤{MAX_FACTORS}  exp={MAX_EXP}  "
      f"euler={euler_exp}  even={even_exp}")
    w(f"# {elapsed:.1f}s  |  {solutions_found} solutions  |  {mode}\n")

    # ── prune stats ──
    pr_total = sum(PRUNE_STATS.values())
    cl_total = CLONE_STATS.get("total", 0)
    saved     = CLONE_STATS.get("saved", 0)
    attempted = cl_total + saved  # branches where a clone was considered
    if pr_total:
        w("\n## Prune statistics")
        w(f"{'reason':>14}  {'count':>10}  {'%prune':>7}  {'‰attempt':>8}")
        for k, v in PRUNE_STATS.most_common():
            pct = 100.0 * v / pr_total
            rate = 1000.0 * v / attempted if attempted else 0.0
            w(f"{k:>14}  {v:>10,}  {pct:>6.1f}%  {rate:>7.1f}‰")

    # ── clone effectiveness ──
    total = CLONE_STATS.get("total", 0)
    saved = CLONE_STATS.get("saved", 0)
    attempted = total + saved
    if attempted:
        productive = CLONE_STATS.get("productive", 0)
        wasted     = CLONE_STATS.get("wasted", 0)
        overhead   = total - productive - wasted
        avoid_rate = 100.0 * saved / attempted
        w("\n## Clone effectiveness")
        w(f"  attempted branches  {attempted:>10,}")
        w(f"  actual clones       {total:>10,}")
        w(f"  avoided (pre-clone) {saved:>10,}")
        w(f"  avoidance rate      {avoid_rate:>10.1f}%")
        w(f"    productive        {productive:>10,}  ({100.0*productive/total:5.1f}% of actual)")
        w(f"    wasted (post-cln) {wasted:>10,}  ({100.0*wasted/total:5.1f}% of actual)")
        w(f"    overhead (other)  {overhead:>10,}  ({100.0*overhead/total:5.1f}% of actual)")

    # ── depth histogram ──
    if DEPTH_STATS:
        dt = sum(DEPTH_STATS.values())
        w("\n## Depth histogram")
        for d in sorted(DEPTH_STATS):
            pct = 100.0 * DEPTH_STATS[d] / dt
            w(f"  depth {d:>2}: {DEPTH_STATS[d]:>12,}  ({pct:4.1f}%)")

    # ── clone payload ──
    if CLONE_PAYLOAD:
        ct = sum(CLONE_PAYLOAD.values())
        w("\n## Clone payload (|assigned|)")
        for s in sorted(CLONE_PAYLOAD):
            w(f"  |f|={s:>2}:   {CLONE_PAYLOAD[s]:>12,}  ({100.0*CLONE_PAYLOAD[s]/ct:5.1f}%)")

    # ── ratio headroom ──
    if RATIO_HEADROOM:
        rt = sum(RATIO_HEADROOM.values())
        order = ["<1e-6","1e-6-1e-5","1e-5-1e-4","1e-4-1e-3","1e-3-1e-2",">1e-2"]
        w("\n## Ratio headroom")
        for b in order:
            v = RATIO_HEADROOM.get(b, 0)
            if v:
                w(f"  {b:>12}  {v:>12,}  ({100.0*v/rt:5.1f}%)")

    # ── headroom by |f| ──
    if HEADROOM_BY_FACTOR:
        w("\n## Headroom by |f|")
        header = f"  {'|f|':>4}"
        for b in order:
            header += f"  {b:>10}"
        w(header)
        f_levels = sorted(set(f for f, _ in HEADROOM_BY_FACTOR))
        for nf in f_levels:
            row = f"  {nf:>4}"
            for b in order:
                v = HEADROOM_BY_FACTOR.get((nf, b), 0)
                row += f"  {v:>10,}"
            w(row)

    # ── obligation recurrence ──
    if OBLIGATION_SIGS:
        ot = sum(OBLIGATION_SIGS.values())
        uniq = len(OBLIGATION_SIGS)
        top10 = sum(v for _, v in OBLIGATION_SIGS.most_common(10))
        w("\n## Obligation recurrence")
        w(f"  unique/total: {uniq:,} / {ot:,}  ({100.0*uniq/ot:.2f}%)")
        w(f"  top-10 coverage: {top10:,}  ({100.0*top10/ot:.1f}%)")
        w("  Top-10 signatures:")
        for (pending, nf, coarse), count in OBLIGATION_SIGS.most_common(10):
            w(f"    pending={set(pending)} |f|={nf} h~=1e-{coarse}  x{count:,}")

    # ── contradiction attribution ──
    if CONTRADICTION_ATTR:
        w("\n## Contradiction attribution (top-15)")
        for (q, reason), count in CONTRADICTION_ATTR.most_common(15):
            w(f"  ({q:>4}, {reason:<14}) {count:>10,}")

    # ── propagation edges ──
    if PROPAGATION_EDGES:
        w("\n## Propagation edges (top-10)")
        for (p, q), count in PROPAGATION_EDGES.most_common(10):
            w(f"  {p:>4} → {q:<8}  {count:>10,}")

    if PROPAGATION_EXP_EDGES:
        w("\n## Propagation edges by exponent (top-10)")
        for (p, exp, q), count in PROPAGATION_EXP_EDGES.most_common(10):
            w(f"  {p:>4}^{exp} → {q:<8}  {count:>10,}")

    # ── sigma map cache ──
    if SIGMA_MAP_STATS:
        hits = SIGMA_MAP_STATS.get("hits", 0)
        misses = SIGMA_MAP_STATS.get("misses", 0)
        total = hits + misses
        factor_s = SIGMA_MAP_STATS.get("factor_seconds", 0.0)
        w("\n## σ-map cache")
        w(f"  hits:     {hits:>12,}")
        w(f"  misses:   {misses:>12,}")
        if total:
            w(f"  hit rate: {100.0*hits/total:>11.1f}%")
        w(f"  factor s: {factor_s:>12.1f}")

    # ── pool analysis stats ──
    if SIGMA_POOL_STATS:
        w("\n## σ pool analysis")
        for k in ["hits", "misses", "exact", "outside_certificates",
                   "exact_from_global_cache", "outside_from_global_cache",
                   "blocks_tested", "positive_blocks", "pool_factors_removed"]:
            v = SIGMA_POOL_STATS.get(k, 0)
            if v:
                w(f"  {k:<28} {v:>12,}")
        ns_val = SIGMA_POOL_STATS.get("analysis_ns", 0)
        if ns_val:
            w(f"  analysis_seconds        {ns_val*1e-9:>12.1f}")

    if OUTSIDE_POOL_SOURCES:
        w("\n## Outside-pool residual sources (top-10)")
        for (p, exp, bits), count in OUTSIDE_POOL_SOURCES.most_common(10):
            w(f"  {p:>6}^{exp} → residual ({bits} bits)  x{count:>6}")

    if WINDOW_KNOWN_HITS:
        w("\n## Known outside-pool cache hits (top-10)")
        w(f"  {'p^exp':>10}  {'cached rejects':>15}")
        for (p, exp), count in WINDOW_KNOWN_HITS.most_common(10):
            w(f"  {p:>6}^{exp:<3}  {count:>15,}")

    if ANALYZER_SLOWEST:
        w("\n## Slowest pool analyses (top-15)")
        w(f"  {'p':>6}  {'exp':>3}  {'residual bits':>13}  {'exact':>5}  {'seconds':>8}")
        for elapsed, p, a, bits, exact in ANALYZER_SLOWEST:
            w(f"  {p:>6}  {a:>3}  {bits:>13}  {str(exact):>5}  {elapsed:>8.3f}")

    # ── slowest σ factorisations ──
    if SIGMA_MISS_TIMES:
        slowest = sorted(SIGMA_MISS_TIMES, key=lambda x: -x[3])[:15]
        w("\n## Slowest σ factorisations (top-15)")
        w(f"  {'p':>6}  {'exp':>3}  {'σ bits':>8}  {'seconds':>8}  {'max odd q':>10}")
        for p, a, bits, sec, maxq in slowest:
            w(f"  {p:>6}  {a:>3}  {bits:>8}  {sec:>8.3f}  {maxq:>10}")

    # ── depth × |f| ──
    if DEPTH_FACTOR_MAP:
        w("\n## Depth × |f| (top-15)")
        for (d, nf), count in DEPTH_FACTOR_MAP.most_common(15):
            w(f"  depth={d:>3}  |f|={nf}  {count:>12,}")

    # ── pending-prime frequency (global) ──
    if PENDING_PRIME_FREQ:
        w("\n## Pending-prime frequency (top-15)")
        for q, count in PENDING_PRIME_FREQ.most_common(15):
            w(f"  {q:>12}  {count:>10,}")

    # ── outside-window sources ──
    if OUTSIDE_WINDOW_SOURCE:
        w("\n## Outside-window sources (top-10)")
        for (p, exp, q), count in OUTSIDE_WINDOW_SOURCE.most_common(10):
            w(f"  {p:>4}^{exp} → {q:<12}  {count:>10,}")

    # ── exp4 filter verification ──
    if EXP4_FILTER_HITS:
        w("\n## EXP4 filter verification")
        w(f"  primes filtered: {len(EXCLUDE_EXP_4)}")
        w(f"  total a=4 branches skipped: {sum(EXP4_FILTER_HITS.values()):,}")
        w("  per-prime hits (top-10):")
        for p, count in EXP4_FILTER_HITS.most_common(10):
            facs = _SIG_FACTORS.get((p, 4), set())
            w(f"    {p:>4}^4  x{count:>10,}  σ factors → {facs}")

    # ── attractor resolvability ──
    if OBLIGATION_SIGS:
        _write_attractor_resolvability(w)

    w("")
    with open(TELEMETRY_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_attractor_resolvability(w) -> None:
    """For each top attractor prime, identify its σ(r^a) source and check
    whether it can be closed within the current search window (q ≤ MAX_PRIME).
    """
    attractor_primes: Dict[int, int] = {}
    for (pending, nf, coarse), count in OBLIGATION_SIGS.most_common(10):
        for q in pending:
            attractor_primes[q] = attractor_primes.get(q, 0) + count

    if not attractor_primes:
        return

    # reverse index: for each q, which (r, a) pairs have q | σ(r^a)
    sources: Dict[int, list] = {}
    for q in attractor_primes:
        sources[q] = []
        for (r, a), factors in _SIG_FACTORS.items():
            if q in factors:
                sources[q].append((r, a))

    w("\n## Attractor source & closability")
    w(f"  {'obligation':>14}  {'freq':>10}  {'in pool':>8}  {'srcs':>5}  {'generated by':>30}")
    for q, count in sorted(attractor_primes.items(), key=lambda x: -x[1]):
        in_pool = "YES" if q <= MAX_PRIME else "NO"
        src = sources.get(q, [])
        n_src = len(src)
        if not src:
            examples = "(unknown)"
        else:
            examples = ", ".join(f"{r}^{a}" for r, a in src[:3])
            if n_src > 3:
                examples += f", ..."
        w(f"  {q:>14}  {count:>10,}  {in_pool:>8}  {n_src:>5}  ← {examples}")


def display_telemetry_brief() -> None:
    """Print a one-line telemetry summary to stdout (details → file)."""
    pr_total = sum(PRUNE_STATS.values())
    cl_total = CLONE_STATS.get("total", 0)
    productive = CLONE_STATS.get("productive", 0)
    ot = sum(OBLIGATION_SIGS.values()) if OBLIGATION_SIGS else 0
    uniq = len(OBLIGATION_SIGS) if OBLIGATION_SIGS else 0

    parts = [f"states={cl_total:,}", f"productive={productive}"]
    if pr_total:
        top = PRUNE_STATS.most_common(1)
        if top:
            parts.append(f"prune=({top[0][0]} {100.0*top[0][1]/pr_total:.0f}%)")
    if ot:
        recur = 100.0 * uniq / ot if ot else 0
        parts.append(f"recur={recur:.1f}%")
    sys.stderr.write(f"\n[telemetry] {' | '.join(parts)}\n")
    sys.stderr.write(f"  full report → {TELEMETRY_FILE}\n")


# ── factor graph export ───────────────────────────────────────

def export_factor_graph(st, path: str = "factor_graph") -> None:
    """Export the σ-factor dependency graph for a candidate state.

    Writes two files:
      - ``{path}.dot`` — Graphviz DOT (human viewing via ``dot -Tpng``)
      - ``{path}.json`` — machine-readable edge list with cycles
    """
    edges: List[dict] = []
    for p, exp in st.assigned.items():
        sig = int(sigma_prime_power(p, exp))
        for q, _ in factorize(sig):
            if q == 2:
                continue
            edges.append({"from": p, "to": q})

    # ── DOT ──
    with open(f"{path}.dot", "w") as f:
        f.write("digraph OPN {\n")
        f.write('  rankdir=LR;\n')
        f.write('  node [shape=circle];\n')
        seen_pairs = set()
        for e in edges:
            pair = (e["from"], e["to"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            f.write(f'  "{e["from"]}" -> "{e["to"]}";\n')
        f.write("}\n")

    # ── JSON ──
    # detect cycles for analysis
    adj: Dict[int, list] = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])
    cycles = _find_cycles(adj)

    with open(f"{path}.json", "w") as f:
        json.dump({
            "edges": edges,
            "cycles": cycles,
            "assigned": {str(p): exp for p, exp in st.assigned.items()},
            "euler_prime": st.euler_prime,
        }, f, indent=2)

    print(f"Factor graph exported: {path}.dot, {path}.json")


def _find_cycles(adj: Dict[int, list]) -> list:
    """Return list of simple cycles in a directed graph (DFS-based)."""
    cycles: list = []
    visited: set = set()
    stack: list = []

    def dfs(node: int):
        if node in stack:
            cycle_start = stack.index(node)
            cycles.append(stack[cycle_start:])
            return
        if node in visited:
            return
        visited.add(node)
        stack.append(node)
        for nb in adj.get(node, []):
            dfs(nb)
        stack.pop()

    for start in adj:
        if start not in visited:
            dfs(start)
    return cycles


# ── checkpoint validation ─────────────────────────────────────

def validate_checkpoint(chk: dict) -> List[str]:
    """Validate a deserialised checkpoint dict.  Returns list of issues (empty = OK)."""
    issues: List[str] = []

    required_keys = [
        "format_version", "search_mode", "primes", "max_factors", "max_exp",
        "heap", "total_states", "elapsed", "use_heap",
    ]
    for k in required_keys:
        if k not in chk:
            issues.append(f"missing key: {k}")

    if issues:
        return issues  # structural damage, stop early

    if chk["format_version"] != CHECKPOINT_FORMAT_VERSION:
        issues.append(
            f"unsupported checkpoint format: {chk['format_version']} "
            f"(expected {CHECKPOINT_FORMAT_VERSION})"
        )
    if chk["search_mode"] != _search_mode_fingerprint():
        issues.append("search mode differs from the current target/Euler rules")
    if bool(chk["use_heap"]) != bool(PROPAGATE):
        issues.append("PROPAGATE mode differs from the saved search strategy")

    primes = chk["primes"]
    if not primes:
        issues.append("prime list is empty")
    elif primes != sorted(set(primes)):
        issues.append("prime list is not strictly increasing and unique")
    elif any(p < 3 or p % 2 == 0 or not is_prime(p) for p in primes):
        issues.append("prime list contains a non-odd-prime candidate")
    if chk["max_factors"] < 1 or chk["max_exp"] < 1:
        issues.append("max_factors and max_exp must be positive")

    # heap counter consistency
    heap = chk.get("heap", [])
    heap_counter = chk.get("heap_counter", 0)
    if chk.get("use_heap", False) and heap:
        tie_ids = []
        valid_entries = True
        for entry in heap:
            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                valid_entries = False
                break
            tie_ids.append(entry[1])
        if not valid_entries:
            issues.append("heap contains a malformed priority entry")
        elif any(not isinstance(tie_id, int) for tie_id in tie_ids):
            issues.append("heap contains a non-integer tie-break identifier")
        else:
            if len(tie_ids) != len(set(tie_ids)):
                issues.append("heap contains duplicate tie-break identifiers")
            if heap_counter <= max(tie_ids):
                issues.append(
                    "heap_counter is not greater than every saved identifier"
                )
            for child in range(1, len(heap)):
                parent = (child - 1) // 2
                if tuple(heap[parent][:2]) > tuple(heap[child][:2]):
                    issues.append(
                        "saved priority queue does not satisfy heap order"
                    )
                    break
    elif not chk.get("use_heap", False) and heap_counter < 0:
        issues.append("stack counter must be non-negative")

    # validate ChainState invariants (only in factor-chain mode)
    if chk.get("use_heap", False):
        for entry in heap:
            if isinstance(entry, (list, tuple)):
                st = entry[2] if len(entry) >= 3 else entry[0]
            else:
                st = entry
            if isinstance(st, ChainState):
                if not validate_chain_state(st):
                    issues.append("ChainState invariant violated in heap")

    return issues


# ── solutions file ────────────────────────────────────────────
def save_solutions_txt(solutions: list) -> None:
    """Write human-readable solution summary to disk."""
    if not solutions:
        return
    with open(SOLUTIONS_FILE, "w", encoding="utf-8") as f:
        f.write("# Odd Perfect Number Search Results\n")
        true_count   = sum(1 for s in solutions if not s[2])
        pseudo_count = sum(1 for s in solutions if s[2])
        f.write(f"# True OPN: {true_count}  |  Pseudo: {pseudo_count}\n\n")
        for i, (factors, euler, pseudo) in enumerate(solutions, 1):
            tag = "PSEUDO" if pseudo else "OPN"
            f.write(f"[{tag}] #{i}:\n")
            f.write(f"  Euler prime: {euler}\n")
            f.write(f"  Factors: {factors}\n\n")
