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

from gmpy2 import mpz

from opn_core import (
    CHECKPOINT_FILE,
    DEPTH_STATS,
    PRUNE_STATS,
    SOLUTIONS_FILE,
    MAX_FACTORS,
    MAX_EXP,
    PROPAGATE,
    factorize,
    power_pa,
    sigma_prime_power,
)
from opn_state import ChainState, DFSState, validate_chain_state


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
    print(f"  σ(N)/N     = {float(st.ratio_num) / float(st.ratio_den):.12f}")
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
    return lhs == 2 * rhs


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
    """Atomically persist search state + solutions to disk."""
    chk = {
        "primes":       state_holder.get("primes", []),
        "max_factors":  state_holder.get("max_factors", MAX_FACTORS),
        "max_exp":      state_holder.get("max_exp", MAX_EXP),
        "heap":         state_holder.get("heap", []),
        "heap_counter": state_holder.get("heap_counter", 0),
        "total_states": state_holder.get("total_states", 0),
        "elapsed":      state_holder.get("elapsed", 0.0),
        "use_heap":     state_holder.get("use_heap", True),
        "solutions":    solutions,
    }
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(chk, f, pickle.HIGHEST_PROTOCOL)
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
        print("将继续使用，但建议删除检查点文件重新开始。")
    return chk


# ── prune telemetry ───────────────────────────────────────────

def display_prune_stats() -> None:
    """Print a summary table of prune reasons and their frequencies."""
    total = sum(PRUNE_STATS.values())
    if total == 0:
        return
    print("\nPrune statistics:")
    for k, v in PRUNE_STATS.most_common():
        pct = 100.0 * v / total
        print(f"  {k:<12} {v:>12,}  ({pct:5.1f}%)")


def display_depth_histogram() -> None:
    """Print a histogram of successful assign depth distribution."""
    if not DEPTH_STATS:
        return
    print("\nDepth histogram (successful assign):")
    for d in sorted(DEPTH_STATS):
        bar = "#" * min(int(DEPTH_STATS[d] / max(DEPTH_STATS.values()) * 40), 40)
        print(f"  depth {d:>2}: {DEPTH_STATS[d]:>12,}  {bar}")


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

    required_keys = ["primes", "max_factors", "max_exp", "heap", "total_states",
                     "elapsed", "use_heap"]
    for k in required_keys:
        if k not in chk:
            issues.append(f"missing key: {k}")

    if issues:
        return issues  # structural damage, stop early

    # primes list vs current MAX_PRIME
    if chk["primes"] and chk["primes"][-1] > MAX_EXP * 500:
        issues.append("prime list looks mismatched (check MAX_PRIME)")

    # heap counter consistency
    heap = chk.get("heap", [])
    heap_counter = chk.get("heap_counter", 0)
    if len(heap) > 0 and heap_counter < len(heap):
        issues.append(f"heap_counter ({heap_counter}) < heap length ({len(heap)})")

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
