"""
opn_io — display, checkpoint persistence, and solution-file output.

Provides human-readable candidate display (including factor-chain
trace for true OPN candidates), atomic pickle-based checkpoint
save/load, and plain-text solution summaries.
"""

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
    SOLUTIONS_FILE,
    MAX_FACTORS,
    MAX_EXP,
    factorize,
    power_pa,
    sigma_prime_power,
)
from opn_state import ChainState, DFSState


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
    """Return saved state dict, or ``None`` if no checkpoint exists."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"警告: 检查点损坏 ({e})")
        return None


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
