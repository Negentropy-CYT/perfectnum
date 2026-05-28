"""
Odd Perfect Number Search — Entry Point.

A constraint-propagation factor-chain engine for enumerating
candidates of the form  N = q^{4k+1} × ∏ p_i^{2a_i}.

Usage
-----
    python opn_main.py            # pseudo-solution search (default)
    # edit PROPAGATE = True       # true-OPN factor-chain search
    # edit MAX_PRIME / MAX_EXP    # adjust search scope

Modules
-------
    opn_core      — arithmetic engine (primes, factorisation, caches)
    opn_state     — State dataclass & constraint propagation
    opn_search    — search engine
    opn_io        — display, checkpoint, file I/O
"""

import os
import sys
import time
from typing import List

from opn_core import (
    CHECKPOINT_FILE,
    MAX_PRIME,
    MAX_FACTORS,
    MAX_EXP,
    PROPAGATE,
    generate_odd_primes,
    valid_euler_exponents,
    valid_even_exponents,
)
from opn_io import (
    display_depth_histogram,
    display_prune_stats,
    display_solution,
    export_factor_graph,
    load_checkpoint,
    save_checkpoint,
    save_solutions_txt,
)
from opn_search import search_opn


# ── main ──────────────────────────────────────────────────────
def main() -> None:
    """Parse checkpoint (if any), run search, display results."""
    chk = load_checkpoint()
    prev_solutions: list = []

    if chk is not None:
        print("=" * 60)
        print("发现已有检查点，将从中断处继续 ...")
        print(f"  已完成状态: {chk['total_states']:,}")
        print(f"  已用时间:   {chk['elapsed']:.1f}s")
        print(f"  已找到解:   {len(chk.get('solutions', []))}")
        print("=" * 60)
        prev_solutions = chk.get("solutions", [])
        primes      = chk["primes"]
        max_factors = chk["max_factors"]
        max_exp     = chk["max_exp"]
        resume_state = {
            "heap":         chk.get("heap", []),
            "heap_counter": chk.get("heap_counter", 0),
            "total_states": chk.get("total_states", 0),
            "elapsed":      chk.get("elapsed", 0.0),
            "use_heap":     chk.get("use_heap", PROPAGATE),
        } if chk.get("heap") else None
    else:
        primes      = generate_odd_primes(MAX_PRIME)
        max_factors = MAX_FACTORS
        max_exp     = MAX_EXP
        resume_state = None

    # header
    mode_str = ("伪解搜索 (独立质数, propagate=False)" if not PROPAGATE
                else "真 OPN 搜索 (因子链约束, propagate=True)")
    print(f"质数范围:    ≤ {primes[-1]}  (共 {len(primes)} 个)")
    print(f"最大因子数:  {max_factors}")
    print(f"最大指数:    {max_exp}")
    print(f"Euler 指数:  {valid_euler_exponents(1, max_exp)}")
    print(f"非 Euler:    {valid_even_exponents(2, max_exp)}")
    print(f"搜索模式:    {mode_str}")
    print("=" * 60)
    print("按 Ctrl+C 安全中断并保存进度\n")

    solutions    = list(prev_solutions)
    state_holder: dict = {}
    t0           = time.time()
    found_true   = 0
    found_pseudo = 0

    # ── progress callback (decoupled from search engine) ──
    def _show_progress(total_states: int, st, elapsed: float) -> None:
        rate = total_states / elapsed if elapsed > 0 else 0
        sys.stdout.write(
            f"\r[Progress] States: {total_states:>12,} | "
            f"Time: {elapsed:>7.1f}s | Rate: {rate:>8.0f}/s | "
            f"|f|={len(st.assigned)} "
            f"ratio={float(st.ratio_num) / float(st.ratio_den):.8f} "
            f"reson={getattr(st, 'resonance', 0.0):+.2f}"
        )
        sys.stdout.flush()

    try:
        for st in search_opn(
            primes, max_factors, max_exp,
            state_holder=state_holder,
            resume_state=resume_state,
            propagate=PROPAGATE,
            progress_callback=_show_progress,
        ):
            if st.pseudo:
                found_pseudo += 1
            else:
                found_true += 1
            solutions.append((dict(st.assigned), st.euler_prime, st.pseudo))
            display_solution(st, len(solutions), time.time() - t0)
            export_factor_graph(st)
            save_checkpoint(state_holder, solutions)
            save_solutions_txt(solutions)

    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在保存 ...")
        save_checkpoint(state_holder, solutions)
        save_solutions_txt(solutions)
        display_prune_stats()
        display_depth_histogram()
        print(
            f"已保存。已完成 {state_holder.get('total_states', 0):,} 个状态"
        )
        print(
            f"已找到 {found_true} 个真解 + {found_pseudo} 个伪解。"
            f"下次运行将从中断处继续。"
        )
        sys.exit(0)

    elapsed = time.time() - t0
    print(
        f"\n搜索完成。总状态: {state_holder.get('total_states', 0):,}, "
        f"耗时: {elapsed:.1f}s"
    )
    display_prune_stats()
    display_depth_histogram()

    if solutions:
        print(f"\n=== 共 {found_true} 个真解 + {found_pseudo} 个伪解 ===")
    else:
        print("\n在搜索范围内无解。")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("已清理检查点文件。")


if __name__ == "__main__":
    main()
