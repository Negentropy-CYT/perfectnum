"""
Odd Perfect Number Search — Entry Point.

A constraint-propagation factor-chain engine for enumerating
candidates of the form  N = q^{4k+1} × ∏ p_i^{2a_i}.

Usage
-----
    python opn_main.py            # run the configured finite search box
    # edit PROPAGATE              # switch spoof/true-OPN mode
    # edit MAX_PRIME / MAX_EXP    # adjust search scope

Modules
-------
    opn_core      — arithmetic engine (primes, factorisation, caches)
    opn_state     — State dataclass & constraint propagation
    opn_search    — search engine
    opn_io        — display, checkpoint, file I/O
"""

import os
import signal
import sys
import time
from typing import List

from opn_core import (
    CHECKPOINT_INTERVAL_SECONDS,
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
    display_solution,
    display_telemetry_brief,
    export_factor_graph,
    load_checkpoint,
    save_checkpoint,
    save_solutions_txt,
    write_telemetry_report,
)
from opn_search import SearchStopped, search_opn


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
        primes      = chk.get("primes", [])
        max_factors = chk.get("max_factors", MAX_FACTORS)
        max_exp     = chk.get("max_exp", MAX_EXP)
        if not primes:
            primes = generate_odd_primes(MAX_PRIME)
        resume_state = {
            "heap":         chk.get("heap", []),
            "heap_counter": chk.get("heap_counter", 0),
            "total_states": chk.get("total_states", 0),
            "elapsed":      chk.get("elapsed", 0.0),
            "use_heap":     chk.get("use_heap", PROPAGATE),
            "snapshot_id":  chk.get("snapshot_id", 0),
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
    print(f"自动检查点:  每 {CHECKPOINT_INTERVAL_SECONDS:g} 秒")
    print("=" * 60)
    print("按一次 Ctrl+C 在当前状态完成后安全保存；再次按下立即中断\n")

    solutions    = list(prev_solutions)
    state_holder: dict = {}
    t0           = time.time()
    found_true   = 0
    found_spoof = 0
    stop_requested = False
    interrupt_count = 0

    # ── progress callback (decoupled from search engine) ──
    def _show_progress(total_states: int, st, elapsed: float) -> None:
        rate = total_states / elapsed if elapsed > 0 else 0
        sys.stdout.write(
            f"\r[Progress] States: {total_states:>12,} | "
            f"Time: {elapsed:>7.1f}s | Rate: {rate:>8.0f}/s | "
            f"|f|={len(st.assigned)} "
            f"ratio={float(st.ratio_num / st.ratio_den):.8f} "
            f"reson={getattr(st, 'resonance', 0.0):+.2f}"
        )
        sys.stdout.flush()

    def _save_stable_boundary(holder: dict, reason: str) -> None:
        if reason in {"initial", "periodic"}:
            save_checkpoint(holder, solutions)

    previous_sigint = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum, frame) -> None:
        nonlocal stop_requested, interrupt_count
        interrupt_count += 1
        if interrupt_count == 1:
            stop_requested = True
            sys.stderr.write(
                "\n收到中断请求；当前状态完成后将精确保存。"
                "再次按 Ctrl+C 可立即中断并回退到最近的自动检查点。\n"
            )
            sys.stderr.flush()
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        for st in search_opn(
            primes, max_factors, max_exp,
            state_holder=state_holder,
            resume_state=resume_state,
            propagate=PROPAGATE,
            progress_callback=_show_progress,
            checkpoint_callback=_save_stable_boundary,
            checkpoint_interval_seconds=CHECKPOINT_INTERVAL_SECONDS,
            stop_requested=lambda: stop_requested,
        ):
            if st.spoof:
                found_spoof += 1
            else:
                found_true += 1
            solutions.append((dict(st.assigned), st.euler_prime, st.spoof))
            display_solution(st, len(solutions), time.time() - t0)
            export_factor_graph(st, path=f"factor_graph_{len(solutions)}")
            # ponytail: chain-mode spoof is non-terminal — the heap at
            # this point lacks st and its successors.  The periodic
            # checkpoint covers the live frontier; skip the snapshot here.
            if not st.spoof or not PROPAGATE:
                save_checkpoint(state_holder, solutions)
            save_solutions_txt(solutions)

    except SearchStopped:
        print("\n\n已到达稳定搜索边界，正在保存 ...")
        save_checkpoint(state_holder, solutions)
        save_solutions_txt(solutions)
        write_telemetry_report(time.time() - t0, found_true + found_spoof)
        display_telemetry_brief()
        print(
            f"已精确保存。已完成 "
            f"{state_holder.get('total_states', 0):,} 个状态；"
            f"前沿还有 {state_holder.get('frontier_size', 0):,} 个状态。"
        )
        return

    except KeyboardInterrupt:
        print("\n\n已立即中断搜索。")
        save_solutions_txt(solutions)
        write_telemetry_report(time.time() - t0, found_true + found_spoof)
        display_telemetry_brief()
        if os.path.exists(CHECKPOINT_FILE):
            print("保留了最近一次完整的原子检查点；恢复时可能重算少量状态。")
        else:
            print("中断发生在首次检查点完成前，本次没有可恢复的完整检查点。")
        return

    finally:
        signal.signal(signal.SIGINT, previous_sigint)

    elapsed = time.time() - t0
    print(
        f"\n搜索完成。总状态: {state_holder.get('total_states', 0):,}, "
        f"耗时: {elapsed:.1f}s"
    )
    write_telemetry_report(elapsed, found_true + found_spoof)
    display_telemetry_brief()

    if solutions:
        print(f"\n=== 共 {found_true} 个真解 + {found_spoof} 个伪解 ===")
    else:
        print("\n在搜索范围内无解。")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("已清理检查点文件。")


if __name__ == "__main__":
    main()
