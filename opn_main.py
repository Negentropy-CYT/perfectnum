"""
Odd Perfect Number Search — Entry Point.

A constraint-propagation factor-chain engine for enumerating
candidates of the form  N = q^{4k+1} × ∏ p_i^{2a_i}.

Usage
-----
    python opn_main.py            # run the configured finite search box
    # edit PROPAGATE              # switch spoof/Euler-form OPN mode
    # edit MAX_PRIME / MAX_EXP    # adjust search scope
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import opn_core
from opn_core import (
    ABUNDANCY_GAP_MAX_DEN,
    ABUNDANCY_GAP_MAX_NUM,
    ABUNDANCY_GAP_MAX_RECORDS,
    ABUNDANCY_GAP_TEXT_LIMIT,
    CAPTURE_ABUNDANCY_GAP_STATES,
    CHECKPOINT_INTERVAL_SECONDS,
    CHECKPOINT_FILE,
    MAX_PRIME,
    MAX_FACTORS,
    MAX_EXP,
    PROPAGATE,
    POOL_GCD_MODE,
    POOL_PLAN_DISK_CACHE_DIR,
    POOL_PLAN_DISK_CACHE_ENABLED,
    POOL_PLAN_DISK_MIN_FREE_BYTES,
    POOL_PLAN_BUILD_POLICY,
    POOL_SUPERBLOCK_FANOUT,
    DOMAIN_RATIO_MODE,
    PENDING_SELECTION,
    PRUNING_POLICY,
    Q3_PREPOOL_MODE,
    SEARCH_MODE,
    TELEMETRY_SCHEMA_VERSION,
    SIGMA_DATABASE_ENABLED,
    SIGMA_DATABASE_FILE,
    prime_pool_typecode,
    valid_euler_exponents,
    valid_even_exponents,
)
from opn_prime_pool import open_or_extend_prime_pool
from opn_abundancy_capture import (
    AbundancyCaptureConfig,
    AbundancyGapRecorder,
)
from opn_io import (
    display_solution,
    export_factor_graph,
    load_checkpoint,
    save_checkpoint,
    save_solutions_txt,
)
from opn_metrics import RunMetrics
from opn_reports import prepare_run_directory, write_all_reports
from opn_runtime import RuntimeSampler
from opn_search import SearchStopped, search_opn


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
        )
        return bool(output.strip())
    except Exception:
        return None


def _make_run_id(max_prime: int, max_factors: int, max_exp: int) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{ts}_P{max_prime}_F{max_factors}_E{max_exp}"


# ── main ──────────────────────────────────────────────────────
def main() -> None:
    """Parse checkpoint (if any), run search, display results."""
    checkpoint_present = os.path.exists(CHECKPOINT_FILE)
    chk = load_checkpoint()
    if checkpoint_present and chk is None:
        print(
            "检查点存在但无法安全恢复；"
            "本次不会启动新搜索或覆盖原文件。"
        )
        return
    prev_solutions: list = []

    if chk is not None:
        print("=" * 60)
        print("发现已有检查点 (v4)，将从中断处继续 ...")
        print(f"  已完成状态: {chk['total_states']:,}")
        print(f"  已用时间:   {chk['elapsed']:.1f}s")
        print(f"  已找到解:   {len(chk.get('solutions', []))}")
        print("=" * 60)
        prev_solutions = chk.get("solutions", [])
        metrics = RunMetrics.from_checkpoint_payload(chk["metrics"])
        run_id = chk["run_id"]
        elapsed_offset = chk["elapsed"]
        # primes, max_factors, max_exp are set below in the unified
        # prime-generation phase (after the sampler is created).
        primes = None
        max_factors = chk.get("max_factors", MAX_FACTORS)
        max_exp = chk.get("max_exp", MAX_EXP)
        resume_state = {
            "heap":         chk.get("heap", []),
            "heap_counter": chk.get("heap_counter", 0),
            "states_started": chk.get("states_started", 0),
            "states_completed": chk.get("states_completed", 0),
            "total_states": chk.get("total_states", 0),
            "elapsed":      chk.get("elapsed", 0.0),
            "use_heap":     chk.get("use_heap", PROPAGATE),
            "snapshot_id":  chk.get("snapshot_id", 0),
        } if chk.get("heap") else None

    else:
        metrics = RunMetrics()
        run_id = _make_run_id(MAX_PRIME, MAX_FACTORS, MAX_EXP)
        elapsed_offset = 0.0
        resume_state = None

    started_at = datetime.now(timezone.utc)
    git_commit = _git_commit()
    git_dirty = _git_dirty()
    run_dir = prepare_run_directory(run_id)
    abundancy_capture_config = AbundancyCaptureConfig(
        enabled=(
            CAPTURE_ABUNDANCY_GAP_STATES
            and PROPAGATE
            and SEARCH_MODE.target_num == 2
            and SEARCH_MODE.target_den == 1
            and SEARCH_MODE.require_euler
        ),
        max_gap_num=ABUNDANCY_GAP_MAX_NUM,
        max_gap_den=ABUNDANCY_GAP_MAX_DEN,
        max_records=ABUNDANCY_GAP_MAX_RECORDS,
        text_limit=ABUNDANCY_GAP_TEXT_LIMIT,
    )
    abundancy_recorder = AbundancyGapRecorder(
        run_dir,
        run_id=run_id,
        target_num=SEARCH_MODE.target_num,
        target_den=SEARCH_MODE.target_den,
        resume_productive_ordinal=metrics.structure.productive_states,
        config=abundancy_capture_config,
    )

    # Create sampler before prime generation so the phase is recorded
    sampler = RuntimeSampler(
        run_dir / "performance_samples.csv",
        elapsed_offset=elapsed_offset,
        append=(chk is not None),
    )
    sampler.start()

    if chk is not None:
        sampler.set_phase("prime_generation")
        primes = open_or_extend_prime_pool(chk["prime_limit"])
        # strict fingerprint verification — any mismatch aborts
        if len(primes) != chk["prime_count"]:
            raise RuntimeError(
                f"checkpoint prime_count mismatch: "
                f"expected {chk['prime_count']}, got {len(primes)}"
            )
        if prime_pool_typecode(primes) != chk["prime_typecode"]:
            raise RuntimeError(
                f"checkpoint prime_typecode mismatch: "
                f"expected {chk['prime_typecode']!r}, "
                f"got {prime_pool_typecode(primes)!r}"
            )
        if int(primes[0]) != chk["first_prime"]:
            raise RuntimeError(
                f"checkpoint first_prime mismatch: "
                f"expected {chk['first_prime']}, got {int(primes[0])}"
            )
        if int(primes[-1]) != chk["last_prime"]:
            raise RuntimeError(
                f"checkpoint last_prime mismatch: "
                f"expected {chk['last_prime']}, got {int(primes[-1])}"
            )
        sampler.capture_memory_phase(
            metrics.performance.memory_phases, "after_prime_generation",
        )
    else:
        sampler.set_phase("prime_generation")
        primes = open_or_extend_prime_pool(MAX_PRIME)
        max_factors = MAX_FACTORS
        max_exp = MAX_EXP
        sampler.capture_memory_phase(
            metrics.performance.memory_phases, "after_prime_generation",
        )

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
    print(f"运行 ID:     {run_id}")
    print("=" * 60)
    print("按一次 Ctrl+C 在当前状态完成后安全保存；再次按下立即中断\n")

    solutions    = list(prev_solutions)
    state_holder: dict = {}
    t0           = time.time()
    found_true = sum(
        1 for _assigned, _euler_prime, spoof in prev_solutions
        if not spoof
    )
    found_spoof = len(prev_solutions) - found_true
    stop_requested = False
    interrupt_count = 0

    sampler.set_phase("startup")

    # ── progress callback ──
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
        abundancy_recorder.commit(
            metrics.structure.productive_states
        )
        if reason in {"initial", "periodic"}:
            save_checkpoint(holder, solutions, run_id=run_id, metrics=metrics)

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

    status = "FAILED"
    try:
        sampler.set_phase("search")
        for st in search_opn(
            primes, max_factors, max_exp,
            metrics=metrics,
            propagate=PROPAGATE,
            state_holder=state_holder,
            resume_state=resume_state,
            observer=sampler,
            productive_observer=abundancy_recorder.capture,
            progress_callback=_show_progress,
            checkpoint_callback=_save_stable_boundary,
            checkpoint_interval_seconds=CHECKPOINT_INTERVAL_SECONDS,
            stop_requested=lambda: stop_requested,
            sigma_database_path=(
                SIGMA_DATABASE_FILE
                if SIGMA_DATABASE_ENABLED
                else None
            ),
            pool_plan_cache_dir=(
                POOL_PLAN_DISK_CACHE_DIR
                if POOL_PLAN_DISK_CACHE_ENABLED
                else None
            ),
            pool_plan_cache_minimum_free_bytes=(
                POOL_PLAN_DISK_MIN_FREE_BYTES
            ),
            pool_plan_build_policy=POOL_PLAN_BUILD_POLICY,
        ):
            if st.spoof:
                found_spoof += 1
            else:
                found_true += 1
            solutions.append((dict(st.assigned), st.euler_prime, st.spoof))
            display_solution(st, len(solutions), time.time() - t0)
            export_factor_graph(st, path=f"factor_graph_{len(solutions)}")
            if not st.spoof or not PROPAGATE:
                save_checkpoint(state_holder, solutions, run_id=run_id, metrics=metrics)
            save_solutions_txt(solutions)
        status = "COMPLETE"

    except SearchStopped:
        print("\n\n已到达稳定搜索边界，正在保存 ...")
        save_checkpoint(state_holder, solutions, run_id=run_id, metrics=metrics)
        save_solutions_txt(solutions)
        status = "STOPPED"

    except KeyboardInterrupt:
        print("\n\n已立即中断搜索。")
        save_solutions_txt(solutions)
        status = "INTERRUPTED"
        if os.path.exists(CHECKPOINT_FILE):
            print("保留了最近一次完整的原子检查点；恢复时可能重算少量状态。")
        else:
            print("中断发生在首次检查点完成前，本次没有可恢复的完整检查点。")

    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        sampler.set_phase("report_write")
        sampler.stop()

        elapsed = elapsed_offset + time.time() - t0

        metrics.performance.cache_sizes = {
            "SIGMA_CACHE": len(opn_core.SIGMA_CACHE),
            "_SIG_VALUATIONS": len(opn_core._SIG_VALUATIONS),
        }
        metrics.performance.memory_phases["at_report"] = (
            sampler.capture_memory()
        )

        abundancy_summary = abundancy_recorder.finalize(
            status=status,
            sigma_database_path=(
                SIGMA_DATABASE_FILE
                if SIGMA_DATABASE_ENABLED
                else None
            ),
        )

        config = {
            "max_prime": int(primes[-1]) if len(primes) > 0 else MAX_PRIME,
            "max_factors": max_factors,
            "max_exp": max_exp,
            "target_num": SEARCH_MODE.target_num,
            "target_den": SEARCH_MODE.target_den,
            "require_euler": SEARCH_MODE.require_euler,
            "propagate": PROPAGATE,
            "pool_gcd_mode": POOL_GCD_MODE,
            "pool_fanout": POOL_SUPERBLOCK_FANOUT,
            "pool_plan_build_policy": POOL_PLAN_BUILD_POLICY,
            "pool_plan_disk_cache_enabled": (
                POOL_PLAN_DISK_CACHE_ENABLED
            ),
            "pool_plan_disk_cache_dir": (
                POOL_PLAN_DISK_CACHE_DIR
                if POOL_PLAN_DISK_CACHE_ENABLED
                else None
            ),
            "sigma_database_enabled": SIGMA_DATABASE_ENABLED,
            "q3_prepool_mode": Q3_PREPOOL_MODE,
            "domain_ratio_mode": DOMAIN_RATIO_MODE,
            "pending_selection": PENDING_SELECTION,
            "abundancy_gap_capture_enabled": (
                abundancy_capture_config.enabled
            ),
            "abundancy_gap_max_num": (
                abundancy_capture_config.max_gap_num
            ),
            "abundancy_gap_max_den": (
                abundancy_capture_config.max_gap_den
            ),
            "abundancy_gap_max_records": (
                abundancy_capture_config.max_records
            ),
            "abundancy_gap_text_limit": (
                abundancy_capture_config.text_limit
            ),
            "git_dirty": git_dirty,
        }
        write_all_reports(
            run_dir=run_dir,
            run_id=run_id,
            git_commit=git_commit,
            git_dirty=git_dirty,
            status=status,
            started_at=started_at.isoformat(),
            config=config,
            metrics=metrics,
            elapsed_seconds=elapsed,
            solutions_found=found_true + found_spoof,
            sampled_peak_rss=sampler.sampled_peak_rss,
            pruning_policy=PRUNING_POLICY,
            telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
            _sig_factors=opn_core._SIG_FACTORS,
            abundancy_capture_summary=abundancy_summary,
        )

        if status == "STOPPED":
            print(
                f"已精确保存。已完成 "
                f"{state_holder.get('total_states', 0):,} 个状态；"
                f"前沿还有 {state_holder.get('frontier_size', 0):,} 个状态。"
            )
            return

        if status == "COMPLETE":
            print(
                f"\n搜索完成。总状态: {state_holder.get('total_states', 0):,}, "
                f"耗时: {elapsed:.1f}s"
            )
            if solutions:
                print(f"\n=== 共 {found_true} 个真解 + {found_spoof} 个伪解 ===")
            else:
                print("\n在搜索范围内无解。")
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                print("已清理检查点文件。")

        if status in ("INTERRUPTED", "FAILED"):
            print(f"运行状态: {status}，报告已保存到 {run_dir}")


if __name__ == "__main__":
    main()
