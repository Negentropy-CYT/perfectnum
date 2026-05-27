"""
legacy/main — original subset-DFS OPN searcher (a_i = 1).

This is the reference implementation preserved for reproducibility.
It searches for pseudo-OPN candidates of the form

    N = r · ∏ p_i²

using a manual stack DFS with suffix-product pruning.

Usage
-----
    python legacy/main.py
"""
import sys
import time

from legacy.core import factorize, generate_odd_primes
from legacy.io import load_checkpoint, save_checkpoint
from legacy.search import search_v4_safe, verify_solution


def main() -> None:
    # ========== configuration ==========
    MAX_PRIME       = 200
    MAX_SUBSET_SIZE = 8
    # ===================================

    checkpoint, prev_solutions = load_checkpoint()

    if checkpoint is not None:
        print("=" * 70)
        print("发现已有检查点，将从中断处继续搜索...")
        print(f"已完成状态数: {checkpoint['total_states']:,}")
        print(f"已用时间: {checkpoint['elapsed']:.1f}s")
        print(f"已找到 {len(prev_solutions)} 个解")
        print("=" * 70)

        if prev_solutions:
            print("\n之前搜索到的解:")
            for i, (n_val, r_val, t) in enumerate(prev_solutions, 1):
                print(f"  解 #{i}: r={r_val}, t={t}")
                print(f"        n={n_val}")
            print()

        primes = checkpoint["primes"]
        max_subset_size = checkpoint["max_subset_size"]
    else:
        print("未发现检查点，开始全新搜索...")
        primes = generate_odd_primes(MAX_PRIME)
        max_subset_size = MAX_SUBSET_SIZE

    print(f"质数列表 (共 {len(primes)} 个): {primes[:10]}...{primes[-10:]}")
    print(f"最大子集大小: {max_subset_size}")
    print("=" * 70)
    print("开始搜索（找到解会立即输出）...")
    print("按 Ctrl+C 可安全中断并保存进度\n")

    solutions = list(prev_solutions) if prev_solutions else []
    state_holder: dict = {}
    start_all = time.time()

    try:
        for n_val, r_val, t in search_v4_safe(
            primes,
            max_subset_size,
            checkpoint_state=checkpoint,
            state_holder=state_holder,
        ):
            solutions.append((n_val, r_val, t))
            solution_count = len(solutions)
            print(f"\n{'=' * 50}")
            print(f"*** 解 #{solution_count} ***")
            print(f"t = {t}")
            print(f"r = {r_val}")
            print(f"n = {n_val}")
            print(f"位数 = {len(str(n_val))}")
            r_factors = factorize(r_val)
            print(f"r 的分解 = {r_factors}")
            print(f"验证 = {verify_solution(n_val, r_val, t)}")
            print(f"发现时间: {time.time() - start_all:.2f} 秒")
            save_checkpoint(state_holder, solutions)

    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在保存进度...")
        save_checkpoint(state_holder, solutions)
        print(
            f"进度已保存。已搜索 "
            f"{state_holder.get('total_states', 0):,} 个状态"
        )
        print(f"已找到 {len(solutions)} 个解")
        print("下次运行将从中断处继续。")
        sys.exit(0)

    if len(solutions) == 0:
        print("\n无解")
    else:
        print(
            f"\n=== 搜索结束，共找到 {len(solutions)} 个解，"
            f"总耗时 {time.time() - start_all:.2f} 秒 ==="
        )

    import os
    from legacy.io import CHECKPOINT_FILE
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("搜索完成，已清理检查点文件。")


if __name__ == "__main__":
    main()
