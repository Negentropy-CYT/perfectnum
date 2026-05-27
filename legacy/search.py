"""
legacy/search — original subset-DFS searcher (a_i = 1, all exponents fixed to 2).

Searches for numbers of the form  N = r · ∏ p_i²  satisfying
(r+1)·∏(p_i²+p_i+1) = 2r·∏p_i²,  using suffix-product pruning and
a manual stack-based DFS.
"""
import sys
import time
from typing import Any, Generator, List, Tuple

from gmpy2 import mpz


def search_v4_safe(
    primes: List[int],
    max_subset_size: int = 7,
    progress_interval: int = 10000,
    checkpoint_state: dict | None = None,
    state_holder: dict | None = None,
) -> Generator[Tuple[Any, Any, Tuple[int, ...]], None, None]:
    """Yield ``(n, r, t)`` for each pseudo-OPN candidate found."""

    primes = sorted(primes)
    n = len(primes)

    # precompute p² and σ(p²)
    sqs = [mpz(p * p) for p in primes]
    sigs = [mpz(p * p + p + 1) for p in primes]

    # suffix products for upper-bound pruning
    suffix_num = [mpz(1) for _ in range(n + 1)]
    suffix_den = [mpz(1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_num[i] = suffix_num[i + 1] * sigs[i]
        suffix_den[i] = suffix_den[i + 1] * sqs[i]

    if checkpoint_state is not None:
        stack = checkpoint_state["stack"]
        current_path = checkpoint_state["current_path"]
        total_states = checkpoint_state["total_states"]
        elapsed_offset = checkpoint_state["elapsed"]
    else:
        stack = [(0, 0, mpz(1), mpz(1), 0, 0)]
        current_path = []
        total_states = 0
        elapsed_offset = 0.0

    start_time = time.time() - elapsed_offset

    if state_holder is not None:
        state_holder.update({
            "primes": primes,
            "max_subset_size": max_subset_size,
            "stack": stack,
            "current_node": None,
            "current_path": current_path,
            "total_states": total_states,
            "start_time": start_time,
        })

    while stack:
        start, size, sigma, sq, ptr, path_len = stack.pop()
        del current_path[path_len:]

        if state_holder is not None:
            state_holder["stack"] = stack
            state_holder["current_node"] = (start, size, sigma, sq, ptr, path_len)
            state_holder["current_path"] = current_path
            state_holder["total_states"] = total_states

        if ptr == 0:
            total_states += 1
            if total_states % progress_interval == 0:
                elapsed = time.time() - start_time
                rate = total_states / elapsed if elapsed > 0 else 0
                sys.stdout.write(
                    f"\r[Progress] States: {total_states:,} | "
                    f"Time: {elapsed:.1f}s | Rate: {rate:.0f} states/s"
                )
                sys.stdout.flush()

            denom = 2 * sq - sigma
            if size >= 1 and denom > 0:
                if sigma % denom == 0:
                    r = sigma // denom
                    if r > 1 and (r & 3) == 1:
                        ok = True
                        for p in current_path:
                            if r % p == 0:
                                ok = False
                                break
                        if ok:
                            yield r * sq, r, tuple(current_path)

            if size >= max_subset_size or denom <= 0:
                continue
            ptr = start

        while ptr < n:
            p = primes[ptr]
            sq_p = sqs[ptr]
            sig_p = sigs[ptr]

            new_sigma = sigma * sig_p
            new_sq = sq * sq_p
            new_denom = 2 * new_sq - new_sigma

            if new_denom <= 0:
                ptr += 1
                continue
            if size + 1 > max_subset_size:
                break
            # pruning: if even all remaining primes can't push ratio high
            # enough for r >= 5, abandon this branch
            if 6 * sigma * suffix_num[ptr] < 10 * sq * suffix_den[ptr]:
                break

            # skip-prime branch
            stack.append((start, size, sigma, sq, ptr + 1, path_len))

            # take-prime branch
            current_path.append(p)
            stack.append((
                ptr + 1,
                size + 1,
                new_sigma,
                new_sq,
                0,
                len(current_path)
            ))
            break

    elapsed = time.time() - start_time
    print(f"\n\n搜索完成: {total_states:,} states, {elapsed:.4f}s")

    if state_holder is not None:
        state_holder["stack"] = []
        state_holder["current_node"] = None
        state_holder["current_path"] = []
        state_holder["total_states"] = total_states


def verify_solution(n: int, r: int, t: Tuple[int, ...]) -> bool:
    """Check (r+1)·∏(p²+p+1) == 2r·∏p²."""
    lhs = r + 1
    rhs = 2 * r
    for p in t:
        lhs *= (p * p + p + 1)
        rhs *= (p * p)
    return lhs == rhs
