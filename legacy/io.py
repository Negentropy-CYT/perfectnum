"""
legacy/io — checkpoint persistence and human-readable solution output.

These functions preserve the exact behaviour of the original ``main.py``
checkpoint mechanism (pickle-based DFS stack serialisation).
"""

import os
import pickle
import time

CHECKPOINT_FILE = "checkpoint.pkl"
SOLUTIONS_FILE  = "solutions.txt"


def save_checkpoint(state_holder: dict, solutions: list) -> None:
    """Atomically persist search state + solutions to disk."""
    current_node = state_holder.get("current_node")
    stack = state_holder.get("stack", [])
    if current_node is not None:
        stack_snapshot = [current_node] + list(stack)
    else:
        stack_snapshot = list(stack) if stack else []

    elapsed = time.time() - state_holder.get("start_time", time.time())

    checkpoint = {
        "primes":         state_holder.get("primes", []),
        "max_subset_size": state_holder.get("max_subset_size", 7),
        "stack":          stack_snapshot,
        "current_path":   list(state_holder.get("current_path", [])),
        "total_states":   state_holder.get("total_states", 0),
        "elapsed":        elapsed,
        "solutions":      solutions,
    }
    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(checkpoint, f, pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, CHECKPOINT_FILE)

    # human-readable solutions
    with open(SOLUTIONS_FILE, "w", encoding="utf-8") as f:
        f.write("# Odd Perfect Number Search Results\n")
        f.write(f"# Solutions found: {len(solutions)}\n\n")
        for i, (n_val, r_val, t) in enumerate(solutions, 1):
            f.write(f"Solution #{i}:\n")
            f.write(f"  n = {n_val}\n")
            f.write(f"  r = {r_val}\n")
            f.write(f"  t = {t}\n")
            f.write(f"  digits of n = {len(str(n_val))}\n")
            f.write("\n")


def load_checkpoint() -> tuple:
    """Return ``(state_dict, solutions)`` or ``(None, [])``."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None, []
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            checkpoint = pickle.load(f)
        return checkpoint, checkpoint.get("solutions", [])
    except Exception as e:
        print(f"警告: 检查点文件损坏 ({e})，将开始全新搜索。")
        return None, []
