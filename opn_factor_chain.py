"""
Odd Perfect Number Search --- Factor Chain Engine.

Extends beyond the a_i = 1 (all non-Euler exponents fixed to 2) restriction.
Supports general even exponents for non-Euler primes, uses σ(p^a)
factorization to propagate forced primes (factor chains), and applies
ratio-interval pruning.

Architecture:
  1. σ(p^a) computation & factorization
  2. Ratio bounds estimation (upper / lower)
  3. Factor chain propagation engine
  4. Constraint-aware DFS with pruning
  5. Checkpoint & solution reporting
"""
import gmpy2
from gmpy2 import mpz, is_prime as gmpy2_is_prime
import time
import sys
import os
import pickle
import math
import itertools
from typing import List, Tuple, Set, Dict, Optional, Generator
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_FILE = "checkpoint_opn_chain.pkl"
SOLUTIONS_FILE = "solutions_opn_chain.txt"

# ---------------------------------------------------------------------------
# Prime generation
# ---------------------------------------------------------------------------
def generate_odd_primes(limit: int) -> List[int]:
    """All odd primes ≤ limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit + 1:step] = [False] * ((limit - start) // step + 1)
    return [i for i in range(3, limit + 1, 2) if sieve[i]]


# ---------------------------------------------------------------------------
# σ(p^a) computation
# ---------------------------------------------------------------------------
def sigma_prime_power(p: int, a: int) -> mpz:
    """σ(p^a) = (p^(a+1) - 1) / (p - 1)   for prime p, exponent a ≥ 0."""
    if a == 0:
        return mpz(1)
    num = pow(p, a + 1) - 1  # gmpy2 pow → mpz for large ints
    den = p - 1
    return mpz(num) // den


def ratio_prime_power(p: int, a: int) -> mpz:
    """σ(p^a) / p^a  as a rational: returns (numerator, denominator)."""
    if a == 0:
        return mpz(1), mpz(1)
    sig = sigma_prime_power(p, a)
    pa = mpz(p ** a)
    return sig, pa


def ratio_bound_upper(p: int) -> mpz:
    """Upper bound of σ(p^a)/p^a as a → ∞:  p/(p-1).
    Returns (numerator, denominator)."""
    return mpz(p), mpz(p - 1)


def ratio_bound_lower(p: int) -> mpz:
    """Lower bound of σ(p^a)/p^a for a ≥ 1:  1 + 1/p.
    Returns (numerator, denominator)."""
    return mpz(p + 1), mpz(p)


# ---------------------------------------------------------------------------
# Integer factorization (trial division + 6k±1 for speed)
# ---------------------------------------------------------------------------
def factorize(x: int) -> List[Tuple[int, int]]:
    """Return list of (prime, exponent) for x.  x may be int or mpz."""
    x = int(x)
    res: List[Tuple[int, int]] = []
    if x <= 1:
        return res
    for p in (2, 3):
        if x % p == 0:
            c = 0
            while x % p == 0:
                x //= p
                c += 1
            res.append((p, c))
    d, step = 5, 2
    limit = int(math.isqrt(x))
    while d <= limit:
        if x % d == 0:
            c = 0
            while x % d == 0:
                x //= d
                c += 1
            res.append((d, c))
            limit = int(math.isqrt(x))
        d += step
        step = 6 - step
    if x > 1:
        res.append((x, 1))
    return res


# ---------------------------------------------------------------------------
# Search state
# ---------------------------------------------------------------------------
@dataclass
class ChainState:
    """One node in the factor-chain DFS."""

    # --- assigned prime factors with exponents ---
    # All exponents are even for non-Euler primes.
    # The Euler prime (if set) is the ONLY prime with odd exponent.
    factors: Dict[int, int]       # prime → exponent

    # Euler prime (None = not yet chosen)
    euler: Optional[int]

    # --- accumulated σ(N) / N as rational ---
    # ratio_num / ratio_den = ∏ σ(p^a) / ∏ p^a  for assigned factors
    num: mpz   # product of all σ(p^a)
    den: mpz   # product of all p^a

    # --- pending primes from factor chains ---
    pending: Set[int]

    # --- explicitly excluded primes (skip branch: must NOT appear in N) ---
    excluded: Set[int]

    # --- next prime index to consider (into sorted prime list) ---
    next_idx: int

    # --- statistics ---
    depth: int = 0

    def clone(self) -> "ChainState":
        return ChainState(
            factors=dict(self.factors),
            euler=self.euler,
            num=mpz(self.num),
            den=mpz(self.den),
            pending=set(self.pending),
            excluded=set(self.excluded),
            next_idx=self.next_idx,
            depth=self.depth + 1,
        )

    def assign(self, p: int, a: int, is_euler: bool) -> Optional["ChainState"]:
        """Return new state with `p^a` assigned, or None if invalid."""
        if p in self.factors:
            return None  # already assigned
        if a <= 0:
            return None

        st = self.clone()
        st.factors[p] = a
        if is_euler:
            st.euler = p

        sig = sigma_prime_power(p, a)
        st.num = st.num * sig
        st.den = st.den * mpz(p ** a)

        # Factor-chain propagation: prime factors of σ(p^a) become pending.
        # Skip 2: N is odd, so 2 cannot divide N. (σ(q^odd) may contain 2,
        # but it is absorbed by the factor 2 in σ(N)=2N.)
        for q, _ in factorize(sig):
            if q == 2:
                continue
            if q != p and q not in st.factors:
                st.pending.add(q)

        return st

    def skip_to(self, skipped_prime: int, next_idx: int) -> "ChainState":
        """Return copy of state with `skipped_prime` excluded and next_idx advanced."""
        st = self.clone()
        st.excluded.add(skipped_prime)
        st.next_idx = next_idx
        return st

    @property
    def ratio(self) -> float:
        """σ(N)/N = num/den as float (for display only)."""
        return float(self.num) / float(self.den)

    @property
    def is_complete(self) -> bool:
        """Check if σ(N) == 2N exactly."""
        return self.num == 2 * self.den

    @property
    def is_valid_solution(self) -> bool:
        """Euler's theorem: odd perfect number N must have exactly one prime
        (the 'special' or 'Euler' prime) with odd exponent.
        That prime must be ≡ 1 (mod 4) and its exponent must be ≡ 1 (mod 4)."""
        if not self.is_complete:
            return False
        if self.euler is None:
            return False
        if len(self.factors) < 2:
            return False
        # Euler prime: both conditions are mandatory (not conditional)
        if self.euler % 4 != 1:
            return False
        euler_exp = self.factors[self.euler]
        if euler_exp % 2 != 1:
            return False
        if euler_exp % 4 != 1:
            return False
        return True


# ---------------------------------------------------------------------------
# Pruning: ratio bounds
# ---------------------------------------------------------------------------
def max_possible_ratio(
    state: ChainState,
    primes: List[int],
    suffix_prod_num: List[mpz],
    suffix_prod_den: List[mpz],
) -> mpz:
    """Maximum possible ratio achievable from current state.
    Uses the upper bound p/(p-1) for each remaining unassigned prime."""
    # suffix_prod_num[i] / suffix_prod_den[i] = ∏_{j≥i} p_j/(p_j-1)
    n = len(primes)
    idx = state.next_idx
    if idx >= n:
        return state.num, state.den
    return state.num * suffix_prod_num[idx], state.den * suffix_prod_den[idx]


def min_possible_ratio(
    state: ChainState,
    primes: List[int],
    suffix_prod_num: List[mpz],
    suffix_prod_den: List[mpz],
) -> mpz:
    """Minimum possible ratio from current state.
    Uses lower bound (p+1)/p for each remaining unassigned prime.
    Note: the minimum is actually the current ratio (add nothing),
    but we also need to consider forced primes from pending."""
    # Current ratio is the minimum (add nothing more)
    return state.num, state.den


def can_reach_target(
    state: ChainState,
    primes: List[int],
    suffix_ub_num: List[mpz],
    suffix_ub_den: List[mpz],
) -> bool:
    """Check if ratio == 2 is reachable from current state."""
    # Already exceeded target?
    if state.num >= 2 * state.den:
        return False  # ratio ≥ 2, can only increase

    # Max possible still below target?
    max_num, max_den = max_possible_ratio(state, primes, suffix_ub_num, suffix_ub_den)
    if max_num < 2 * max_den:
        return False  # even with all remaining primes at max ratio, can't reach 2

    return True


# ---------------------------------------------------------------------------
# Exponent selection
# ---------------------------------------------------------------------------
NON_EULER_EXPONENTS = (2, 4, 6)      # Even exponents for non-Euler primes
EULER_EXPONENTS = (1, 5, 9)          # Odd exponents ≡ 1 mod 4 for Euler prime
EXTRA_EXPONENTS = (8, 10)            # Additional even exponents for small primes


def valid_exponents(p: int, is_euler_candidate: bool, max_exp: int = 6) -> List[int]:
    """Return list of valid exponents for prime p."""
    if is_euler_candidate:
        # Euler prime: odd exponent, ≡ 1 mod 4 if p ≡ 1 mod 4
        if p % 4 == 1:
            # Must be ≡ 1 mod 4
            return [e for e in EULER_EXPONENTS if e <= max_exp]
        else:
            # p ≡ 3 mod 4: any odd exponent, but p ≡ 3 mod 4 cannot be Euler
            # prime by Euler's theorem (Euler prime must be ≡ 1 mod 4)
            return []  # p ≡ 3 mod 4 cannot be the Euler prime
    else:
        # Non-Euler prime: even exponent
        exps = list(NON_EULER_EXPONENTS)
        if p <= 7:
            exps.extend(EXTRA_EXPONENTS)
        return [e for e in exps if e <= max_exp]


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------
def search_opn(
    primes: List[int],
    max_factors: int = 9,
    max_exp: int = 8,
    progress_interval: int = 100_000,
    state_holder: dict | None = None,
    resume_state: dict | None = None,
) -> Generator[Tuple[ChainState, int], None, None]:
    """
    Main search entry point.

    Yields (state, solution_number) for each valid odd perfect number found.

    If resume_state is provided, restores DFS stack / statistics from
    a previous interrupted run.  state_holder (mutable dict) is updated
    on each iteration for external checkpoint saves.
    """
    primes = sorted(primes)
    n = len(primes)

    # Precompute suffix upper-bound products: ∏ p/(p-1)
    suffix_ub_num = [mpz(1) for _ in range(n + 1)]
    suffix_ub_den = [mpz(1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ub_n, ub_d = ratio_bound_upper(primes[i])
        suffix_ub_num[i] = suffix_ub_num[i + 1] * ub_n
        suffix_ub_den[i] = suffix_ub_den[i + 1] * ub_d

    # Precompute suffix lower-bound products: ∏ (p+1)/p
    suffix_lb_num = [mpz(1) for _ in range(n + 1)]
    suffix_lb_den = [mpz(1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        lb_n, lb_d = ratio_bound_lower(primes[i])
        suffix_lb_num[i] = suffix_lb_num[i + 1] * lb_n
        suffix_lb_den[i] = suffix_lb_den[i + 1] * lb_d

    if resume_state is not None:
        stack = resume_state["stack"]
        total_states = resume_state["total_states"]
        elapsed_offset = resume_state["elapsed"]
    else:
        stack = [
            ChainState(
                factors={},
                euler=None,
                num=mpz(1),
                den=mpz(1),
                pending=set(),
                excluded=set(),
                next_idx=0,
                depth=0,
            )
        ]
        total_states = 0
        elapsed_offset = 0.0

    solutions_found = 0
    t0 = time.time() - elapsed_offset

    if state_holder is not None:
        state_holder.update({
            "primes": primes,
            "max_factors": max_factors,
            "max_exp": max_exp,
            "stack": stack,
            "total_states": total_states,
            "elapsed": elapsed_offset,
        })

    while stack:
        state = stack.pop()

        if state_holder is not None:
            state_holder["stack"] = [state] + list(stack)
            state_holder["total_states"] = total_states
            state_holder["elapsed"] = time.time() - t0

        # ---- Phase 1: process pending (forced) primes ----
        if state.pending:
            p = state.pending.pop()
            if p in state.factors:
                # Already assigned; put back and continue processing phase
                stack.append(state)
                continue

            # Contradiction: forced prime was explicitly excluded earlier
            if p in state.excluded:
                continue

            # Check for infeasibility
            if p > primes[-1]:
                continue  # Can't assign primes beyond our search range

            # Branch: could be Euler prime or regular prime
            is_euler_cand = (state.euler is None and p % 4 == 1)
            exps = valid_exponents(p, is_euler_cand, max_exp)
            if not exps and state.euler is None:
                # Can't be Euler (≡ 3 mod 4), use even exponents
                exps = valid_exponents(p, False, max_exp)

            if not exps:
                continue

            # Push expanded states in reverse order for DFS correctness
            for a in reversed(exps):
                is_euler = is_euler_cand and (a % 2 == 1)
                new_state = state.assign(p, a, is_euler)
                if new_state is not None:
                    stack.append(new_state)
            continue

        # ---- Phase 2: check solution ----
        total_states += 1

        if total_states % progress_interval == 0:
            elapsed = time.time() - t0
            rate = total_states / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\r[Progress] States: {total_states:,} | "
                f"Depth: {state.depth} | "
                f"Factors: {len(state.factors)} | "
                f"Ratio: {state.ratio:.6f} | "
                f"Rate: {rate:.0f}/s"
            )
            sys.stdout.flush()

        if state.is_valid_solution:
            solutions_found += 1
            yield state, solutions_found

        # ---- Phase 3: pruning ----
        if state.num >= 2 * state.den:
            continue  # ratio already exceeded target

        if len(state.factors) >= max_factors:
            continue  # depth limit

        if not can_reach_target(state, primes, suffix_ub_num, suffix_ub_den):
            continue

        # ---- Phase 4: expand ----
        idx = state.next_idx
        while idx < n:
            p = primes[idx]
            if p in state.factors:
                idx += 1
                continue

            # Upper-bound pruning: if even with all remaining primes at max
            # ratio we can't reach 2, stop exploring this branch entirely.
            max_num, max_den = max_possible_ratio(
                state, primes, suffix_ub_num, suffix_ub_den,
            )
            if max_num < 2 * max_den:
                break

            # Branch: skip this prime (explicitly exclude it)
            skip_state = state.skip_to(p, idx + 1)
            if skip_state is not None:
                stack.append(skip_state)

            # Branch: include this prime with various exponents
            is_euler_cand = (state.euler is None and p % 4 == 1)
            exps = valid_exponents(p, is_euler_cand, max_exp)

            for a in reversed(exps):
                is_euler = is_euler_cand and (a % 2 == 1)
                new_state = state.assign(p, a, is_euler)
                if new_state is not None:
                    new_state.next_idx = idx + 1
                    stack.append(new_state)

            break  # Only branch on the first unassigned prime

    elapsed = time.time() - t0
    print(f"\n\n搜索完成: {total_states:,} states, {elapsed:.1f}s")

    if state_holder is not None:
        state_holder["stack"] = []
        state_holder["total_states"] = total_states
        state_holder["elapsed"] = elapsed


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_opn(state: ChainState) -> bool:
    """Full verification of an odd perfect number candidate."""
    lhs = mpz(1)
    rhs = mpz(1)
    for p, a in state.factors.items():
        lhs = lhs * sigma_prime_power(p, a)
        rhs = rhs * mpz(p ** a)
    return lhs == 2 * rhs


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def display_solution(state: ChainState, sol_num: int, elapsed: float) -> None:
    """Print a found solution."""
    n_val = int(state.den)  # N = ∏ p^a
    print(f"\n{'=' * 60}")
    print(f"*** 解 #{sol_num} ***")
    print(f"N = {n_val}")
    print(f"log10(N) = {math.log10(n_val):.1f}")
    print(f"因子数 = {len(state.factors)}")
    print(f"Euler 质数 = {state.euler}")
    print(f"因子分解:")
    for p, a in sorted(state.factors.items()):
        marker = " (Euler)" if p == state.euler else ""
        print(f"  {p}^{a}{marker}  σ={sigma_prime_power(p, a)}")
    print(f"σ(N)/N = {float(state.num) / float(state.den):.10f}")
    print(f"验证 = {verify_opn(state)}")
    print(f"发现时间: {elapsed:.1f}s")

    # Factor-chain trace for the Euler prime
    if state.euler:
        print(f"\n因子链 (从 Euler 质数 {state.euler} 开始):")
        seen = set()
        todo = [(state.euler, state.factors[state.euler])]
        while todo:
            p, a = todo.pop(0)
            if p in seen:
                continue
            seen.add(p)
            sig = int(sigma_prime_power(p, a))
            facs = factorize(sig)
            fac_str = " × ".join(f"{q}^{e}" for q, e in facs)
            print(f"  σ({p}^{a}) = {sig} = {fac_str}")
            for q, _ in facs:
                if q not in seen and q in state.factors:
                    todo.append((q, state.factors[q]))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint_chain(state_holder: dict, solutions: list) -> None:
    """Save full DFS state + solutions to checkpoint file."""
    chk = {
        "primes": state_holder.get("primes", []),
        "max_factors": state_holder.get("max_factors", 9),
        "max_exp": state_holder.get("max_exp", 6),
        "stack": state_holder.get("stack", []),
        "total_states": state_holder.get("total_states", 0),
        "elapsed": state_holder.get("elapsed", 0.0),
        "solutions": solutions,
    }
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(chk, f, pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, CHECKPOINT_FILE)


def load_checkpoint_chain() -> Optional[dict]:
    """Load checkpoint. Returns full state dict or None."""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"警告: 检查点损坏 ({e})")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ========== configuration ==========
    MAX_PRIME = 400          # Smaller prime range for initial factor-chain testing
    MAX_FACTORS = 12          # Max distinct prime factors
    MAX_EXP = 3               # Max exponent to try
    # ===================================

    chk = load_checkpoint_chain()
    prev_solutions: list = []
    prev_states = 0
    prev_elapsed = 0.0

    if chk is not None:
        print("=" * 60)
        print("发现已有检查点，将从中断处继续...")
        print(f"已完成状态: {chk['total_states']:,}")
        print(f"已用时间:   {chk['elapsed']:.1f}s")
        print(f"已找到解:   {len(chk['solutions'])}")
        print("=" * 60)
        prev_solutions = chk["solutions"]
        prev_states = chk["total_states"]
        prev_elapsed = chk["elapsed"]
        primes = chk["primes"]
        max_factors = chk["max_factors"]
        max_exp = chk["max_exp"]
    else:
        primes = generate_odd_primes(MAX_PRIME)
        max_factors = MAX_FACTORS
        max_exp = MAX_EXP

    print(f"质数范围: ≤ {primes[-1]}  (共 {len(primes)} 个)")
    print(f"最大因子数: {max_factors}")
    print(f"最大指数:   {max_exp}")
    print(f"非 Euler 指数候选: {list(NON_EULER_EXPONENTS)}")
    print(f"Euler 指数候选:    {list(EULER_EXPONENTS)}")
    print("=" * 60)
    print("开始搜索（按 Ctrl+C 安全中断）...\n")

    solutions = list(prev_solutions)
    state_holder: dict = {}
    resume_state: dict | None = None

    if chk is not None and chk.get("stack"):
        resume_state = {
            "stack": chk["stack"],
            "total_states": chk.get("total_states", 0),
            "elapsed": chk.get("elapsed", 0.0),
        }

    start_all = time.time()

    try:
        for state, sol_num in search_opn(
            primes, max_factors, max_exp,
            state_holder=state_holder,
            resume_state=resume_state,
        ):
            solutions.append((dict(state.factors), state.euler))
            display_solution(state, sol_num + len(prev_solutions),
                             time.time() - start_all + prev_elapsed)
            save_checkpoint_chain(state_holder, solutions)

    except KeyboardInterrupt:
        print(f"\n\n收到中断信号，正在保存...")
        save_checkpoint_chain(state_holder, solutions)
        print(f"已保存。找到 {len(solutions)} 个解。")
        print(f"下次运行将从中断处继续。")
        sys.exit(0)

    elapsed = time.time() - start_all + prev_elapsed
    print(f"\n搜索完成。总状态: {state_holder.get('total_states', 0):,}, 耗时: {elapsed:.1f}s")

    if solutions:
        print(f"\n=== 共 {len(solutions)} 个解 ===")
    else:
        print("\n在搜索范围内无解。")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


if __name__ == "__main__":
    main()
