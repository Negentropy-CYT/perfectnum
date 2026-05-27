"""
legacy/core — prime generation and integer factorisation.

Mirrors the original ``main.py`` implementation exactly.
"""
from typing import List, Tuple


def generate_odd_primes(limit: int) -> List[int]:
    """Return all odd primes ≤ *limit* (Eratosthenes sieve)."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:limit + 1:step] = [False] * ((limit - start) // step + 1)
    return [i for i in range(3, limit + 1, 2) if sieve[i]]


def factorize(x: int) -> List[Tuple[int, int]]:
    """Trial-division factorisation (6k±1 wheel)."""
    res: List[Tuple[int, int]] = []
    if x % 2 == 0:
        c = 0
        while x % 2 == 0:
            x //= 2
            c += 1
        res.append((2, c))
    if x % 3 == 0:
        c = 0
        while x % 3 == 0:
            x //= 3
            c += 1
        res.append((3, c))
    d = 5
    step = 2
    while d * d <= x:
        if x % d == 0:
            c = 0
            while x % d == 0:
                x //= d
                c += 1
            res.append((d, c))
        d += step
        step = 6 - step
    if x > 1:
        res.append((x, 1))
    return res
