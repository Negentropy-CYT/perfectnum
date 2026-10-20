#!/usr/bin/env python3
"""
Factor-chain search for perfect numbers below a bound B.

Proves (by exhaustive branching) that no perfect number N <= B exists with the
required parity, or finds them all.  This is the Brent-Cohen-te Riele method
(Math. Comp. 1991) in simplified form; the same family of methods underlies the
published bound N > 10^1500 (Ochem & Rao, Math. Comp. 2012).

State of a search node:
  S       : dict prime -> exponent (fixed: p^a exactly divides N)
  U       : frozenset of primes known to divide N, exponent not yet fixed
  lo      : every prime factor of N outside S|U is >= lo
  special : the Euler prime (q = 1 mod 4 with exponent = 1 mod 4) once fixed;
            odd mode only -- sigma(N) = 2N forces exactly one such prime
  num/den : A_S = num/den = prod_{S} sigma(p^a)/p^a   (den = prod_S p^a)
  vcount  : r -> sum over S of v_r(sigma(p^a)) = lower bound on v_r(N)
            (for r = 2 in "all" mode the bound on v_2(N) is vcount[2]-1,
             because sigma(N) = 2N carries one extra factor 2)

Branching:
  * U nonempty: take p = max(U); branch over admissible exponents a.
      Forced part K = prod(S) * p^a * prod_{U-p} r^emin divides N, and
      sigma(p^a) | 2N, hence N >= K * sigma(p^a) / gcd(sigma(p^a), 2K) -- this
      is checked BEFORE factoring anything and cuts the exponent range to
      roughly p^(2a) <= 2B.  Then every prime r of sigma(p^a) divides N with
      v_r(N) >= v_r(sigma(p^a)); new primes must be >= lo and join U.
  * U empty: M = N / prod(S) satisfies sigma(M)/M = 2/A_S exactly.
      A_S > 2  -> dead.   A_S == 2 -> M = 1, perfect number found.
      A_S < 2  -> M > 1; branch on its smallest prime r1.  Finitely many
      candidates: omega(M) is bounded by B, and prod r/(r-1) over the primes of
      M must exceed 2/A_S, which fails once r1 is large.
A branch ends productively when the forced part of N exceeds B.

Roadblocks: when a cyclotomic piece Phi_d(p) of sigma(p^a) keeps a composite
cofactor c we cannot factor within the effort budget, strip the primes already
in S|U; the rest c' is a product of NEW primes r with v_r(N) >= v_r(c'), so
N >= K * c'.  If the special prime is already fixed, every prime of c' is
non-special and must have an even exponent in N, so a non-square c' gives one
more factor > L (its primes are certified > L by trial division).  If the bound
still does not exceed B the branch is recorded as UNRESOLVED and the proof is
incomplete.

Mode "all" treats 2 like any prime and must recover exactly the even perfect
numbers <= B -- the correctness self-test.  Mode "odd" is the real search.
"""
import argparse
import bisect
import json
import os
import sys
import time
import urllib.request
from fractions import Fraction

import gmpy2
from gmpy2 import mpz
from sympy import factorint, isprime, nextprime, primerange
from sympy.functions.combinatorial.numbers import mobius
from sympy.ntheory import ecm, pollard_pm1, pollard_rho

# --------------------------------------------------------------------------
# sigma(p^a) and its factorisation via cyclotomic polynomials
# --------------------------------------------------------------------------

TRIAL_LIMIT = 10**6          # plain trial division bound (all primes)
CONGR_LIMIT = 10**8          # trial division bound with q = 1 (mod d) candidates
CONGR_MAX_ITER = 3_000_000   # cap on candidates tried per stubborn composite
FULL_FACTORINT_BITS = 128    # below this, hand the number to sympy.factorint outright
FACTOR_CACHE = {}            # (d, p) -> (primes dict, [(composite, certified_L)], effort_used, fdb_checked)
CACHE_FILE = None
EFFORT = 2                   # 1..4, bounded factoring effort

# factordb.com lookup (Cunningham-project tables live there).  Every answer is
# verified locally: the product must equal the number and each claimed prime
# must pass isprime; nothing is trusted on the database's word.
FDB_ENABLED = True
FDB_MIN_DIGITS = 30          # do not bother the database with small numbers
FDB_SLEEP = 0.3              # seconds between queries (politeness)
FDB_STATS = {"queries": 0, "full": 0, "partial": 0, "none": 0, "errors": 0}


def factordb_lookup(n):
    """Returns list of (factor, exp) verified to multiply to n (each factor prime or
    composite as reported), or None when unavailable / useless."""
    n = int(n)
    if not FDB_ENABLED or len(str(n)) < FDB_MIN_DIGITS:
        return None
    FDB_STATS["queries"] += 1
    try:
        req = urllib.request.Request("http://factordb.com/api?query=" + str(n),
                                     headers={"User-Agent": "opn-factor-chain-search/1.0 (research; throttled)"})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.load(r)
    except Exception:
        FDB_STATS["errors"] += 1
        return None
    finally:
        time.sleep(FDB_SLEEP)
    factors = []
    prod = 1
    try:
        for f, e in data.get("factors", []):
            f, e = int(f), int(e)
            factors.append((f, e))
            prod *= f ** e
    except Exception:
        FDB_STATS["errors"] += 1
        return None
    if prod != n or len(factors) == 0 or (len(factors) == 1 and factors[0] == (n, 1)):
        FDB_STATS["none"] += 1
        return None
    if all(isprime(f) for f, _ in factors):
        FDB_STATS["full"] += 1
    else:
        FDB_STATS["partial"] += 1
    return factors

SMALL_PRIMES = list(primerange(2, 3_000_000))
_TRIAL_PRIMES = [mpz(q) for q in SMALL_PRIMES if q <= TRIAL_LIMIT]


def cyclotomic_value(d, p):
    """Phi_d(p) as an exact integer via the Moebius formula."""
    num = mpz(1)
    den = mpz(1)
    for e in range(1, d + 1):
        if d % e:
            continue
        mu = mobius(e)
        if mu == 1:
            num *= mpz(p) ** (d // e) - 1
        elif mu == -1:
            den *= mpz(p) ** (d // e) - 1
    q, r = divmod(num, den)
    assert r == 0
    return int(q)


def _ecm_one(c, B1, B2, curves):
    try:
        fs = ecm(c, B1=B1, B2=B2, max_curve=curves)
    except ValueError:
        return None
    for f in fs:
        if 1 < f < c and c % f == 0:
            return int(f)
    return None


def _try_split(c, effort):
    """Return a proper factor of composite c, or None, within a bounded effort."""
    c = int(c)
    tries = []
    if effort >= 1:
        tries.append(lambda: pollard_pm1(c, B=20000))
        tries.append(lambda: pollard_rho(c, max_steps=300_000))
        tries.append(lambda: _ecm_one(c, 2000, 200_000, 40))
    if effort >= 2:
        tries.append(lambda: _ecm_one(c, 11000, 1_100_000, 90))
    if effort >= 3:
        tries.append(lambda: _ecm_one(c, 50000, 5_000_000, 240))
    if effort >= 4:
        tries.append(lambda: _ecm_one(c, 250000, 25_000_000, 600))
    for t in tries:
        try:
            f = t()
        except Exception:
            f = None
        if f and 1 < f < c and c % f == 0:
            return int(f)
    return None


def _trial_divide(c):
    """Strip all prime factors <= TRIAL_LIMIT.  Returns (primes dict, cofactor)."""
    primes = {}
    c = mpz(c)
    for q in _TRIAL_PRIMES:
        if q * q > c:
            break
        if c % q == 0:
            e = 0
            while c % q == 0:
                c //= q
                e += 1
            primes[int(q)] = e
    if c == 1:
        return primes, mpz(1)
    if c <= TRIAL_LIMIT * TRIAL_LIMIT or isprime(c):
        primes[int(c)] = primes.get(int(c), 0) + 1
        return primes, mpz(1)
    return primes, c


def _congruence_divide(c, d):
    """Last resort for a stubborn composite c | Phi_d(p): trial-divide by every
    q = 1 (mod d) [1 (mod 2d) for odd d] above TRIAL_LIMIT.  Every prime factor
    of Phi_d(p) not dividing d is = 1 mod d and factors <= TRIAL_LIMIT are gone,
    so afterwards no prime factor <= certified_L remains.
    Returns (primes dict, cofactor, certified_L)."""
    primes = {}
    c = mpz(c)
    step = d if d % 2 == 0 else 2 * d
    lim = min(CONGR_LIMIT, TRIAL_LIMIT + step * CONGR_MAX_ITER)
    q = mpz(1 + step * (TRIAL_LIMIT // step))
    while q <= lim:
        if c % q == 0:
            e = 0
            while c % q == 0:
                c //= q
                e += 1
            primes[int(q)] = primes.get(int(q), 0) + e
            if c == 1 or isprime(c):
                if c != 1:
                    primes[int(c)] = primes.get(int(c), 0) + 1
                return primes, mpz(1), int(lim)
        q += step
    return primes, c, int(lim)


def _factor_stack(stack, d, effort, primes, comps, use_fdb=True):
    """Work a stack of (composite, multiplicity, certified_L) items down."""
    asked = set()
    while stack:
        c, e, L = stack.pop()
        c = mpz(c)
        if c == 1:
            continue
        if isprime(c):
            primes[int(c)] = primes.get(int(c), 0) + e
            continue
        if gmpy2.is_power(c):
            for k in range(2, c.bit_length() + 1):
                r, exact = gmpy2.iroot(c, k)
                if exact:
                    stack.append((mpz(r), e * k, L))
                    break
            continue
        if use_fdb and int(c) not in asked:
            asked.add(int(c))
            fdb = factordb_lookup(c)
            if fdb:
                for f, k in fdb:
                    stack.append((mpz(f), e * k, L))
                    if not isprime(f):
                        asked.add(int(f))   # the database already told us all it knows about this cofactor
                continue
        f = _try_split(c, effort)
        if f is None:
            if L < CONGR_LIMIT:
                pr, c2, L2 = _congruence_divide(c, d)
                for q, k in pr.items():
                    primes[q] = primes.get(q, 0) + e * k
                if c2 != 1:
                    stack.append((c2, e, L2))
                continue
            comps.append((int(c), int(L)))
            continue
        f = mpz(f)
        stack.append((f, e, L))
        stack.append((c // f, e, L))


def bounded_factor(n, d, effort, use_fdb=True):
    """Factor Phi_d(p) = n with bounded effort.  Returns (primes, [(composite, L)])."""
    primes, c = _trial_divide(n)
    comps = []
    if c != 1:
        _factor_stack([(c, 1, TRIAL_LIMIT)], d, effort, primes, comps, use_fdb)
    return primes, comps


def factor_cyclotomic(d, p):
    key = (d, p)
    if key in FACTOR_CACHE:
        primes, comps, eff, fdb = FACTOR_CACHE[key]
        if not comps or (eff >= EFFORT and (fdb or not FDB_ENABLED)):
            return primes, comps
        primes = dict(primes)
        newc = []
        _factor_stack([(c, 1, L) for c, L in comps], d, EFFORT, primes, newc, use_fdb=not fdb)
        res = (primes, newc, EFFORT, fdb or FDB_ENABLED)
    else:
        v = cyclotomic_value(d, p)
        if v == 1:
            res = ({}, [], EFFORT, True)
        elif v.bit_length() <= FULL_FACTORINT_BITS:
            res = (dict(factorint(v)), [], EFFORT, True)
        else:
            pr, co = bounded_factor(v, d, EFFORT)
            res = (pr, co, EFFORT, FDB_ENABLED)
    FACTOR_CACHE[key] = res
    return res[0], res[1]


def prefactor_lookup(d, p):
    """Cheap main-process step: trial division + factordb only (no ECM).  Leaves
    anything unresolved in the cache marked effort 0 so workers finish it."""
    key = (d, p)
    if key in FACTOR_CACHE and (not FACTOR_CACHE[key][1] or FACTOR_CACHE[key][3]):
        return FACTOR_CACHE[key]
    if key in FACTOR_CACHE:
        primes, comps, eff, _ = FACTOR_CACHE[key]
        primes = dict(primes)
        newc = []
        _factor_stack([(c, 1, L) for c, L in comps], d, 0, primes, newc, use_fdb=True)
        res = (primes, newc, eff if not newc else 0, True)
    else:
        v = cyclotomic_value(d, p)
        if v == 1:
            res = ({}, [], EFFORT, True)
        elif v.bit_length() <= FULL_FACTORINT_BITS:
            res = (dict(factorint(v)), [], EFFORT, True)
        else:
            pr, co = bounded_factor(v, d, 0, use_fdb=True)
            res = (pr, co, 0, True)
    FACTOR_CACHE[key] = res
    return res


def sigma_factorisation(p, a):
    """sigma(p^a) = prod_{d | a+1, d>1} Phi_d(p).  Returns (dict prime->exp, [(composite, L)])."""
    total, comps = {}, []
    n = a + 1
    for d in range(2, n + 1):
        if n % d:
            continue
        f, c = factor_cyclotomic(d, p)
        for q, e in f.items():
            total[q] = total.get(q, 0) + e
        comps.extend(c)
    return total, comps


def sigma_pow(p, a):
    return (mpz(p) ** (a + 1) - 1) // (p - 1)


def primes_from(lo):
    i = bisect.bisect_left(SMALL_PRIMES, lo)
    while i < len(SMALL_PRIMES):
        yield SMALL_PRIMES[i]
        i += 1
    q = SMALL_PRIMES[-1]
    while True:
        q = int(nextprime(q))
        yield q


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

STAT_KEYS = ["nodes", "closed_size", "closed_sigma", "closed_abundant", "closed_parity",
             "closed_lo", "closed_exp", "closed_noprime", "roadblock_closed",
             "roadblock_open", "max_depth", "max_prime"]


class Search:
    def __init__(self, B, odd, verbose=False, frontier_depth=None, quiet=False):
        self.B = mpz(B)
        self.odd = odd
        self.verbose = verbose
        self.quiet = quiet
        self.frontier_depth = frontier_depth
        self.frontier = []
        self.found = []
        self.unresolved = []
        self.stats = {k: 0 for k in STAT_KEYS}
        self.t0 = time.time()
        self.last_report = self.t0

    # ---- helpers ----
    def emin(self, r, vcount, special_fixed=False):
        """Smallest admissible exponent of r in N given the forced valuation."""
        if not self.odd:
            if r == 2:
                return max(1, vcount.get(2, 0) - 1)
            return max(1, vcount.get(r, 0))
        v = max(1, vcount.get(r, 0))
        if r % 4 == 1 and not special_fixed:
            # even exponent, or special (= 1 mod 4)
            return v if (v % 2 == 0 or v % 4 == 1) else v + 1
        return v if v % 2 == 0 else v + 1

    def min_remaining(self, lo, S, k):
        prod = mpz(1)
        ps = []
        for q in primes_from(lo):
            if q in S:
                continue
            ps.append(q)
            if len(ps) == k:
                break
        if self.odd:
            for q in ps[:-1]:
                prod *= q * q
            prod *= ps[-1]
        else:
            for q in ps:
                prod *= q
        return prod

    def max_omega_remaining(self, lo, S, budget):
        k = 0
        while k < 500 and self.min_remaining(lo, S, k + 1) <= budget:
            k += 1
        return k

    def admissible_exponents(self, p, emin, special_fixed):
        a = max(emin, 1)
        while True:
            if not self.odd:
                yield a, False
            elif a % 2 == 0:
                yield a, False
            elif p % 4 == 1 and a % 4 == 1 and not special_fixed:
                yield a, True
            a += 1

    def report(self, force=False):
        if self.quiet and not force:
            return
        now = time.time()
        if force or now - self.last_report > 30:
            self.last_report = now
            s = self.stats
            print(f"  [{now - self.t0:8.1f}s] " + " ".join(f"{k}={v}" for k, v in s.items() if k != "max_prime")
                  + f" maxp~10^{len(str(s['max_prime']))-1}", flush=True)

    def root_candidates(self):
        s = Search(self.B, self.odd, frontier_depth=1, quiet=True)
        s.run()
        return sorted(min(st[1]) for st in s.frontier)

    def level2_pairs(self):
        """(d, p) cyclotomic values the first exponent level will need (for parallel pre-factoring)."""
        pairs = set()
        for r1 in self.root_candidates():
            for a, _sp in self.admissible_exponents(r1, 1, False):
                pa = mpz(r1) ** a
                if pa > self.B:
                    break
                sig = sigma_pow(r1, a)
                if pa * (sig // gmpy2.gcd(sig, 2 * pa)) > self.B:
                    continue
                for d in range(2, a + 2):
                    if (a + 1) % d == 0:
                        pairs.add((d, r1))
        return pairs

    # ---- core recursion ----
    def run(self):
        self.recurse({}, frozenset(), 2 if not self.odd else 3, None, mpz(1), mpz(1), {}, 0)
        return self.found

    def recurse(self, S, U, lo, special, num, den, vcount, depth):
        if self.frontier_depth is not None and depth >= self.frontier_depth:
            self.frontier.append((dict(S), frozenset(U), lo, special, int(num), int(den), dict(vcount), depth))
            return
        self.stats["nodes"] += 1
        self.stats["max_depth"] = max(self.stats["max_depth"], depth)
        self.report()
        num, den = mpz(num), mpz(den)

        for r, a in S.items():
            need = vcount.get(r, 0) - (1 if (not self.odd and r == 2) else 0)
            if need > a:
                self.stats["closed_exp"] += 1
                return
        if self.odd and vcount.get(2, 0) > 1:
            self.stats["closed_parity"] += 1
            return
        sfx = special is not None
        lnum, lden = num, den
        for r in U:
            e = self.emin(r, vcount, sfx)
            lnum *= sigma_pow(r, e)
            lden *= mpz(r) ** e
        if lnum > 2 * lden:
            self.stats["closed_abundant"] += 1
            return
        if lden > self.B:
            self.stats["closed_size"] += 1
            return

        if U:
            p = max(U)
            U2 = U - {p}
            emin = self.emin(p, vcount, sfx)
            base = lden // (mpz(p) ** emin)
            for a, is_special in self.admissible_exponents(p, emin, special is not None):
                pa = mpz(p) ** a
                K = base * pa
                if K > self.B:
                    self.stats["closed_size"] += 1
                    break
                sig = sigma_pow(p, a)
                if K * (sig // gmpy2.gcd(sig, 2 * K)) > self.B:
                    self.stats["closed_sigma"] += 1
                    continue
                sfac, comps = sigma_factorisation(p, a)
                S2 = dict(S)
                S2[p] = a
                vc = dict(vcount)
                for r, e in sfac.items():
                    vc[r] = vc.get(r, 0) + e
                if self.odd and vc.get(2, 0) > 1:
                    self.stats["closed_parity"] += 1
                    continue
                newU = set(U2)
                dead = False
                for r in sfac:
                    if r == 2:
                        if self.odd:
                            continue
                        if 2 in S2 or 2 in newU:
                            continue
                        if vc[2] - 1 >= 1:
                            if lo > 2:
                                dead = True
                                break
                            newU.add(2)
                        continue
                    if r in S2:
                        if vc[r] > S2[r]:
                            dead = True
                            break
                        continue
                    if r in newU:
                        continue
                    if r < lo:
                        dead = True
                        break
                    newU.add(r)
                    self.stats["max_prime"] = max(self.stats["max_prime"], r)
                if dead:
                    self.stats["closed_lo"] += 1
                    continue
                sp = p if is_special else special
                if comps:
                    known = set(S2) | newU
                    extra_min = mpz(1)
                    open_block = False
                    for c, L in comps:
                        c = mpz(c)
                        for r in known:
                            while c % r == 0:
                                c //= r
                        if c == 1:
                            continue
                        if isprime(c):
                            if c < lo:
                                dead = True
                                break
                            newU.add(int(c))
                            continue
                        # c is a product of new primes > L with v_r(N) >= v_r(c)
                        extra_min *= c
                        if self.odd and sp is not None and not gmpy2.is_square(c):
                            extra_min *= mpz(L)   # a non-special prime with odd multiplicity needs one more copy
                        open_block = True
                    if dead:
                        self.stats["closed_lo"] += 1
                        continue
                    if open_block:
                        forced = K
                        for r in newU:
                            if r not in U2:
                                forced *= mpz(r) ** self.emin(r, vc, sp is not None)
                        if forced * extra_min > self.B:
                            self.stats["roadblock_closed"] += 1
                            continue
                        self.stats["roadblock_open"] += 1
                        self.unresolved.append((p, a, [(int(c), int(L)) for c, L in comps], int(forced)))
                        if not self.quiet:
                            print(f"  UNRESOLVED roadblock: sigma({p}^{a}) composites "
                                  f"{[(len(str(c)), L) for c, L in comps]} digits; forced~10^{len(str(int(forced)))-1}", flush=True)
                        continue
                self.recurse(S2, frozenset(newU), lo, sp,
                             num * sig, den * pa, vc, depth + 1)
            return

        # U empty
        if num > 2 * den:
            self.stats["closed_abundant"] += 1
            return
        if num == 2 * den:
            N = int(den)
            tot = 1
            for q, a in S.items():
                tot *= int(sigma_pow(q, a))
            if tot == 2 * N and (not self.odd or N % 2 == 1):
                self.found.append((N, dict(S)))
                print(f"  *** PERFECT NUMBER FOUND: {N} = {S}", flush=True)
            return
        budget = self.B // den
        target = Fraction(2 * int(den), int(num))
        cands = []
        for r1 in primes_from(lo):
            if r1 in S:
                continue
            m = self.max_omega_remaining(r1, S, budget)
            if m == 0:
                break
            F = Fraction(1)
            cnt = 0
            for q in primes_from(r1):
                if q in S:
                    continue
                F *= Fraction(q, q - 1)
                cnt += 1
                if cnt == m:
                    break
            if F <= target:
                break
            cands.append(r1)
        if not cands:
            self.stats["closed_noprime"] += 1
            return
        for r1 in cands:
            self.recurse(S, frozenset({r1}), int(nextprime(r1)), special, num, den, vcount, depth + 1)


# --------------------------------------------------------------------------
# cache + CLI
# --------------------------------------------------------------------------

def load_cache(path):
    global CACHE_FILE
    CACHE_FILE = path
    if path and os.path.exists(path):
        with open(path) as f:
            raw = json.load(f)
        FACTOR_CACHE.update(raw_to_cache(raw))


def cache_to_raw(cache):
    return {f"{d},{p}": {"p": {str(q): e for q, e in v[0].items()},
                         "c": [[str(c), int(L)] for c, L in v[1]], "e": v[2], "f": bool(v[3])}
            for (d, p), v in cache.items()}


def raw_to_cache(raw):
    out = {}
    for k, v in raw.items():
        d, p = map(int, k.split(","))
        comps = []
        for c in v.get("c", []):
            if isinstance(c, list):
                comps.append((int(c[0]), int(c[1])))
            else:
                comps.append((int(c), TRIAL_LIMIT))
        out[(d, p)] = ({int(q): e for q, e in v["p"].items()}, comps, int(v.get("e", 0)), bool(v.get("f", False)))
    return out


def save_cache(path=None):
    path = path or CACHE_FILE
    if path:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache_to_raw(FACTOR_CACHE), f)
        for attempt in range(40):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                time.sleep(0.5)


def parse_bound(s):
    if "e" in s:
        mant, ex = s.split("e")
        return int(Fraction(mant) * Fraction(10) ** int(ex))
    return int(eval(s, {"__builtins__": {}}))


def main():
    global EFFORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", required=True, help="bound, e.g. 1e30 or 10**30")
    ap.add_argument("--mode", choices=["odd", "all"], default="odd")
    ap.add_argument("--effort", type=int, default=2, help="bounded factoring effort 1..4")
    ap.add_argument("--cache", default=None)
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    EFFORT = args.effort
    B = parse_bound(args.B)
    load_cache(args.cache)
    print(f"Search mode={args.mode} B={args.B} ({len(str(B))} digits) effort={EFFORT}", flush=True)
    s = Search(B, odd=(args.mode == "odd"), verbose=args.v)
    t0 = time.time()
    try:
        found = s.run()
    finally:
        save_cache()
    dt = time.time() - t0
    s.report(force=True)
    print(f"factordb: {FDB_STATS}")
    print(f"done in {dt:.1f}s; found={[n for n, _ in found]}; unresolved={len(s.unresolved)}")
    for p, a, cs, forced in s.unresolved[:50]:
        print(f"  sigma({p}^{a}) has unfactored composite(s): {cs}")
    if args.mode == "odd" and not found and not s.unresolved:
        print(f"PROVED: no odd perfect number <= {args.B}")
    return 0 if not s.unresolved else 2


if __name__ == "__main__":
    sys.exit(main())
