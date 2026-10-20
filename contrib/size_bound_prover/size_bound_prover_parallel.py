#!/usr/bin/env python3
"""Parallel driver for size_bound_prover.

Phase 0  pre-factor, in parallel, every cyclotomic value Phi_d(p) the first
         exponent level will need (these are the largest numbers in the search).
Phase 1  expand the search tree to a frontier depth in the main process
         (fast now, everything it needs is cached).
Phase 2  farm the frontier subtrees out to a process pool.
Phase 3  if any roadblock stayed unresolved, raise the factoring effort and
         repeat phases 1-2 (cached entries without composites are instant; only
         the stubborn composites get the extra ECM effort).
Workers read a snapshot of the cache and return their new entries for merging.
"""
import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time

import size_bound_prover as core


def _init(cache_path, effort):
    core.EFFORT = effort
    core.load_cache(cache_path)
    # workers may query factordb too, but rarely and gently (the bulk lookups are serial in the main process)
    core.FDB_MIN_DIGITS = 45
    core.FDB_SLEEP = 1.0


def _prefactor(task):
    d, p = task
    t0 = time.time()
    core.factor_cyclotomic(d, p)
    return (d, p), core.cache_to_raw({(d, p): core.FACTOR_CACHE[(d, p)]}), time.time() - t0


def _work(task):
    idx, B, odd, state = task
    before = set(core.FACTOR_CACHE)
    s = core.Search(B, odd=odd, quiet=True)
    t0 = time.time()
    s.recurse(*state)
    new = {k: v for k, v in core.FACTOR_CACHE.items() if k not in before}
    return idx, s.stats, s.found, s.unresolved, core.cache_to_raw(new), time.time() - t0, state


def merge_raw(raw):
    for k, v in core.raw_to_cache(raw).items():
        old = core.FACTOR_CACHE.get(k)
        if old is None or (old[1] and (not v[1] or v[2] > old[2] or (v[3] and not old[3]))):
            core.FACTOR_CACHE[k] = v


def run_phase(args, B, odd, effort, t0):
    core.EFFORT = effort
    snap = args.cache + ".workers.json"
    core.save_cache()
    shutil.copyfile(args.cache, snap)

    # Phase 0a: serial, throttled factordb + trial division pass over the level-2 cyclotomic values
    probe = core.Search(B, odd=odd, quiet=True)
    pairs = sorted(probe.level2_pairs(), key=lambda dp: -dp[0] * dp[1])
    need = [(d, p) for (d, p) in pairs
            if (d, p) not in core.FACTOR_CACHE
            or (core.FACTOR_CACHE[(d, p)][1] and not core.FACTOR_CACHE[(d, p)][3])]
    print(f"[effort {effort}] phase 0a: {len(need)} of {len(pairs)} level-2 cyclotomic values to look up", flush=True)
    for i, (d, p) in enumerate(need):
        core.prefactor_lookup(d, p)
        if (i + 1) % 50 == 0:
            print(f"  [{time.time()-t0:8.1f}s] lookup {i+1}/{len(need)} factordb={core.FDB_STATS}", flush=True)
            core.save_cache()
    core.save_cache()
    shutil.copyfile(args.cache, snap)
    # Phase 0b: parallel ECM on whatever is still composite
    todo = [(d, p) for (d, p) in pairs
            if core.FACTOR_CACHE[(d, p)][1] and core.FACTOR_CACHE[(d, p)][2] < effort]
    print(f"[effort {effort}] phase 0b: {len(todo)} composites left for local ECM; factordb={core.FDB_STATS}", flush=True)
    if todo:
        with mp.Pool(args.workers, initializer=_init, initargs=(snap, effort)) as pool:
            n = 0
            for key, raw, dt in pool.imap_unordered(_prefactor, todo, chunksize=1):
                merge_raw(raw)
                n += 1
                if dt > 30 or n % 100 == 0:
                    comps = core.FACTOR_CACHE[key][1]
                    print(f"  [{time.time()-t0:8.1f}s] prefactor {n}/{len(todo)} Phi_{key[0]}({key[1]}) {dt:.1f}s"
                          f"{' UNFACTORED ' + str([len(str(c)) for c, _ in comps]) + ' digits' if comps else ''}", flush=True)
        core.save_cache()
        shutil.copyfile(args.cache, snap)

    # Phase 1: frontier
    s = core.Search(B, odd=odd, frontier_depth=args.depth, quiet=True)
    s.run()
    core.save_cache()
    shutil.copyfile(args.cache, snap)
    tasks = [(i, B, odd, st) for i, st in enumerate(s.frontier)]
    print(f"[effort {effort}] phase 1: {len(tasks)} subtrees at depth {args.depth} after {time.time()-t0:.1f}s; "
          f"top stats {{{', '.join(f'{k}={v}' for k, v in s.stats.items() if v and k != 'max_prime')}}}", flush=True)
    found = list(s.found)
    unresolved = list(s.unresolved)
    agg = dict(s.stats)

    # Phase 2: pool
    done = 0
    slow = []
    with mp.Pool(args.workers, initializer=_init, initargs=(snap, effort)) as pool:
        for idx, stats, f, unres, newcache, dt, state in pool.imap_unordered(_work, tasks, chunksize=1):
            done += 1
            for k, v in stats.items():
                agg[k] = max(agg[k], v) if k in ("max_depth", "max_prime") else agg[k] + v
            found.extend(f)
            unresolved.extend(unres)
            merge_raw(newcache)
            slow.append((dt, idx, state[0], sorted(state[1])))
            if done % 50 == 0 or dt > 120:
                S, U = state[0], sorted(state[1])
                print(f"  [{time.time()-t0:8.1f}s] {done}/{len(tasks)} done; last {dt:.1f}s "
                      f"S={S} U~{[len(str(u)) for u in U[:4]]}d; nodes={agg['nodes']} "
                      f"rb_closed={agg['roadblock_closed']} rb_open={agg['roadblock_open']}", flush=True)
    core.save_cache()
    slow.sort(reverse=True)
    return found, unresolved, agg, slow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", required=True)
    ap.add_argument("--mode", choices=["odd", "all"], default="odd")
    ap.add_argument("--effort", type=int, default=2)
    ap.add_argument("--max-effort", type=int, default=4)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = ap.parse_args()
    B = core.parse_bound(args.B)
    odd = args.mode == "odd"
    core.load_cache(args.cache)
    if not os.path.exists(args.cache):
        core.save_cache()
    print(f"Parallel search mode={args.mode} B={args.B} ({len(str(B))} digits) effort={args.effort}..{args.max_effort} "
          f"depth={args.depth} workers={args.workers}", flush=True)
    t0 = time.time()
    effort = args.effort
    while True:
        found, unresolved, agg, slow = run_phase(args, B, odd, effort, t0)
        print(f"[effort {effort}] result after {time.time()-t0:.1f}s: found={[n for n, _ in found]} "
              f"unresolved={len(unresolved)} stats={{{', '.join(f'{k}={v}' for k, v in agg.items() if k != 'max_prime')}}}", flush=True)
        for t, idx, S, U in slow[:3]:
            print(f"  slow subtree {t:8.1f}s  S={S} U~{[len(str(u)) for u in U[:5]]} digits")
        for p, a, cs, forced in unresolved[:30]:
            print(f"  UNRESOLVED sigma({p}^{a}) composites {[(len(str(c)), L) for c, L in cs]} (digits, certified L) forced~10^{len(str(forced))-1}")
        if found or not unresolved or effort >= args.max_effort:
            break
        effort += 1
        print(f"raising factoring effort to {effort} for {len(unresolved)} unresolved roadblocks", flush=True)
    print(f"total time {time.time()-t0:.1f}s")
    if odd and not found and not unresolved:
        print(f"PROVED: no odd perfect number <= {args.B}")
    return 0 if not unresolved else 2


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
