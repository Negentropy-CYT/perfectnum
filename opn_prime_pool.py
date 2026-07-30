"""
opn_prime_pool — persistent prime-pool storage backed by a single uint64
binary file and minimal JSON metadata.

The public entry point is ``open_or_extend_prime_pool(limit)`` which
returns a read-only NumPy array (memmap when the pool is on disk).

Reliability rule
----------------
Metadata only describes a fully committed prefix.  Any partial tail
left by an interrupted extension is silently truncated on the next
open.  No file locking — only one process may build or extend at a time.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

_POOL_DIR = Path("prime_pool")
_DATA_NAME = "odd_primes_u64.bin"
_META_NAME = "odd_primes_u64.json"

_META_FORMAT_VERSION = 1


# ── segment sieve (shared with opn_core.generate_odd_primes) ──────

def iter_odd_prime_chunks(
    start: int,
    limit: int,
    *,
    segment_odds: int = 2_000_000,
):
    """Yield ``np.ndarray(dtype=uint64)`` chunks of odd primes in [*start*, *limit*].

    The segment sieve uses O(segment_odds) working memory.  *start* is
    silently aligned up to the nearest odd integer so the caller does not
    need to track parity of ``committed_limit + 1``.
    """
    if limit < 3:
        return

    # ── align start to odd ──
    if start % 2 == 0:
        start += 1
    if start > limit:
        return

    if segment_odds <= 0:
        raise ValueError("segment_odds must be positive")

    # Small-sieve primes up to sqrt(limit) — always relative to *limit*,
    # not *start*, because a future segment may contain numbers whose
    # smallest prime factor exceeds sqrt(start).
    root = math.isqrt(limit)
    base_sieve = np.ones(root + 1, dtype=np.bool_)
    base_sieve[:2] = False
    for p in range(2, math.isqrt(root) + 1):
        if base_sieve[p]:
            base_sieve[p * p: root + 1: p] = False
    base_primes = np.flatnonzero(base_sieve)
    odd_base = base_primes[base_primes >= 3]

    segment_span = 2 * segment_odds  # integer range per segment

    for low in range(start, limit + 1, segment_span):
        high = min(limit, low + segment_span - 2)
        if high % 2 == 0:
            high -= 1
        count = ((high - low) // 2) + 1
        segment = np.ones(count, dtype=np.bool_)

        for p_val in odd_base:
            p = int(p_val)
            p_sq = p * p
            if p_sq > high:
                break
            start_pos = max(p_sq, ((low + p - 1) // p) * p)
            if start_pos % 2 == 0:
                start_pos += p
            first = (start_pos - low) // 2
            segment[first::p] = False

        indices = np.flatnonzero(segment)
        yield (low + 2 * indices).astype(np.uint64, copy=False)


# ── metadata ──────────────────────────────────────────────────────

def _read_metadata() -> dict | None:
    """Return the parsed metadata document, or None if missing / corrupt."""
    meta_path = _POOL_DIR / _META_NAME
    if not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(doc, dict):
        return None
    if doc.get("format_version") != _META_FORMAT_VERSION:
        return None
    if not isinstance(doc.get("committed_count"), int) or doc["committed_count"] < 1:
        return None
    if not isinstance(doc.get("committed_limit"), int) or doc["committed_limit"] < 3:
        return None
    if not isinstance(doc.get("last_prime"), int) or doc["last_prime"] < 3:
        return None

    return doc


def _write_metadata(
    *,
    committed_limit: int,
    committed_count: int,
    last_prime: int,
) -> None:
    """Atomically write the pool metadata file."""
    _POOL_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = _POOL_DIR / _META_NAME
    tmp_path = _POOL_DIR / (_META_NAME + ".tmp")

    doc = {
        "format_version": _META_FORMAT_VERSION,
        "committed_limit": committed_limit,
        "committed_count": committed_count,
        "last_prime": last_prime,
    }

    with tmp_path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())

    os.replace(str(tmp_path), str(meta_path))


# ── data-file validation ──────────────────────────────────────────

def _validate_and_open_data_file(meta: dict) -> np.ndarray | None:
    """Open the committed data file as a read-only memmap, or None if invalid."""
    data_path = _POOL_DIR / _DATA_NAME
    if not data_path.exists():
        return None

    expected_bytes = meta["committed_count"] * 8
    actual_bytes = data_path.stat().st_size

    if actual_bytes < expected_bytes:
        # Metadata promises more data than exists — unrecoverable.
        return None

    if actual_bytes > expected_bytes:
        # Tail from an interrupted extension — truncate it.
        try:
            with data_path.open("r+b") as fh:
                fh.truncate(expected_bytes)
        except OSError:
            return None

    try:
        pool = np.memmap(
            str(data_path),
            dtype=np.dtype("<u8"),
            mode="r",
            shape=(meta["committed_count"],),
        )
    except (ValueError, OSError):
        return None

    if meta["committed_count"] == 0:
        return None

    # O(1) sanity checks — the first odd prime must be 3 and the last
    # must match what metadata records.
    try:
        if int(pool[0]) != 3:
            return None
        if int(pool[-1]) != meta["last_prime"]:
            return None
    except IndexError:
        return None

    return pool


# ── helpers ───────────────────────────────────────────────────────

def _prefix_view(pool: np.ndarray, limit: int) -> np.ndarray:
    """Return a read-only view of *pool* restricted to primes ≤ *limit*.

    The returned array is a plain ``np.ndarray`` view (not ``np.memmap``),
    so downstream code cannot accidentally close the underlying file mapping.
    """
    stop = int(np.searchsorted(pool, np.uint64(limit), side="right"))
    view = pool[:stop].view(np.ndarray)
    view.flags.writeable = False
    return view


def _close_mmap(pool: np.ndarray) -> None:
    """Release the underlying file mapping of a memmap, if any."""
    if isinstance(pool, np.memmap):
        mm = getattr(pool, "_mmap", None)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, OSError):
                pass


def _open_prefix(requested_limit: int) -> np.ndarray:
    """Read metadata, validate data, and return the requested prefix."""
    meta = _read_metadata()
    if meta is None:
        raise RuntimeError(
            "pool metadata missing after build — cannot open"
        )
    pool = _validate_and_open_data_file(meta)
    if pool is None:
        raise RuntimeError(
            "pool data file invalid after build — cannot open"
        )
    return _prefix_view(pool, requested_limit)


# ── build / extend ────────────────────────────────────────────────

def _rebuild_pool(limit: int) -> None:
    """Build the complete pool from scratch and commit metadata."""
    _POOL_DIR.mkdir(parents=True, exist_ok=True)
    data_path = _POOL_DIR / _DATA_NAME

    count = 0
    last_prime = 0

    with data_path.open("wb") as fh:
        for chunk in iter_odd_prime_chunks(3, limit):
            fh.write(chunk.tobytes())
            count += len(chunk)
            if len(chunk):
                last_prime = int(chunk[-1])
        fh.flush()
        os.fsync(fh.fileno())

    _write_metadata(
        committed_limit=limit,
        committed_count=count,
        last_prime=last_prime,
    )


def _extend_pool(meta: dict, limit: int) -> None:
    """Screen (*committed_limit*, *limit*] and append to the data file."""
    data_path = _POOL_DIR / _DATA_NAME
    committed_bytes = meta["committed_count"] * 8

    # Truncate any tail from a previously interrupted extension.
    with data_path.open("r+b") as fh:
        fh.truncate(committed_bytes)

    count = meta["committed_count"]
    last_prime = meta["last_prime"]

    with data_path.open("ab") as fh:
        for chunk in iter_odd_prime_chunks(
            meta["committed_limit"] + 1,
            limit,
        ):
            fh.write(chunk.tobytes())
            count += len(chunk)
            if len(chunk):
                last_prime = int(chunk[-1])
        fh.flush()
        os.fsync(fh.fileno())

    _write_metadata(
        committed_limit=limit,
        committed_count=count,
        last_prime=last_prime,
    )


# ── public entry point ────────────────────────────────────────────

def open_or_extend_prime_pool(limit: int) -> np.ndarray:
    """Return a read-only uint64 array of all odd primes ≤ *limit*.

    On first call for a given *limit* the pool is generated segment by
    segment and written to disk.  Subsequent calls (including with a
    smaller *limit*) reuse the existing file.  A larger *limit*
    triggers incremental extension.

    The returned array is a ``np.memmap`` view when a disk pool exists;
    it must not be written to.
    """
    meta = _read_metadata()

    if meta is not None:
        pool = _validate_and_open_data_file(meta)
        if pool is not None:
            if meta["committed_limit"] >= limit:
                # Already have enough — return prefix view (zero-copy).
                return _prefix_view(pool, limit)

            # Extend: release the current mapping, append new primes,
            # then re-open.
            _close_mmap(pool)
            _extend_pool(meta, limit)
            return _open_prefix(limit)

    # No usable pool on disk — build from scratch.
    _rebuild_pool(limit)
    return _open_prefix(limit)
