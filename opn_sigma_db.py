"""Persistent, validated cache for sigma-pool analysis records.

The database is a performance cache, never an authority.  Records are
checksummed and arithmetically validated before they can affect pruning.
Invalid or incompatible rows are ignored so the caller falls back to the
normal prime-pool scan.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import gmpy2
from gmpy2 import mpz


DATABASE_SCHEMA_VERSION = 1
SCAN_SEMANTICS_VERSION = 1
_COMMIT_BATCH_SIZE = 64


@dataclass(slots=True, frozen=True)
class PersistedSigmaRecord:
    """One validated exact or window-partial sigma analysis."""

    p: int
    exp: int
    exact: bool
    scanned_limit: int
    pool_digest: bytes
    valuations: Dict[int, int]
    residual: mpz


def _encode_valuations(valuations: Dict[int, int]) -> bytes:
    rows = [
        [int(q), int(exponent)]
        for q, exponent in sorted(valuations.items())
    ]
    return json.dumps(
        rows,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")


def _decode_valuations(payload: bytes) -> Dict[int, int]:
    rows = json.loads(payload.decode("ascii"))
    if not isinstance(rows, list):
        raise ValueError("sigma valuations must be a list")

    valuations: Dict[int, int] = {}
    previous = 1
    for row in rows:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not all(isinstance(value, int) for value in row)
        ):
            raise ValueError("malformed sigma valuation row")
        q, exponent = row
        if q < 3 or q % 2 == 0 or q <= previous:
            raise ValueError("sigma factors must be increasing odd integers")
        if exponent <= 0:
            raise ValueError("sigma valuations must be positive")
        if not gmpy2.is_prime(q):
            raise ValueError("sigma valuation key is not prime")
        valuations[q] = exponent
        previous = q
    return valuations


def _encode_mpz(value: mpz) -> bytes:
    integer = int(value)
    if integer < 1:
        raise ValueError("sigma residual must be positive")
    return integer.to_bytes(
        max(1, (integer.bit_length() + 7) // 8),
        "big",
    )


def _record_checksum(
    *,
    p: int,
    exp: int,
    exact: bool,
    scanned_limit: int,
    pool_digest: bytes,
    valuations_payload: bytes,
    residual_payload: bytes,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"opn-sigma-record-v1\0")
    digest.update(
        struct.pack(
            ">QIQB",
            int(p),
            int(exp),
            int(scanned_limit),
            int(bool(exact)),
        )
    )
    for payload in (
        pool_digest,
        valuations_payload,
        residual_payload,
    ):
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.digest()


def _validate_record(
    row,
    *,
    sigma_odd: mpz,
) -> PersistedSigmaRecord:
    (
        p,
        exp,
        scanned_limit,
        exact_raw,
        pool_digest,
        valuations_payload,
        residual_payload,
        checksum,
        semantics_version,
    ) = row

    if exact_raw not in (0, 1):
        raise ValueError("sigma record exact flag is invalid")
    exact = bool(exact_raw)
    pool_digest = bytes(pool_digest)
    valuations_payload = bytes(valuations_payload)
    residual_payload = bytes(residual_payload)
    checksum = bytes(checksum)

    if semantics_version != SCAN_SEMANTICS_VERSION:
        raise ValueError("unsupported sigma scan semantics")
    if p < 3 or p % 2 == 0 or exp < 1:
        raise ValueError("invalid sigma record key")
    if exact:
        if scanned_limit != 0 or pool_digest:
            raise ValueError("exact sigma record has window metadata")
    else:
        if scanned_limit < 3 or len(pool_digest) != 32:
            raise ValueError("partial sigma record lacks a pool certificate")

    expected_checksum = _record_checksum(
        p=p,
        exp=exp,
        exact=exact,
        scanned_limit=scanned_limit,
        pool_digest=pool_digest,
        valuations_payload=valuations_payload,
        residual_payload=residual_payload,
    )
    if checksum != expected_checksum:
        raise ValueError("sigma record checksum mismatch")

    valuations = _decode_valuations(valuations_payload)
    residual = mpz(int.from_bytes(residual_payload, "big"))

    if exact:
        if residual != 1:
            raise ValueError("exact sigma record has non-unit residual")
    else:
        if residual <= 1:
            raise ValueError("partial sigma record has unit residual")
        if any(q > scanned_limit for q in valuations):
            raise ValueError("partial sigma valuation exceeds scanned window")

    reconstructed = mpz(residual)
    for q, exponent in valuations.items():
        reconstructed *= mpz(q) ** exponent
    if reconstructed != sigma_odd:
        raise ValueError("sigma record fails the arithmetic identity")

    return PersistedSigmaRecord(
        p=int(p),
        exp=int(exp),
        exact=exact,
        scanned_limit=int(scanned_limit),
        pool_digest=pool_digest,
        valuations=valuations,
        residual=residual,
    )


class SigmaAnalysisDatabase:
    """SQLite-backed cache with bounded batched commits."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._connection = sqlite3.connect(str(self.path))
        self._pending_writes = 0

        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sigma_records (
                p INTEGER NOT NULL,
                exp INTEGER NOT NULL,
                scanned_limit INTEGER NOT NULL,
                exact INTEGER NOT NULL,
                pool_digest BLOB NOT NULL,
                valuations BLOB NOT NULL,
                residual BLOB NOT NULL,
                checksum BLOB NOT NULL,
                semantics_version INTEGER NOT NULL,
                PRIMARY KEY (p, exp, scanned_limit)
            )
            """
        )
        self._connection.execute(
            """
            CREATE INDEX IF NOT EXISTS sigma_records_lookup
            ON sigma_records (p, exp, exact DESC, scanned_limit DESC)
            """
        )

        row = self._connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(DATABASE_SCHEMA_VERSION),),
            )
            self._connection.commit()
        elif int(row[0]) != DATABASE_SCHEMA_VERSION:
            self._connection.close()
            raise ValueError(
                "unsupported sigma database schema version: "
                f"{row[0]}"
            )

    def load_candidates(
        self,
        p: int,
        exp: int,
        *,
        sigma_odd: mpz,
    ) -> tuple[list[PersistedSigmaRecord], int]:
        """Return arithmetically valid candidates and an invalid-row count."""
        rows = self._connection.execute(
            """
            SELECT
                p, exp, scanned_limit, exact, pool_digest,
                valuations, residual, checksum, semantics_version
            FROM sigma_records
            WHERE p=? AND exp=?
            ORDER BY exact DESC, scanned_limit DESC
            """,
            (int(p), int(exp)),
        ).fetchall()

        valid: list[PersistedSigmaRecord] = []
        invalid = 0
        for row in rows:
            try:
                valid.append(
                    _validate_record(
                        row,
                        sigma_odd=sigma_odd,
                    )
                )
            except (TypeError, ValueError, OverflowError, json.JSONDecodeError):
                invalid += 1
        return valid, invalid

    def store(
        self,
        *,
        p: int,
        exp: int,
        exact: bool,
        scanned_limit: int,
        pool_digest: bytes,
        valuations: Dict[int, int],
        residual: mpz,
        sigma_odd: mpz,
    ) -> None:
        """Validate and enqueue one cache record for a batched commit."""
        if exact:
            scanned_limit = 0
            pool_digest = b""

        valuations_payload = _encode_valuations(valuations)
        residual_payload = _encode_mpz(residual)
        checksum = _record_checksum(
            p=p,
            exp=exp,
            exact=exact,
            scanned_limit=scanned_limit,
            pool_digest=pool_digest,
            valuations_payload=valuations_payload,
            residual_payload=residual_payload,
        )

        row = (
            int(p),
            int(exp),
            int(scanned_limit),
            int(bool(exact)),
            sqlite3.Binary(pool_digest),
            sqlite3.Binary(valuations_payload),
            sqlite3.Binary(residual_payload),
            sqlite3.Binary(checksum),
            SCAN_SEMANTICS_VERSION,
        )
        _validate_record(row, sigma_odd=sigma_odd)

        if exact:
            self._connection.execute(
                "DELETE FROM sigma_records WHERE p=? AND exp=?",
                (int(p), int(exp)),
            )

        self._connection.execute(
            """
            INSERT OR REPLACE INTO sigma_records(
                p, exp, scanned_limit, exact, pool_digest,
                valuations, residual, checksum, semantics_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        self._pending_writes += 1
        if self._pending_writes >= _COMMIT_BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        if self._pending_writes:
            self._connection.commit()
            self._pending_writes = 0

    def close(self) -> None:
        self.flush()
        self._connection.close()
