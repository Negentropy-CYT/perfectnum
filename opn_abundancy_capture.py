"""Capture and report productive states with small positive abundancy gap.

The recorder is observability-only.  It never changes a search state, a
priority, or a pruning decision.  Raw JSONL is the authoritative output;
the CSV, text report, and shared sigma maps are derived from that stream.
"""

from __future__ import annotations

import csv
import heapq
import json
import os
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, TextIO

from gmpy2 import mpz

from opn_core import ratio_lower_bound
from opn_metrics import exact_headroom, exact_headroom_bucket


CAPTURE_SCHEMA_VERSION = 1
RAW_FILENAME = "abundancy_gap_states.jsonl"
INDEX_FILENAME = "abundancy_gap_index.csv"
TEXT_FILENAME = "abundancy_gap_top.txt"
SIGMA_MAPS_FILENAME = "abundancy_sigma_maps.json"
SUMMARY_FILENAME = "abundancy_gap_summary.json"
_FLUSH_EVERY = 128


@dataclass(frozen=True, slots=True)
class AbundancyCaptureConfig:
    """Configuration for the observability-only state recorder."""

    enabled: bool = True
    max_gap_num: int = 1
    max_gap_den: int = 100
    max_records: int = 50_000
    text_limit: int = 100

    def __post_init__(self) -> None:
        if self.max_gap_num <= 0 or self.max_gap_den <= 0:
            raise ValueError("abundancy-gap threshold must be positive")
        if self.max_records < 1:
            raise ValueError("abundancy capture record limit must be positive")
        if self.text_limit < 0:
            raise ValueError("abundancy text limit must not be negative")


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _factor_text(items: Iterable[tuple[int, int]]) -> str:
    parts = []
    for prime, exponent in items:
        parts.append(str(prime) if exponent == 1 else f"{prime}^{exponent}")
    return " * ".join(parts) if parts else "1"


def _record_pairs(record: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (int(prime), int(exponent))
        for prime, exponent in record["assigned"]
    ]


def _record_gap(record: dict[str, Any]) -> Fraction:
    return Fraction(
        int(record["gap_num"]),
        int(record["gap_den"]),
    )


def _decimal_ratio(numerator: int, denominator: int) -> str:
    with localcontext() as context:
        context.prec = 24
        return format(Decimal(numerator) / Decimal(denominator), ".18g")


class AbundancyGapRecorder:
    """Append-only capture stream with checkpoint-aligned recovery."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        target_num: int,
        target_den: int,
        resume_productive_ordinal: int,
        config: AbundancyCaptureConfig,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id)
        self.target_num = int(target_num)
        self.target_den = int(target_den)
        self.config = config
        self.raw_path = self.run_dir / RAW_FILENAME
        self.summary_path = self.run_dir / SUMMARY_FILENAME

        self.qualifying_seen = 0
        self.small_gap_seen = 0
        self.pending_overshoot_rejected = 0
        self.records_written = 0
        self.dropped_limit = 0
        self.bucket_counts: Counter[str] = Counter()
        self.errors: list[str] = []
        self.tail_repairs = 0
        self.last_record_ordinal = 0
        self.last_committed_ordinal = int(resume_productive_ordinal)
        self.last_committed_qualifying = 0
        self.last_committed_small_gap = 0
        self.last_committed_pending_rejected = 0
        self.last_committed_records = 0
        self.last_committed_dropped = 0
        self._since_flush = 0
        self._handle: TextIO | None = None
        self._active = bool(config.enabled)

        if not self._active:
            return

        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._load_and_repair(
                keep_through=int(resume_productive_ordinal),
            )
            self._restore_committed_counts(
                productive_ordinal=int(resume_productive_ordinal),
            )
            self._handle = self.raw_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            )
        except Exception as exc:
            self._disable(f"capture initialization failed: {exc}")

    @property
    def active(self) -> bool:
        return self._active and self._handle is not None

    def _disable(self, message: str) -> None:
        self.errors.append(str(message))
        self._active = False
        if self._handle is not None:
            try:
                self._handle.close()
            except Exception:
                pass
            self._handle = None

    def _load_and_repair(self, *, keep_through: int) -> None:
        if not self.raw_path.exists():
            return

        retained = 0
        buckets: Counter[str] = Counter()
        previous_ordinal = 0
        with self.raw_path.open("r+b") as handle:
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.truncate(line_start)
                    self.tail_repairs += 1
                    break
                try:
                    record = json.loads(line.decode("utf-8"))
                    ordinal = self._validate_record(record)
                except Exception as exc:
                    raise ValueError(
                        "abundancy JSONL contains an invalid complete line "
                        f"at byte {line_start}: {exc}"
                    ) from exc
                if ordinal <= previous_ordinal:
                    raise ValueError(
                        "abundancy JSONL ordinals are not strictly increasing"
                    )
                if ordinal > keep_through:
                    handle.truncate(line_start)
                    break
                previous_ordinal = ordinal
                retained += 1
                buckets[str(record["gap_bucket"])] += 1

        self.records_written = retained
        self.qualifying_seen = retained
        self.bucket_counts = buckets
        self.last_record_ordinal = previous_ordinal

    def _restore_committed_counts(self, *, productive_ordinal: int) -> None:
        if not self.summary_path.exists():
            self.small_gap_seen = self.qualifying_seen
            self.last_committed_qualifying = self.qualifying_seen
            self.last_committed_small_gap = self.small_gap_seen
            self.last_committed_records = self.records_written
            return
        try:
            document = json.loads(
                self.summary_path.read_text(encoding="utf-8")
            )
            if (
                document.get("run_id") == self.run_id
                and int(document.get("committed_productive_ordinal", -1))
                == productive_ordinal
            ):
                self.qualifying_seen = max(
                    self.records_written,
                    int(document.get("qualifying_states", 0)),
                )
                self.small_gap_seen = max(
                    self.qualifying_seen,
                    int(
                        document.get(
                            "small_gap_states_seen",
                            self.qualifying_seen,
                        )
                    ),
                )
                self.pending_overshoot_rejected = max(
                    0,
                    int(
                        document.get(
                            "pending_lower_bound_rejections",
                            0,
                        )
                    ),
                )
                self.dropped_limit = max(
                    0,
                    int(document.get("dropped_due_to_limit", 0)),
                )
                restored_buckets = document.get("gap_buckets")
                if isinstance(restored_buckets, dict):
                    self.bucket_counts = Counter(
                        {
                            str(bucket): int(count)
                            for bucket, count in restored_buckets.items()
                        }
                    )
        except Exception:
            # The raw stream remains authoritative.  A stale or damaged
            # progress summary must not disable otherwise valid capture.
            pass
        self.last_committed_qualifying = self.qualifying_seen
        self.last_committed_small_gap = self.small_gap_seen
        self.last_committed_pending_rejected = (
            self.pending_overshoot_rejected
        )
        self.last_committed_records = self.records_written
        self.last_committed_dropped = self.dropped_limit

    def _validate_record(self, record: Any) -> int:
        if not isinstance(record, dict):
            raise ValueError("record must be an object")
        if record.get("schema_version") != CAPTURE_SCHEMA_VERSION:
            raise ValueError("record schema version mismatch")
        if record.get("run_id") != self.run_id:
            raise ValueError("record run_id mismatch")
        ordinal = record.get("productive_ordinal")
        if not isinstance(ordinal, int) or ordinal < 1:
            raise ValueError("invalid productive ordinal")
        if not isinstance(record.get("depth"), int) or record["depth"] < 0:
            raise ValueError("invalid search depth")

        def valuation_rows(name: str) -> list[list[int]]:
            rows = record.get(name)
            if not isinstance(rows, list):
                raise ValueError(f"{name} must be a list")
            previous = 0
            for row in rows:
                if (
                    not isinstance(row, list)
                    or len(row) != 2
                    or not all(isinstance(value, int) for value in row)
                ):
                    raise ValueError(f"invalid {name} row")
                prime, value = row
                if prime <= previous or prime < 3 or prime % 2 == 0:
                    raise ValueError(f"invalid prime order in {name}")
                if value <= 0:
                    raise ValueError(f"invalid valuation in {name}")
                previous = prime
            return rows

        assigned_rows = record.get("assigned")
        if not isinstance(assigned_rows, list):
            raise ValueError("assigned must be a list")
        assigned_primes: set[int] = set()
        assigned_map: dict[int, int] = {}
        for row in assigned_rows:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not all(isinstance(value, int) for value in row)
            ):
                raise ValueError("invalid assigned row")
            prime, exponent = row
            if (
                prime < 3
                or prime % 2 == 0
                or exponent < 1
                or prime in assigned_primes
            ):
                raise ValueError("invalid assigned prime power")
            assigned_primes.add(prime)
            assigned_map[prime] = exponent

        euler_prime = record.get("euler_prime")
        if euler_prime is not None:
            if (
                not isinstance(euler_prime, int)
                or euler_prime not in assigned_map
                or euler_prime % 4 != 1
                or assigned_map[euler_prime] % 4 != 1
                or any(
                    prime != euler_prime and exponent % 2 != 0
                    for prime, exponent in assigned_map.items()
                )
            ):
                raise ValueError("invalid Euler component")
        elif any(exponent % 2 != 0 for exponent in assigned_map.values()):
            raise ValueError("odd exponent without an Euler component")

        pending = record.get("pending")
        if (
            not isinstance(pending, list)
            or not all(
                isinstance(prime, int)
                and prime >= 3
                and prime % 2 == 1
                for prime in pending
            )
            or len(pending) != len(set(pending))
            or assigned_primes.intersection(pending)
        ):
            raise ValueError("invalid pending-prime list")

        valuation_rows("required_v")
        current_rows = valuation_rows("current_v")
        if dict(current_rows) != assigned_map:
            raise ValueError("current valuations do not match assigned powers")

        integer_fields = ("next_idx", "excluded_count")
        if any(
            not isinstance(record.get(name), int) or record[name] < 0
            for name in integer_fields
        ):
            raise ValueError("invalid search-position field")

        exact_values: dict[str, int] = {}
        for name in ("ratio_num", "ratio_den", "gap_num", "gap_den"):
            raw_value = record.get(name)
            if not isinstance(raw_value, str):
                raise ValueError(f"{name} must be a decimal string")
            try:
                value = int(raw_value)
            except ValueError as exc:
                raise ValueError(f"{name} is not an integer") from exc
            if str(value) != raw_value or value <= 0:
                raise ValueError(f"{name} is not a canonical positive integer")
            exact_values[name] = value

        expected_gap_num, expected_gap_den = exact_headroom(
            ratio_num=exact_values["ratio_num"],
            ratio_den=exact_values["ratio_den"],
            target_num=self.target_num,
            target_den=self.target_den,
        )
        if (
            exact_values["gap_num"] != expected_gap_num
            or exact_values["gap_den"] != expected_gap_den
        ):
            raise ValueError("stored abundancy gap fails exact reconstruction")
        if (
            expected_gap_num * self.config.max_gap_den
            > expected_gap_den * self.config.max_gap_num
        ):
            raise ValueError("stored abundancy gap exceeds capture interval")
        expected_bucket = exact_headroom_bucket(
            ratio_num=exact_values["ratio_num"],
            ratio_den=exact_values["ratio_den"],
            target_num=self.target_num,
            target_den=self.target_den,
        )
        if record.get("gap_bucket") != expected_bucket:
            raise ValueError("stored abundancy-gap interval is inconsistent")
        return ordinal

    def capture(self, state: Any, productive_ordinal: int) -> None:
        """Record one qualifying productive state; never raise to search."""
        if not self.active:
            return
        try:
            ratio_num = int(state.ratio_num)
            ratio_den = int(state.ratio_den)
            gap_num, gap_den = exact_headroom(
                ratio_num=ratio_num,
                ratio_den=ratio_den,
                target_num=self.target_num,
                target_den=self.target_den,
            )
            if gap_num <= 0:
                return
            if (
                gap_num * self.config.max_gap_den
                > gap_den * self.config.max_gap_num
            ):
                return

            self.small_gap_seen += 1
            live_pending = {
                int(prime)
                for prime in state.pending
                if prime not in state.assigned
            }
            lower_num, lower_den = ratio_lower_bound(
                mpz(ratio_num),
                mpz(ratio_den),
                live_pending,
            )
            if (
                lower_num * self.target_den
                > self.target_num * lower_den
            ):
                self.pending_overshoot_rejected += 1
                return

            self.qualifying_seen += 1
            bucket = exact_headroom_bucket(
                ratio_num=ratio_num,
                ratio_den=ratio_den,
                target_num=self.target_num,
                target_den=self.target_den,
            )
            self.bucket_counts[bucket] += 1
            if self.records_written >= self.config.max_records:
                self.dropped_limit += 1
                return

            ordinal = int(productive_ordinal)
            if ordinal <= self.last_record_ordinal:
                raise ValueError(
                    "productive ordinal did not increase in capture stream"
                )

            record = {
                "schema_version": CAPTURE_SCHEMA_VERSION,
                "run_id": self.run_id,
                "productive_ordinal": ordinal,
                "depth": int(state.depth),
                "assigned": [
                    [int(prime), int(exponent)]
                    for prime, exponent in state.assigned.items()
                ],
                "euler_prime": (
                    int(state.euler_prime)
                    if state.euler_prime is not None
                    else None
                ),
                "pending": [int(prime) for prime in state.pending],
                "required_v": [
                    [int(prime), int(value)]
                    for prime, value in sorted(state.required_v.items())
                ],
                "current_v": [
                    [int(prime), int(value)]
                    for prime, value in sorted(state.current_v.items())
                ],
                "ratio_num": str(ratio_num),
                "ratio_den": str(ratio_den),
                "gap_num": str(gap_num),
                "gap_den": str(gap_den),
                "gap_bucket": bucket,
                "next_idx": int(state.next_idx),
                "excluded_count": len(state.excluded),
            }
            assert self._handle is not None
            self._handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            self._handle.write("\n")
            self.records_written += 1
            self.last_record_ordinal = ordinal
            self._since_flush += 1
            if self._since_flush >= _FLUSH_EVERY:
                self._handle.flush()
                self._since_flush = 0
        except Exception as exc:
            self._disable(f"capture write failed: {exc}")

    def _progress_summary(
        self,
        *,
        status: str,
        committed_productive_ordinal: int,
        derived: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        document: dict[str, Any] = {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": status,
            "configuration": {
                "enabled": self.config.enabled,
                "target_num": self.target_num,
                "target_den": self.target_den,
                "max_gap_num": self.config.max_gap_num,
                "max_gap_den": self.config.max_gap_den,
                "max_records": self.config.max_records,
                "text_limit": self.config.text_limit,
            },
            "committed_productive_ordinal": int(
                committed_productive_ordinal
            ),
            "qualifying_states": self.qualifying_seen,
            "small_gap_states_seen": self.small_gap_seen,
            "pending_lower_bound_rejections": (
                self.pending_overshoot_rejected
            ),
            "records_written": self.records_written,
            "dropped_due_to_limit": self.dropped_limit,
            "truncated": self.dropped_limit > 0,
            "gap_buckets": dict(sorted(self.bucket_counts.items())),
            "tail_repairs": self.tail_repairs,
            "errors": list(self.errors),
            "complete": not self.errors and self.dropped_limit == 0,
        }
        if derived is not None:
            document["derived_outputs"] = derived
        return document

    def commit(self, productive_ordinal: int) -> None:
        """Durably align the capture stream with a stable search frontier."""
        if not self.config.enabled:
            return
        if self.active:
            try:
                assert self._handle is not None
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except Exception as exc:
                self._disable(f"capture checkpoint flush failed: {exc}")
        self.last_committed_ordinal = int(productive_ordinal)
        self.last_committed_qualifying = self.qualifying_seen
        self.last_committed_small_gap = self.small_gap_seen
        self.last_committed_pending_rejected = (
            self.pending_overshoot_rejected
        )
        self.last_committed_records = self.records_written
        self.last_committed_dropped = self.dropped_limit
        try:
            _atomic_json(
                self.summary_path,
                self._progress_summary(
                    status="RUNNING",
                    committed_productive_ordinal=self.last_committed_ordinal,
                ),
            )
        except Exception as exc:
            self._disable(f"capture progress summary failed: {exc}")

    def _close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.flush()
            self._handle.close()
        finally:
            self._handle = None

    def _repair_trailing_partial_line(self) -> None:
        """Remove only a final non-newline-terminated JSONL fragment."""
        if not self.raw_path.exists():
            return
        with self.raw_path.open("r+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size == 0:
                return
            handle.seek(size - 1)
            if handle.read(1) == b"\n":
                return
            position = size - 1
            while position > 0:
                position -= 1
                handle.seek(position)
                if handle.read(1) == b"\n":
                    handle.truncate(position + 1)
                    self.tail_repairs += 1
                    return
            handle.truncate(0)
            self.tail_repairs += 1

    def _rollback_to_committed(self) -> None:
        self._close()
        if not self.raw_path.exists():
            return
        try:
            self._load_and_repair(
                keep_through=self.last_committed_ordinal,
            )
            self.qualifying_seen = max(
                self.records_written,
                self.last_committed_qualifying,
            )
            self.small_gap_seen = max(
                self.qualifying_seen,
                self.last_committed_small_gap,
            )
            self.pending_overshoot_rejected = (
                self.last_committed_pending_rejected
            )
            self.dropped_limit = self.last_committed_dropped
        except Exception as exc:
            self._disable(f"capture rollback failed: {exc}")

    def finalize(
        self,
        *,
        status: str,
        sigma_database_path: str | Path | None,
    ) -> dict[str, Any]:
        """Close capture and build deterministic derived reports."""
        if not self.config.enabled:
            document = self._progress_summary(
                status=status,
                committed_productive_ordinal=(
                    self.last_committed_ordinal
                ),
            )
            try:
                _atomic_json(self.summary_path, document)
            except Exception:
                pass
            return document

        if status in {"INTERRUPTED", "FAILED"}:
            self._rollback_to_committed()
        else:
            self._close()
            self.last_committed_ordinal = max(
                self.last_committed_ordinal,
                self.last_record_ordinal,
            )
            try:
                self._repair_trailing_partial_line()
            except Exception as exc:
                self.errors.append(
                    f"capture trailing-line repair failed: {exc}"
                )

        derived: dict[str, Any]
        try:
            derived = _write_derived_outputs(
                self.run_dir,
                run_id=self.run_id,
                raw_path=self.raw_path,
                text_limit=self.config.text_limit,
                sigma_database_path=sigma_database_path,
                record_validator=self._validate_record,
                target_num=self.target_num,
                target_den=self.target_den,
            )
            if derived["raw_records"] != self.records_written:
                raise ValueError(
                    "raw capture count does not match recorder state"
                )
        except Exception as exc:
            self.errors.append(f"derived report generation failed: {exc}")
            derived = {
                "complete": False,
                "error": str(exc),
            }

        document = self._progress_summary(
            status=status,
            committed_productive_ordinal=self.last_committed_ordinal,
            derived=derived,
        )
        if "funnel" in derived:
            document["funnel"] = derived["funnel"]
        try:
            _atomic_json(self.summary_path, document)
        except Exception as exc:
            self.errors.append(f"final capture summary failed: {exc}")
            document["errors"] = list(self.errors)
            document["complete"] = False
        return document


def _read_records(
    raw_path: Path,
    *,
    run_id: str,
    record_validator=None,
) -> Iterable[dict[str, Any]]:
    if not raw_path.exists():
        return
    with raw_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"raw capture has an incomplete line {line_number}"
                )
            record = json.loads(line)
            if (
                record.get("schema_version") != CAPTURE_SCHEMA_VERSION
                or record.get("run_id") != run_id
            ):
                raise ValueError(
                    f"raw capture metadata mismatch at line {line_number}"
                )
            if record_validator is not None:
                record_validator(record)
            yield record


def _push_top_record(
    heap: list[tuple[Fraction, int, dict[str, Any]]],
    record: dict[str, Any],
    *,
    limit: int,
) -> None:
    if limit <= 0:
        return
    gap = _record_gap(record)
    ordinal = int(record["productive_ordinal"])
    entry = (-gap, -ordinal, record)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    worst_gap = -heap[0][0]
    worst_ordinal = -heap[0][1]
    if (gap, ordinal) < (worst_gap, worst_ordinal):
        heapq.heapreplace(heap, entry)


def _write_index_and_collect(
    run_dir: Path,
    *,
    run_id: str,
    raw_path: Path,
    text_limit: int,
    record_validator=None,
    target_num: int,
    target_den: int,
) -> tuple[
    int,
    list[dict[str, Any]],
    set[tuple[int, int]],
    dict[str, Any],
]:
    target = run_dir / INDEX_FILENAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    top_heap: list[tuple[Fraction, int, dict[str, Any]]] = []
    sigma_pairs: set[tuple[int, int]] = set()
    count = 0

    # funnel stats (merged from _compute_funnel_statistics)
    from collections import Counter as _C
    euler_dist: _C[int | str] = _C()
    assigned_dist: _C[int] = _C()
    pending_dist: _C[int] = _C()
    debt_dist: _C[int] = _C()
    unique_assigned: set[tuple[tuple[int, int], ...]] = set()
    unique_states: set[tuple] = set()
    min_raw_num: int | None = None
    min_raw_den: int | None = None
    min_adj_num: int | None = None
    min_adj_den: int | None = None

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "productive_ordinal",
            "gap_bucket",
            "gap_num",
            "gap_den",
            "gap_decimal",
            "ratio_num",
            "ratio_den",
            "ratio_decimal",
            "depth",
            "assigned_count",
            "euler_prime",
            "pending_count",
            "factorization",
        ])
        for record in _read_records(
            raw_path,
            run_id=run_id,
            record_validator=record_validator,
        ):
            count += 1
            pairs = _record_pairs(record)
            sigma_pairs.update(pairs)
            gap_num = int(record["gap_num"])
            gap_den = int(record["gap_den"])
            ratio_num = int(record["ratio_num"])
            ratio_den = int(record["ratio_den"])
            writer.writerow([
                record["productive_ordinal"],
                record["gap_bucket"],
                record["gap_num"],
                record["gap_den"],
                _decimal_ratio(gap_num, gap_den),
                record["ratio_num"],
                record["ratio_den"],
                _decimal_ratio(ratio_num, ratio_den),
                record["depth"],
                len(pairs),
                (
                    record["euler_prime"]
                    if record["euler_prime"] is not None
                    else ""
                ),
                len(record["pending"]),
                _factor_text(sorted(pairs)),
            ])
            _push_top_record(
                top_heap,
                record,
                limit=text_limit,
            )

            # funnel statistics (merged from _compute_funnel_statistics)
            ep = record.get("euler_prime")
            euler_dist[ep if ep is not None else "none"] += 1
            assigned_dist[len(pairs)] += 1
            pending_dist[len(record["pending"])] += 1

            req_v: dict[int, int] = {
                int(p): int(v) for p, v in record.get("required_v", [])
            }
            cur_v: dict[int, int] = {
                int(p): int(v) for p, v in record.get("current_v", [])
            }
            debt = 0
            for q, req in req_v.items():
                cur = cur_v.get(q, 0)
                if req > cur:
                    debt += req - cur
            debt_dist[debt] += 1

            canonical_assigned = tuple(sorted((int(p), int(e)) for p, e in pairs))
            unique_assigned.add(canonical_assigned)
            unique_states.add((
                canonical_assigned,
                ep,
                tuple(sorted(record["pending"])),
                tuple(sorted(
                    (int(p), int(v)) for p, v in record.get("required_v", [])
                )),
                tuple(sorted(
                    (int(p), int(v)) for p, v in record.get("current_v", [])
                )),
            ))

            if min_raw_num is None or gap_num * min_raw_den < min_raw_num * gap_den:
                min_raw_num = gap_num
                min_raw_den = gap_den

            num = mpz(record["ratio_num"])
            den = mpz(record["ratio_den"])
            for q in record["pending"]:
                num *= q + 1
                den *= q
            adj_num = target_num * den - target_den * num
            adj_den = target_den * int(den)
            if min_adj_num is None or adj_num * min_adj_den < min_adj_num * adj_den:
                min_adj_num = int(adj_num)
                min_adj_den = adj_den

    os.replace(temporary, target)
    top_records = [entry[2] for entry in top_heap]
    top_records.sort(
        key=lambda record: (
            _record_gap(record),
            int(record["productive_ordinal"]),
        )
    )

    funnel: dict[str, Any] = {
        "records_scanned": count,
        "unique_assigned_factorizations": len(unique_assigned),
        "unique_complete_states": len(unique_states),
    }
    if min_raw_num is not None:
        funnel["min_raw_gap"] = {"num": min_raw_num, "den": min_raw_den}
    if min_adj_num is not None:
        funnel["min_pending_adjusted_gap"] = {
            "num": min_adj_num, "den": min_adj_den,
        }
    if euler_dist:
        funnel["euler_prime_distribution"] = dict(sorted(euler_dist.items()))
    if assigned_dist:
        funnel["assigned_count_distribution"] = dict(sorted(assigned_dist.items()))
    if pending_dist:
        funnel["pending_count_distribution"] = dict(sorted(pending_dist.items()))
    if debt_dist:
        funnel["valuation_debt_distribution"] = dict(sorted(debt_dist.items()))

    return count, top_records, sigma_pairs, funnel


def _load_sigma_maps(
    sigma_pairs: set[tuple[int, int]],
    *,
    sigma_database_path: str | Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    maps: dict[str, dict[str, Any]] = {}
    status: dict[str, Any] = {
        "database_available": False,
        "records": 0,
        "missing": [],
        "invalid_rows": 0,
        "error": None,
    }
    if sigma_database_path is None:
        status["error"] = "sigma database is disabled"
        return maps, status
    path = Path(sigma_database_path)
    if not path.exists():
        status["error"] = "sigma database file is unavailable"
        return maps, status

    database = None
    try:
        from opn_core import sigma_prime_power
        from opn_sigma_db import SigmaAnalysisDatabase

        database = SigmaAnalysisDatabase(path)
        status["database_available"] = True
        for prime, exponent in sorted(sigma_pairs):
            sigma_value = mpz(sigma_prime_power(prime, exponent))
            odd_sigma = mpz(sigma_value)
            v2 = 0
            while odd_sigma % 2 == 0:
                odd_sigma //= 2
                v2 += 1
            candidates, invalid = database.load_candidates(
                prime,
                exponent,
                sigma_odd=odd_sigma,
            )
            status["invalid_rows"] += invalid
            exact = next(
                (record for record in candidates if record.exact),
                None,
            )
            if exact is None:
                status["missing"].append(f"{prime}:{exponent}")
                continue
            reconstructed = mpz(1)
            for q, valuation in exact.valuations.items():
                reconstructed *= mpz(q) ** valuation
            if reconstructed != odd_sigma:
                raise ArithmeticError(
                    "validated sigma record failed reconstruction"
                )
            key = f"{prime}:{exponent}"
            maps[key] = {
                "prime": prime,
                "exponent": exponent,
                "sigma": str(sigma_value),
                "v2": v2,
                "odd_valuations": [
                    [int(q), int(valuation)]
                    for q, valuation in exact.valuations.items()
                ],
            }
        status["records"] = len(maps)
    except Exception as exc:
        status["error"] = str(exc)
    finally:
        if database is not None:
            try:
                database.close()
            except Exception:
                pass
    return maps, status


def _sigma_factorization(mapping: dict[str, Any]) -> str:
    factors: list[tuple[int, int]] = []
    if int(mapping["v2"]) > 0:
        factors.append((2, int(mapping["v2"])))
    factors.extend(
        (int(prime), int(exponent))
        for prime, exponent in mapping["odd_valuations"]
    )
    return _factor_text(factors)


def _write_text_report(
    run_dir: Path,
    *,
    records: list[dict[str, Any]],
    sigma_maps: dict[str, dict[str, Any]],
) -> None:
    target = run_dir / TEXT_FILENAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        w = handle.write
        w("Productive partial states with small positive abundancy gap\n")
        w("=" * 72 + "\n\n")
        w("For each recorded partial factorization S:\n")
        w("  I(S) = sigma(S) / S\n")
        w("  0 < 2 - I(S) <= 10^-2\n\n")
        w(
            "States for which the mandatory pending-prime lower bound "
            "already exceeds 2 are excluded.\n\n"
        )
        w(
            "These are productive partial search states, not odd-perfect-"
            "number solutions or complete candidates.\n"
        )
        w(
            "The q-adic difference is "
            "v_q(product sigma(p^a)) - v_q(S) for assigned p^a.\n\n"
        )

        for rank, record in enumerate(records, 1):
            pairs = _record_pairs(record)
            assigned = dict(pairs)
            pending = {int(prime) for prime in record["pending"]}
            required = {
                int(prime): int(value)
                for prime, value in record["required_v"]
            }
            current = {
                int(prime): int(value)
                for prime, value in record["current_v"]
            }
            ratio = Fraction(
                int(record["ratio_num"]),
                int(record["ratio_den"]),
            )
            gap = _record_gap(record)

            w(f"State rank {rank:04d}\n")
            w("-" * 72 + "\n")
            w(
                f"productive ordinal: {record['productive_ordinal']}\n"
                f"depth:              {record['depth']}\n"
                f"assigned factors:   {len(pairs)}\n"
                f"gap interval:        {record['gap_bucket']}\n"
            )
            w("\nPartial factorization\n")
            w(f"  S = {_factor_text(sorted(pairs))}\n")
            euler_prime = record["euler_prime"]
            if euler_prime is None:
                w("  Euler component: not assigned\n")
            else:
                w(
                    "  Euler component: "
                    f"{euler_prime}^{assigned[int(euler_prime)]}\n"
                )

            w("\nAssignment order\n")
            for index, (prime, exponent) in enumerate(pairs, 1):
                suffix = "  [Euler component]" if prime == euler_prime else ""
                w(f"  {index:>2}. {prime}^{exponent}{suffix}\n")

            w("\nAbundancy index\n")
            w(f"  I(S) exact:      {ratio.numerator} / {ratio.denominator}\n")
            w(
                "  I(S) decimal:    "
                f"{_decimal_ratio(ratio.numerator, ratio.denominator)}\n"
            )
            w(f"  2 - I(S) exact:  {gap.numerator} / {gap.denominator}\n")
            w(
                "  2 - I(S) decimal:"
                f"  {_decimal_ratio(gap.numerator, gap.denominator)}\n"
            )

            w("\nSigma-factor relations\n")
            for prime, exponent in pairs:
                key = f"{prime}:{exponent}"
                mapping = sigma_maps.get(key)
                if mapping is None:
                    w(
                        f"  sigma({prime}^{exponent}): "
                        "validated factorization unavailable\n"
                    )
                    continue
                w(
                    f"  sigma({prime}^{exponent}) = {mapping['sigma']}"
                    f" = {_sigma_factorization(mapping)}\n"
                )
                for q, valuation in mapping["odd_valuations"]:
                    q = int(q)
                    if q in assigned:
                        relation = "already assigned in S"
                    elif q in pending:
                        relation = "unassigned required prime factor"
                    else:
                        relation = "recorded in sigma factorization"
                    w(f"    {q}^{valuation}: {relation}\n")

            w("\nUnassigned required prime factors\n")
            if record["pending"]:
                w("  " + ", ".join(map(str, record["pending"])) + "\n")
            else:
                w("  none\n")

            w("\nq-adic valuations\n")
            w(
                f"  {'q':>12} {'v_q(S)':>10} "
                f"{'v_q(product sigma)':>19} {'difference':>12}  meaning\n"
            )
            for q in sorted(set(required) | set(current)):
                current_value = current.get(q, 0)
                required_value = required.get(q, 0)
                difference = required_value - current_value
                if difference == 0:
                    meaning = "equal in the current partial state"
                elif difference > 0:
                    meaning = "S-side valuation must increase"
                else:
                    meaning = "future sigma factors must contribute"
                w(
                    f"  {q:>12} {current_value:>10} "
                    f"{required_value:>19} {difference:>12}  {meaning}\n"
                )

            w("\nSearch position\n")
            w(
                f"  next prime index: {record['next_idx']}\n"
                f"  excluded count:  {record['excluded_count']}\n\n"
            )
            w(
                "The valuation comparison describes only this partial state; "
                "it does not establish that the state can be completed.\n\n"
            )
            w("=" * 72 + "\n\n")
    os.replace(temporary, target)


def _write_derived_outputs(
    run_dir: Path,
    *,
    run_id: str,
    raw_path: Path,
    text_limit: int,
    sigma_database_path: str | Path | None,
    record_validator=None,
    target_num: int,
    target_den: int,
) -> dict[str, Any]:
    count, top_records, sigma_pairs, funnel = _write_index_and_collect(
        run_dir,
        run_id=run_id,
        raw_path=raw_path,
        text_limit=text_limit,
        record_validator=record_validator,
        target_num=target_num,
        target_den=target_den,
    )
    sigma_maps, sigma_status = _load_sigma_maps(
        sigma_pairs,
        sigma_database_path=sigma_database_path,
    )
    _atomic_json(
        run_dir / SIGMA_MAPS_FILENAME,
        {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "run_id": run_id,
            "status": sigma_status,
            "records": sigma_maps,
        },
    )
    _write_text_report(
        run_dir,
        records=top_records,
        sigma_maps=sigma_maps,
    )
    return {
        "complete": True,
        "raw_records": count,
        "text_records": len(top_records),
        "sigma_maps": sigma_status,
        "funnel": funnel,
        "files": [
            RAW_FILENAME,
            INDEX_FILENAME,
            TEXT_FILENAME,
            SIGMA_MAPS_FILENAME,
            SUMMARY_FILENAME,
        ],
    }
# ponytail: _compute_funnel_statistics removed — merged into
# _write_index_and_collect to eliminate the second JSONL scan.
